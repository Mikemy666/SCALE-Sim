"""DATE3 execution path for PIVOT-CA using DATE2 Bank/mapping primitives."""

from __future__ import annotations

import csv
import hashlib
import json
from math import ceil
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scalesim.memory.chunk_residency import WeightChunk, _intersection_cycles
from scalesim.memory.date3_ep_model import (
    RouteReplica, localize_detailed_npu,
)
from scalesim.memory.date3_ep_system import build_ep_system_timeline
from scalesim.memory.memdomain_experiment import workload_digest
from scalesim.memory.memdomain_runner import (
    RunnerConfig, _bank_metrics, _compute_interference,
    compiled_dynamic_config, compiler_bank_service_cycles,
    critical_path_miss_stalls, _pressure_snapshot, load_runner_config,
    offchip_load_cycles, is_nonstationary_multilayer,
)
from scalesim.memory.pivot_ca_prefetch import (
    CandidateTile, CoverageAccuracyConstrainedPrefetchPolicy,
    CoverageAccuracyPolicyConfig, PrefetchCandidate, PrefetchQualityStats, TileLifetime,
    quality_from_lifetimes,
)
from scalesim.memory.streaming_residency import (
    StreamingLoadPlan, StreamingResidencyEngine,
)
from scalesim.memory.unified_bank_domain import UnifiedBankDomain
from scalesim.memory.virtual_bank_mapping import BankPressure, VirtualBankMappingTable


POLICY_NAME = "PIVOT-CA"

_IMPLEMENTATION_FILES = (
    "scalesim/memory/pivot_ca_runner.py",
    "scalesim/memory/pivot_ca_prefetch.py",
    "scalesim/memory/date3_ep_model.py",
    "scalesim/memory/date3_ep_system.py",
    "scalesim/memory/memdomain_runner.py",
    "scalesim/memory/memdomain_experiment.py",
    "scalesim/memory/buckyball_compiler.py",
    "scalesim/memory/buckyball_memdomain.py",
    "scalesim/memory/prefetch_policy.py",
    "scalesim/memory/streaming_residency.py",
)


def _canonical_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def implementation_digest() -> str:
    """Bind resumable DATE3 outputs to the simulator implementation."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _workload_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: payload[key] for key in (
            "chunks", "compute_cycles", "compute_intervals", "compute_requests",
            "hardware", "system", "topology_provenance", "workload_name",
        ) if key in payload
    }


@dataclass(frozen=True)
class QualityEpochRow:
    epoch_id: int
    required_bytes: int
    prefetched_bytes: int
    useful_timely_bytes: int
    late_bytes: int
    unused_bytes: int
    evicted_before_use_bytes: int
    coverage: Optional[float]
    accuracy: Optional[float]
    coverage_valid: bool
    accuracy_valid: bool
    coverage_ema: Optional[float]
    accuracy_ema: Optional[float]
    baseline_coverage_ema: Optional[float]
    baseline_accuracy_ema: Optional[float]


@dataclass(frozen=True)
class PivotExecution:
    summary: Mapping[str, object]
    decisions: Tuple[Mapping[str, object], ...]
    epochs: Tuple[QualityEpochRow, ...]
    plans: Tuple[StreamingLoadPlan, ...]
    routes: Tuple[RouteReplica, ...]
    local_workload: Mapping[str, object]
    peer_workloads: Tuple[Mapping[str, object], ...]
    ep_timeline: Tuple[Mapping[str, object], ...]
    combine_rows: Tuple[Mapping[str, object], ...]
    guard_rows: Tuple[Mapping[str, object], ...]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plans_to_lifetimes(
    plans: Sequence[StreamingLoadPlan], results, required_ids: Sequence[str],
) -> Tuple[TileLifetime, ...]:
    by_plan = {item.chunk.chunk_id: item for item in plans}
    required = set(required_ids)
    lifetimes = []
    for result in results:
        plan = by_plan[result.chunk_id]
        prefetched = result.planned_kind == "prefetch"
        lifetimes.append(TileLifetime(
            tile_id=result.chunk_id,
            size_bytes=plan.chunk.size_bytes,
            first_required_cycle=(plan.chunk.use_cycle if result.chunk_id in required else None),
            prefetch_issue_cycle=(result.actual_issue_cycle if prefetched else None),
            prefetch_complete_cycle=(result.completion_cycle if prefetched else None),
            first_use_cycle=(result.first_use_cycle if result.chunk_id in required else None),
            release_cycle=result.release_cycle,
            eviction_cycle=result.eviction_cycle,
        ))
    return tuple(lifetimes)


def _run_plans(config: RunnerConfig, plans: Sequence[StreamingLoadPlan],
               compute_requests=None, mapping_policy="least_occupied"):
    mapping = VirtualBankMappingTable(config.resources, mapping_policy)
    domain = UnifiedBankDomain(config.resources, config.interleave_bytes)
    compute = config.compute_requests if compute_requests is None else tuple(compute_requests)
    compute_only = domain.simulate(compute)
    pressure = {
        bank: BankPressure(
            compute_only.per_bank_max_queue_depth[bank],
            compute_only.per_bank_busy_cycles[bank],
            compute_only.per_bank_conflicts[bank],
        ) for bank in range(config.resources.bank_count)
    }
    report = StreamingResidencyEngine(domain, mapping).run(
        plans, compute, pressure,
        dynamic_compute_mapping=not config.dynamic_honor_preferred_banks,
        bind_prefetched_weight_reads=bool(
            config.payload.get("multi_layer_prefetch", {}).get("enabled")
        ),
    )
    return report, compute_only, mapping.statistics()


def _epoch_groups(config: RunnerConfig, width: int):
    groups = []
    by_stage = {}
    for chunk in config.chunks:
        by_stage.setdefault((chunk.expert_id, chunk.ffn_part), []).append(chunk)
    for identity in sorted(by_stage):
        chunks = sorted(
            by_stage[identity], key=lambda item: (item.use_cycle, item.tile_id)
        )
        for start in range(0, len(chunks), width):
            groups.append(tuple(chunks[start:start + width]))
    groups.sort(key=lambda items: (
        items[0].use_cycle, items[0].expert_id, items[0].ffn_part,
        items[0].tile_id,
    ))
    return tuple(groups)


def _atomic_pivot_config(config: RunnerConfig) -> RunnerConfig:
    """Recover an invariant one-tile stream for runtime Chunk coalescing.

    Fixed-PF controls are generated with physical 1/2/4/8-tile requests.  A
    PIVOT run must not inherit that offline seed as its immutable request
    granularity.  This function splits those configured requests back into
    architectural weight tiles before the online controller groups them.
    """
    provenance = config.payload.get("topology_provenance", {})
    tile_size = int(provenance.get("tile_size", 16))
    weight_bytes = int(provenance.get("weight_bytes_per_element", 1))
    atomic_bytes = tile_size * tile_size * weight_bytes
    if atomic_bytes <= 0:
        raise ValueError("invalid atomic weight-tile size")

    by_stage = {}
    for item in config.chunks:
        by_stage.setdefault((item.expert_id, item.ffn_part), []).append(item)
    atomic = []
    for identity in sorted(by_stage):
        source = sorted(
            by_stage[identity], key=lambda item: (item.use_cycle, item.tile_id)
        )
        for index, item in enumerate(source):
            pieces = int(ceil(item.size_bytes / atomic_bytes))
            if pieces == 1:
                atomic.append(item)
                continue
            if index + 1 < len(source):
                gap = max(1, source[index + 1].use_cycle - item.use_cycle)
                step = max(1, int(round(gap / pieces)))
            elif index:
                gap = max(1, item.use_cycle - source[index - 1].use_cycle)
                step = max(1, int(round(gap / pieces)))
            else:
                step = max(1, tile_size * 2)
            for offset in range(pieces):
                size = min(atomic_bytes, item.size_bytes - offset * atomic_bytes)
                atomic.append(WeightChunk(
                    chunk_id=f"{item.chunk_id}__tile{offset}",
                    expert_id=item.expert_id,
                    ffn_part=item.ffn_part,
                    tile_id=item.tile_id * pieces + offset,
                    size_bytes=size,
                    use_cycle=max(0, item.use_cycle - (pieces - 1 - offset) * step),
                    logical_address=item.logical_address + offset * atomic_bytes,
                    bank_group_size=item.bank_group_size,
                ))
    return replace(config, chunks=tuple(sorted(
        atomic, key=lambda item: (
            item.use_cycle, item.expert_id, item.ffn_part, item.tile_id
        )
    )))


def _coalesce_epoch(
    tiles: Sequence[WeightChunk], chunk_tiles: int, decision_id: int,
    label: str, stable_order: bool = False,
) -> Tuple[WeightChunk, ...]:
    """Partition one invariant atomic epoch into true variable-size Chunks."""
    if chunk_tiles <= 0:
        return ()
    groups = []
    for group_id, start in enumerate(range(0, len(tiles), chunk_tiles)):
        members = tuple(tiles[start:start + chunk_tiles])
        first = members[0]
        decision_tag = f"{decision_id:04d}" if stable_order else str(decision_id)
        group_tag = f"{group_id:04d}" if stable_order else str(group_id)
        groups.append(WeightChunk(
            chunk_id=(
                f"pivot_d{decision_tag}_{label}_g{group_tag}_"
                f"e{first.expert_id}_ff{first.ffn_part}_c{chunk_tiles}"
            ),
            expert_id=first.expert_id,
            ffn_part=first.ffn_part,
            tile_id=first.tile_id,
            size_bytes=sum(item.size_bytes for item in members),
            # A coalesced request must be ready when its first constituent
            # tile is consumed.
            use_cycle=min(item.use_cycle for item in members),
            logical_address=first.logical_address,
            bank_group_size=max(item.bank_group_size for item in members),
        ))
    return tuple(groups)


def _prefix_memory_cost(config: RunnerConfig, report, compute_only) -> int:
    """Comparable completed-prefix cost used by the online incumbent guard."""
    demand, late = critical_path_miss_stalls(
        report.chunks,
        multi_layer=is_nonstationary_multilayer(config),
    )
    bank = max((item.queue_wait_cycles for item in compute_only.services), default=0)
    mapping = sum(min(item.mapping_latency_cycles, item.miss_stall_cycles)
                  for item in report.chunks)
    full_services = {
        item.request_id: item for item in report.memory_report.services
    }
    interference = max((
        max(0, full_services[item.request_id].completion_cycle
            - item.completion_cycle)
        for item in compute_only.services if item.request_id in full_services
    ), default=0)
    return int(bank + demand + late + mapping + interference)


def run_pivot_ca(config: RunnerConfig) -> PivotExecution:
    input_config = config
    input_payload = config.payload
    detailed_workload = localize_detailed_npu(config)
    localized = detailed_workload.config
    annotations = localized.payload.get("coverage_accuracy_policy", {})
    speculative = set(annotations.get("speculative_chunk_ids", ()))
    evict = set(annotations.get("evict_before_use_chunk_ids", ()))
    forced_late = set(annotations.get("forced_late_chunk_ids", ()))
    # Fault-injection unit cases retain their per-object annotations.  Normal
    # paper runs use an invariant atomic stream and true runtime coalescing.
    granularity_mode = not (speculative or evict or forced_late)
    pivot_input = (
        _atomic_pivot_config(localized) if granularity_mode else localized
    )
    config = compiled_dynamic_config(
        pivot_input, prefetch_coordinated=True
    )
    policy_payload = config.payload.get("coverage_accuracy_policy", {})
    policy_config = CoverageAccuracyPolicyConfig.from_mapping(policy_payload)
    if config.payload.get("policy", {}).get("prefetch_policy") not in (
        None, "coverage_accuracy_constrained"
    ):
        raise ValueError("DATE3 runner requires coverage_accuracy_constrained policy")
    policy = CoverageAccuracyConstrainedPrefetchPolicy(policy_config)
    multi_layer = config.payload.get("multi_layer_prefetch", {})
    multi_layer_enabled = bool(multi_layer.get("enabled"))
    experts_per_layer = int(multi_layer.get("experts_per_layer", 0))
    layer_visibility = {
        int(item["layer_id"]): int(item["start_cycle"])
        for item in config.payload.get("topology_provenance", {}).get(
            "layer_profiles", ()
        )
    }
    # One online decision covers a bounded scheduling epoch, then partitions
    # that same atomic-tile horizon with each candidate granularity.  Using a
    # horizon larger than the largest Chunk both exposes temporal variation
    # and avoids an O(number_of_tiles^2) prefix-resimulation path.
    max_epoch_tiles = (
        max(policy_config.candidate_chunks) ** 2
        if granularity_mode else max(policy_config.candidate_chunks)
    )
    epochs = _epoch_groups(config, max_epoch_tiles)
    dynamic_weight_pools = dict(config.dynamic_weight_bank_pools)
    stage_weight_pools = {}
    for item in config.chunks:
        stage_weight_pools.setdefault((item.expert_id, item.ffn_part), set()).update(
            dynamic_weight_pools.get(item.chunk_id, config.static_weight_banks)
        )
    stage_weight_pools = {
        key: tuple(sorted(value)) for key, value in stage_weight_pools.items()
    }
    ordered_demand = tuple(sorted(
        config.chunks, key=lambda item: (item.use_cycle, item.chunk_id)
    ))
    fixed_issue = {}
    for index, chunk in enumerate(ordered_demand):
        trigger = index - config.prefetch_window
        fixed_issue[chunk.chunk_id] = (
            0 if trigger < 0 else ordered_demand[trigger].use_cycle
        )
    all_plans: List[StreamingLoadPlan] = []
    detail_rows: List[Mapping[str, object]] = []
    epoch_rows: List[QualityEpochRow] = []
    selected_actions = []
    fallback_reasons = Counter()
    online_guard_count = 0
    online_guard_saved = 0
    admission_rejection_count = 0
    guard_rows = []
    previous_decision_cycle = 0
    gaps = [
        right.use_cycle - left.use_cycle
        for left, right in zip(config.chunks, config.chunks[1:])
        if right.use_cycle > left.use_cycle
    ]
    tile_compute = max(1, int(mean(gaps)) if gaps else 1)

    for epoch_id, chunks in enumerate(epochs, 1):
        # Routing/stage metadata exposes the upcoming weight stream before the
        # preceding epoch is fully consumed.  Waiting until ``previous_end``
        # serialized decisions and made every later large-Chunk proposal late.
        # Decisions are monotonic but may overlap execution of an earlier
        # epoch, as a real asynchronous prefetch controller does.
        layer_id = (
            chunks[0].expert_id // experts_per_layer
            if experts_per_layer else 0
        )
        # The router materializes the complete token-to-expert list at the
        # beginning of a MoE layer.  Every bounded epoch in that layer is
        # therefore schedulable from this route-visibility point.  Deriving
        # visibility from the currently selected Window made the controller
        # learn about a request only shortly before its use and erased the
        # principal advantage of online layer-aware prefetch scheduling.
        visibility_cycle = (
            layer_visibility[layer_id]
            if layer_id in layer_visibility
            else max(
                0,
                chunks[0].use_cycle
                - max(policy_config.candidate_windows) * tile_compute,
            )
        )
        cycle = max(previous_decision_cycle, visibility_cycle)
        current_compute = tuple(
            request for request in config.compute_requests
            if request.issue_cycle <= cycle
        )
        if all_plans or current_compute:
            prefix, _, _ = _run_plans(config, all_plans, current_compute)
            snapshot_pressure = _pressure_snapshot(
                prefix.memory_report, cycle, config.resources.bank_count,
                horizon=max(policy_config.candidate_windows) * tile_compute,
            )
        else:
            snapshot_pressure = {
                bank: BankPressure() for bank in range(config.resources.bank_count)
            }
        per_bank_capacity = config.resources.capacity_bytes // config.resources.bank_count
        snapshot = __import__(
            "scalesim.memory.prefetch_policy", fromlist=["BankSnapshot"]
        ).BankSnapshot(
            cycle, snapshot_pressure,
            {bank: per_bank_capacity for bank in range(config.resources.bank_count)},
            {bank: per_bank_capacity for bank in range(config.resources.bank_count)},
            {bank: config.resources.request_buffer_depth
             for bank in range(config.resources.bank_count)},
        )
        candidate_tiles = tuple(CandidateTile(
            item.chunk_id, item.size_bytes, item.use_cycle,
            0.35 if item.chunk_id in speculative else 1.0,
            max(1.0, item.size_bytes / config.resources.bandwidth_bytes_per_cycle),
        ) for item in chunks)
        leads = {
            window: window * tile_compute
            for window in policy_config.candidate_windows
        }
        prior_action = (
            policy.state.current_chunk, policy.state.current_window,
            policy.state.current_bank_group,
        )
        layer_name = (
            f"L{chunks[0].expert_id // experts_per_layer}"
            if experts_per_layer else config.workload_name
        )
        # Reconstruct the shared-HBM reservation cursor from already committed
        # requests.  The next local decision must cover this backlog as well
        # as its own bytes; otherwise every epoch independently assumes an
        # idle HBM channel and selects a Window that is too short globally.
        hbm_cursor = 0
        if multi_layer_enabled:
            for prior in sorted(
                all_plans,
                key=lambda item: (
                    item.issue_cycle, item.chunk.use_cycle, item.chunk.chunk_id
                ),
            ):
                if prior.offchip_latency_cycles <= 0:
                    continue
                hbm_cursor = (
                    max(hbm_cursor, prior.issue_cycle)
                    + prior.offchip_latency_cycles
                )
        hbm_backlog = max(0, hbm_cursor - cycle)
        chosen, rows = policy.choose(
            cycle=cycle, layer=layer_name,
            expert=chunks[0].expert_id, stage=chunks[0].ffn_part,
            tiles=candidate_tiles, snapshot=snapshot, lead_cycles=leads,
            # Prefetch timeliness is limited by the shared off-chip link, not
            # by the much wider on-chip Bank fabric.  Using the latter made
            # the controller predict that W=16 was timely even though the
            # measured HBM transfer required roughly W=256, so an apparently
            # adaptive decision still emitted every request too late.
            bandwidth_bytes_per_cycle=(
                float(config.payload["hardware"].get(
                    "offchip_bandwidth_bits_per_cycle", 8
                )) / 8.0
                if multi_layer_enabled
                else config.resources.bandwidth_bytes_per_cycle
            ),
            setup_cycles=float(config.payload["hardware"].get("offchip_startup_cycles", 0)),
            mapping_cycles=float(config.mapping_overhead_per_object),
            # The action selects a candidate Bank *pool*.  Individual DATE2
            # Tiles still retain their own bank_group_size when the mapping
            # table allocates within that pool.
            group_size=min(
                config.resources.bank_count,
                max(
                    max(item.bank_group_size for item in chunks),
                    len(config.static_weight_banks),
                    int(ceil(
                        # PIVOT maps the whole bounded online epoch, not just
                        # one coalesced request.  Sizing this pool from a
                        # single Chunk silently recreated the old five-Weight-
                        # Bank boundary and caused later requests to fall back
                        # to demand after the pool filled.  The unified domain
                        # may borrow enough Banks for the live epoch; measured
                        # pressure/interference and the incumbent guard still
                        # reject an expansion when it hurts the critical path.
                        (
                            sum(item.size_bytes for item in chunks)
                            if multi_layer_enabled else
                            max(item.size_bytes for item in chunks)
                            * max(policy_config.candidate_chunks)
                        )
                        / max(1.0, policy_config.max_residency_ratio
                              * per_bank_capacity)
                    )),
                ),
            ),
            chunk_granularity=granularity_mode,
            hbm_backlog_cycles=float(hbm_backlog),
        )
        decision_rows = [row.to_dict() for row in rows]
        epoch_plan_start = len(all_plans)
        lead = leads.get(chosen.window, 0)
        adaptive_chunks = (
            _coalesce_epoch(
                chunks, chosen.chunk_size, epoch_id, "adaptive",
                stable_order=multi_layer_enabled,
            )
            if granularity_mode else tuple(chunks)
        )
        selected_count = (
            len(adaptive_chunks) if granularity_mode
            else max(0, min(chosen.chunk_size, len(chunks)))
        )
        epoch_hbm_cursor = hbm_cursor
        for index, chunk in enumerate(adaptive_chunks):
            use = granularity_mode or chunk.chunk_id not in speculative
            prefetched = (
                chosen.chunk_size > 0
                and (granularity_mode or index < selected_count)
            )
            if not use and not prefetched:
                continue
            issue = max(cycle, chunk.use_cycle - lead) if prefetched else chunk.use_cycle
            if prefetched and chunk.chunk_id in forced_late:
                issue = max(cycle, chunk.use_cycle - 1)
            transfer_cycles = offchip_load_cycles(config, chunk.size_bytes)
            if multi_layer_enabled and prefetched:
                hbm_start = max(issue, epoch_hbm_cursor)
                per_bank_bandwidth = (
                    config.resources.bandwidth_bytes_per_cycle
                    / config.resources.bank_count
                )
                ingress_cycles = int(ceil(
                    chunk.size_bytes
                    / max(1.0, len(chosen.bank_group) * per_bank_bandwidth)
                ))
                predicted_ready = (
                    hbm_start + transfer_cycles
                    + config.mapping_overhead_per_object + ingress_cycles
                )
                if predicted_ready >= chunk.use_cycle:
                    # Do not inject a prefetch already known to be late.  The
                    # same runtime Chunk remains a legal demand-coalesced
                    # request, so admission changes timing, not useful bytes.
                    prefetched = False
                    issue = chunk.use_cycle
                    admission_rejection_count += 1
                else:
                    epoch_hbm_cursor = hbm_start + transfer_cycles
            preferred_banks = (
                chosen.bank_group if prefetched else
                stage_weight_pools.get(
                    (chunk.expert_id, chunk.ffn_part), ()
                )
            )
            all_plans.append(StreamingLoadPlan(
                chunk, issue, "prefetch" if prefetched else "demand",
                preferred_banks,
                config.mapping_overhead_per_object,
                will_use=use,
                eviction_cycle=(
                    chunk.use_cycle - 1
                    if prefetched and chunk.chunk_id in evict else None
                ),
                unused_release_cycle=(chunk.use_cycle if not use else None),
                offchip_latency_cycles=transfer_cycles,
            ))

        # Shadow fixed follows the same known demand sequence in a separate
        # simulator instance; its requests never enter the real prefix report.
        shadow_plans = []
        # Exact Dynamic-FixedPF incumbent: same complete demand sequence,
        # fixed look-ahead issue rule, and per-stage compiler weight pool.
        # The former four-chunk shadow was weaker than the public control and
        # could not protect PIVOT against it.
        shadow_chunks = (
            _coalesce_epoch(
                chunks, policy_config.reference_chunk, epoch_id, "fixed",
                stable_order=multi_layer_enabled,
            ) if granularity_mode else tuple(chunks)
        )
        shadow_count = len(shadow_chunks)
        for index, chunk in enumerate(shadow_chunks):
            use = granularity_mode or chunk.chunk_id not in speculative
            pf = True
            if not use and not pf:
                continue
            shadow_issue = (
                max(cycle, chunk.use_cycle - leads.get(
                    policy_config.reference_window, 0
                )) if granularity_mode
                else fixed_issue[chunk.chunk_id]
            ) if pf else chunk.use_cycle
            if pf and chunk.chunk_id in forced_late:
                shadow_issue = max(cycle, chunk.use_cycle - 1)
            shadow_plans.append(StreamingLoadPlan(
                chunk,
                shadow_issue,
                "prefetch" if pf else "demand",
                (stage_weight_pools.get(
                    (chunk.expert_id, chunk.ffn_part), config.static_weight_banks
                ) if granularity_mode else dynamic_weight_pools.get(
                    chunk.chunk_id, config.static_weight_banks
                ))
                if pf else (),
                will_use=use,
                eviction_cycle=(
                    chunk.use_cycle - 1
                    if pf and chunk.chunk_id in evict else None
                ),
                unused_release_cycle=chunk.use_cycle if not use else None,
                offchip_latency_cycles=offchip_load_cycles(
                    config, chunk.size_bytes
                ),
            ))
        # Window=0 Coalesced-Demand is a legal PIVOT action: it performs no
        # early fetch but retains the runtime Chunk coalescer.  It is distinct
        # from the public NoPF control, whose architectural demand request is
        # one atomic Weight tile and which has no PIVOT Chunk engine.
        noprefetch_chunks = (
            _coalesce_epoch(
                chunks, policy_config.reference_chunk, epoch_id,
                "coalesced_demand", stable_order=multi_layer_enabled,
            ) if granularity_mode else tuple(chunks)
        )
        noprefetch_plans = [
            StreamingLoadPlan(
                chunk, chunk.use_cycle, "demand",
                (stage_weight_pools.get(
                    (chunk.expert_id, chunk.ffn_part), ()
                ) if granularity_mode else dynamic_weight_pools.get(
                    chunk.chunk_id, ()
                )), will_use=True,
                offchip_latency_cycles=offchip_load_cycles(
                    config, chunk.size_bytes
                ),
            )
            for chunk in noprefetch_chunks
            if granularity_mode or chunk.chunk_id not in speculative
        ]
        epoch_end = max(item.use_cycle for item in chunks)
        prefix_compute = tuple(
            request for request in config.compute_requests
            if request.issue_cycle <= epoch_end
        )
        prefix, prefix_compute_only, _ = _run_plans(
            config, all_plans, prefix_compute
        )
        proposal_cost = _prefix_memory_cost(config, prefix, prefix_compute_only)
        incumbent_cost = proposal_cost
        fixed_cost = proposal_cost
        noprefetch_cost = proposal_cost
        incumbent_action = "adaptive"
        applied_kind = "adaptive"
        if policy_config.online_incumbent_guard:
            fixed_plans = all_plans[:epoch_plan_start] + shadow_plans
            fixed, fixed_compute_only, _ = _run_plans(
                config, fixed_plans, prefix_compute
            )
            fixed_cost = _prefix_memory_cost(config, fixed, fixed_compute_only)
            demand_plans = all_plans[:epoch_plan_start] + noprefetch_plans
            noprefetch, noprefetch_compute_only, _ = _run_plans(
                config, demand_plans, prefix_compute
            )
            noprefetch_cost = _prefix_memory_cost(
                config, noprefetch, noprefetch_compute_only
            )
            candidates = (
                (proposal_cost, "adaptive", all_plans[epoch_plan_start:], prefix),
                (fixed_cost, "fixed_incumbent", shadow_plans, fixed),
                (noprefetch_cost, "coalesced_demand_incumbent", noprefetch_plans,
                 noprefetch),
            )
            incumbent_cost, incumbent_action, incumbent_plans, incumbent = min(
                candidates, key=lambda item: item[0]
            )
            if incumbent_cost < proposal_cost:
                applied_kind = incumbent_action
                online_guard_count += 1
                online_guard_saved += proposal_cost - incumbent_cost
                all_plans[epoch_plan_start:] = incumbent_plans
                prefix = incumbent
                if applied_kind == "fixed_incumbent":
                    applied_count = (
                        policy_config.reference_chunk
                        if granularity_mode else shadow_count
                    )
                    applied_window = policy_config.reference_window
                    applied_group = (
                        stage_weight_pools.get(
                            (chunks[0].expert_id, chunks[0].ffn_part),
                            config.static_weight_banks,
                        ) if granularity_mode else tuple(sorted({
                            bank for chunk in chunks
                            for bank in dynamic_weight_pools.get(
                                chunk.chunk_id, config.static_weight_banks
                            )
                        }))
                    )
                elif applied_kind == "coalesced_demand_incumbent":
                    applied_count = policy_config.reference_chunk
                    applied_window = 0
                    applied_group = stage_weight_pools.get(
                        (chunks[0].expert_id, chunks[0].ffn_part),
                        config.static_weight_banks,
                    )
                else:
                    applied_count = 0
                    applied_window = 0
                    applied_group = ()
                applied = PrefetchCandidate(
                    applied_count, applied_window, applied_group,
                    chosen.predicted_coverage, chosen.predicted_accuracy,
                    0.0, chosen.predicted_occupancy, chosen.predicted_conflict,
                    chosen.pressure, 0.0, chosen.score, True,
                    f"online_incumbent_guard:{applied_kind}",
                )
                proposed_action = (
                    policy.state.current_chunk, policy.state.current_window,
                    policy.state.current_bank_group,
                )
                actual_action = (
                    applied.chunk_size, applied.window, applied.bank_group,
                )
                if proposed_action != prior_action:
                    policy.adaptation_count -= 1
                if actual_action != prior_action:
                    policy.adaptation_count += 1
                policy.state.current_chunk = applied.chunk_size
                policy.state.current_window = applied.window
                policy.state.current_bank_group = applied.bank_group
                policy.state.current_score = applied.score
                chosen = applied
                for row in decision_rows:
                    row["selected"] = False
                template = dict(decision_rows[-1]) if decision_rows else {}
                template.update({
                    "current_chunk": applied.chunk_size,
                    "current_window": applied.window,
                    "current_bank_group": ":".join(map(str, applied.bank_group)),
                    "candidate_chunk": applied.chunk_size,
                    "candidate_window": applied.window,
                    "candidate_bank_group": ":".join(map(str, applied.bank_group)),
                    "selected": True,
                    "fallback_used": True,
                    "fallback_level": 4,
                    "fallback_reason": f"online_incumbent_guard:{applied_kind}",
                    "rejection_reason": f"online_incumbent_guard:{applied_kind}",
                })
                decision_rows.append(template)
        guard_rows.append({
            "epoch_id": epoch_id,
            "expert_id": chunks[0].expert_id,
            "ffn_part": chunks[0].ffn_part,
            "proposal_prefix_cost_cycles": proposal_cost,
            "fixed_prefix_cost_cycles": fixed_cost,
            "noprefetch_prefix_cost_cycles": noprefetch_cost,
            "incumbent_prefix_cost_cycles": incumbent_cost,
            "applied_prefix_cost_cycles": min(proposal_cost, incumbent_cost),
            "applied_action": applied_kind,
            "incumbent_action": incumbent_action,
            "saved_prefix_cycles": max(0, proposal_cost - incumbent_cost),
        })
        detail_rows.extend(decision_rows)
        selected_actions.append(chosen)
        applied_epoch_plans = all_plans[epoch_plan_start:]
        epoch_ids = {item.chunk.chunk_id for item in applied_epoch_plans}
        epoch_results = tuple(item for item in prefix.chunks if item.chunk_id in epoch_ids)
        required_ids = tuple(
            item.chunk.chunk_id for item in applied_epoch_plans if item.will_use
        )
        stats = quality_from_lifetimes(
            _plans_to_lifetimes(
                [item for item in all_plans if item.chunk.chunk_id in epoch_ids],
                epoch_results, required_ids,
            )
        )
        shadow, _, _ = _run_plans(config, shadow_plans, prefix_compute)
        shadow_required_ids = tuple(
            item.chunk.chunk_id for item in shadow_plans if item.will_use
        )
        shadow_stats = quality_from_lifetimes(_plans_to_lifetimes(
            shadow_plans, shadow.chunks, shadow_required_ids,
        ))
        prefetch_results = [item for item in epoch_results if item.planned_kind == "prefetch"]
        timing_errors = [
            item.completion_cycle - item.use_cycle for item in prefetch_results
        ]
        policy.update_feedback(
            stats, baseline=shadow_stats,
            mean_pressure=chosen.pressure,
            occupancy_byte_cycles=sum(
                next(plan.chunk.size_bytes for plan in all_plans
                     if plan.chunk.chunk_id == item.chunk_id)
                * max(0, item.release_cycle - item.actual_issue_cycle)
                for item in prefetch_results
            ),
            mean_timing_error=mean(timing_errors) if timing_errors else 0.0,
        )
        state = policy.state
        epoch_rows.append(QualityEpochRow(
            epoch_id, stats.required_bytes, stats.prefetched_bytes,
            stats.useful_timely_bytes, stats.late_bytes, stats.unused_bytes,
            stats.evicted_before_use_bytes, stats.coverage, stats.accuracy,
            stats.coverage_valid, stats.accuracy_valid, state.coverage_ema,
            state.accuracy_ema, state.baseline_coverage_ema,
            state.baseline_accuracy_ema,
        ))
        selected_detail = next(
            (row for row in reversed(decision_rows) if row.get("selected")), None
        )
        if selected_detail and selected_detail.get("fallback_used"):
            fallback_reasons[str(selected_detail.get("fallback_reason"))] += 1
        previous_decision_cycle = cycle

    final, compute_only, mapping_stats = _run_plans(config, all_plans)
    reference_plans = []
    for epoch_id, chunks in enumerate(epochs, 1):
        reference_chunks = (
            _coalesce_epoch(
                chunks, policy_config.reference_chunk, epoch_id, "reference",
                stable_order=multi_layer_enabled,
            ) if granularity_mode else tuple(chunks)
        )
        for chunk in reference_chunks:
            use = granularity_mode or chunk.chunk_id not in speculative
            prefetched = True
            if not use and not prefetched:
                continue
            reference_issue = (
                max(0, chunk.use_cycle - leads.get(
                    policy_config.reference_window, 0
                )) if granularity_mode
                else fixed_issue[chunk.chunk_id]
            ) if prefetched else chunk.use_cycle
            if prefetched and chunk.chunk_id in forced_late:
                reference_issue = max(0, chunk.use_cycle - 1)
            reference_plans.append(StreamingLoadPlan(
                chunk,
                reference_issue,
                "prefetch" if prefetched else "demand",
                (stage_weight_pools.get(
                    (chunk.expert_id, chunk.ffn_part), config.static_weight_banks
                ) if granularity_mode else dynamic_weight_pools.get(
                    chunk.chunk_id, config.static_weight_banks
                ))
                if prefetched else (),
                will_use=use,
                eviction_cycle=(
                    chunk.use_cycle - 1
                    if prefetched and chunk.chunk_id in evict else None
                ),
                unused_release_cycle=chunk.use_cycle if not use else None,
                offchip_latency_cycles=offchip_load_cycles(
                    config, chunk.size_bytes
                ),
            ))
    reference, reference_compute_only, _ = _run_plans(
        config, reference_plans
    )
    required_ids = tuple(
        plan.chunk.chunk_id for plan in all_plans if plan.will_use
    )
    quality = quality_from_lifetimes(
        _plans_to_lifetimes(all_plans, final.chunks, required_ids)
    )
    by_chunk = {item.chunk.chunk_id: item.chunk for item in (
        list(all_plans) + list(reference_plans)
    )}
    prefetches = [item for item in final.chunks if item.planned_kind == "prefetch"]
    multi_layer_stalls = is_nonstationary_multilayer(config)
    demand_stall, late_stall = critical_path_miss_stalls(
        final.chunks, multi_layer=multi_layer_stalls,
    )
    components = {
        "compute_cycles": config.compute_cycles,
        "bank_stall_cycles": (
            compiler_bank_service_cycles(config)
            + max((service.queue_wait_cycles for service in compute_only.services),
                  default=0)
        ),
        "weight_load_stall_cycles": demand_stall,
        "prefetch_miss_stall_cycles": late_stall,
        "prefetch_interference_stall_cycles": _compute_interference(
            config, final.memory_report, compute_only),
        "mapping_overhead_cycles": sum(
            min(item.mapping_latency_cycles, item.miss_stall_cycles)
            for item in final.chunks),
        "communication_stall_cycles": 0,
        "other_stall_cycles": 0,
    }
    detailed_ready = sum(components.values())
    system_timeline = build_ep_system_timeline(
        input_config, detailed_workload, detailed_ready
    )
    components["communication_stall_cycles"] = (
        system_timeline.exposed_remote_wait_cycles
    )
    components["other_stall_cycles"] = system_timeline.combine_cycles
    total = sum(components.values())
    if total != system_timeline.total_cycles:
        raise AssertionError("DATE3 EP critical path and additive total disagree")
    reference_demand, reference_late = critical_path_miss_stalls(
        reference.chunks, multi_layer=multi_layer_stalls,
    )
    reference_bank = (
        compiler_bank_service_cycles(config)
        + max(
            (service.queue_wait_cycles for service in reference_compute_only.services),
            default=0,
        )
    )
    reference_interference = _compute_interference(
        config, reference.memory_report, reference_compute_only
    )
    reference_memory_stall = (
        reference_bank + reference_demand + reference_late
        + reference_interference
    )
    actual_memory_stall = (
        components["bank_stall_cycles"] + components["weight_load_stall_cycles"]
        + components["prefetch_miss_stall_cycles"]
        + components["prefetch_interference_stall_cycles"]
    )
    reference_detailed_ready = config.compute_cycles + reference_memory_stall
    reference_system_timeline = build_ep_system_timeline(
        input_config, detailed_workload, reference_detailed_ready
    )
    reference_total = reference_system_timeline.total_cycles
    chunks_selected = [item.chunk_size for item in selected_actions]
    windows_selected = [item.window for item in selected_actions]
    groups_selected = {item.bank_group for item in selected_actions if item.bank_group}
    transfer_intervals = [
        (service.issue_cycle, service.completion_cycle)
        for service in final.memory_report.services
        if service.request_id.startswith("load:")
    ]
    hbm_active = [
        item for item in final.chunks if item.offchip_latency_cycles > 0
    ]
    hbm_span = (
        max(item.hbm_complete_cycle for item in hbm_active)
        - min(item.hbm_issue_cycle for item in hbm_active)
        if hbm_active else 0
    )
    bank = _bank_metrics(final.memory_report)
    occupancy = sum(
        by_chunk[item.chunk_id].size_bytes
        * max(0, item.release_cycle - item.actual_issue_cycle)
        for item in prefetches
    )
    late_ratio = quality.late_bytes / quality.prefetched_bytes if quality.prefetched_bytes else 0.0
    timely_ratio = quality.useful_timely_bytes / quality.prefetched_bytes if quality.prefetched_bytes else 0.0
    summary = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "workload_name": config.workload_name,
        "layer_count": int(multi_layer.get("layer_count", 1)),
        "controller_state_persistent": bool(
            multi_layer.get("controller_state") == "persistent_across_layers"
        ),
        "policy_name": POLICY_NAME,
        "chunk_semantics": (
            "runtime_tiles_per_request" if granularity_mode
            else "legacy_prefetch_degree"
        ),
        "atomic_tile_bytes": min(item.size_bytes for item in config.chunks),
        "ep_schema_version": 1,
        "global_expert_count": detailed_workload.contract.num_experts,
        "num_npus": detailed_workload.contract.num_npus,
        "detailed_npu_id": detailed_workload.contract.detailed_npu_id,
        "top_k": detailed_workload.contract.top_k,
        "local_expert_count": len(detailed_workload.local_experts),
        "active_local_expert_count": len(detailed_workload.active_local_experts),
        "local_route_replicas": detailed_workload.local_route_replicas,
        "remote_route_replicas": detailed_workload.remote_route_replicas,
        "dispatch_bytes": system_timeline.dispatch_bytes,
        "dispatch_cycles": system_timeline.dispatch_cycles,
        "return_bytes": system_timeline.return_bytes,
        "return_cycles": system_timeline.return_cycles,
        "detailed_ready_cycle": system_timeline.detailed_ready_cycle,
        "peer_ready_cycle": system_timeline.peer_ready_cycle,
        "result_ready_cycle": system_timeline.result_ready_cycle,
        "combine_cycles": system_timeline.combine_cycles,
        "decision_count": policy.decision_count,
        "adaptation_count": policy.adaptation_count,
        "selected_chunk_mean": mean(chunks_selected),
        "selected_chunk_min": min(chunks_selected),
        "selected_chunk_max": max(chunks_selected),
        "selected_window_mean": mean(windows_selected),
        "selected_window_min": min(windows_selected),
        "selected_window_max": max(windows_selected),
        "selected_bank_group_count": len(groups_selected),
        **asdict(quality),
        "coverage_ema_final": policy.state.coverage_ema,
        "accuracy_ema_final": policy.state.accuracy_ema,
        "baseline_coverage_ema_final": policy.state.baseline_coverage_ema,
        "baseline_accuracy_ema_final": policy.state.baseline_accuracy_ema,
        "occupancy_byte_cycles": occupancy,
        "late_prefetch_ratio": late_ratio,
        "timely_prefetch_ratio": timely_ratio,
        "fallback_count": sum(fallback_reasons.values()),
        "fallback_rate": sum(fallback_reasons.values()) / max(1, policy.decision_count),
        "fallback_reason_summary": json.dumps(fallback_reasons, sort_keys=True),
        "online_incumbent_guard_count": online_guard_count,
        "online_incumbent_guard_rate": online_guard_count / max(1, policy.decision_count),
        "online_incumbent_guard_saved_prefix_cycles": online_guard_saved,
        "admission_rejection_count": admission_rejection_count,
        "mean_bank_pressure": mean(item.pressure for item in selected_actions),
        "predicted_latency_benefit": sum(
            item.predicted_latency_benefit for item in selected_actions),
        "actual_memory_stall_reduction": reference_memory_stall - actual_memory_stall,
        "reference_fixed_total_cycles": reference_total,
        **components,
        "total_cycles": total,
        **bank,
        "prefetch_requests": len(prefetches),
        "prefetch_bytes": sum(by_chunk[item.chunk_id].size_bytes for item in prefetches),
        "compute_transfer_overlap_cycles": _intersection_cycles(
            config.compute_intervals, transfer_intervals),
        "mapping_count": mapping_stats.mapping_count,
        "mapping_failures": mapping_stats.allocation_failures,
        "peak_occupied_bytes": mapping_stats.peak_occupied_bytes,
        "hbm_queue_wait_cycles": final.hbm_queue_wait_cycles,
        "hbm_service_cycles": final.hbm_service_cycles,
        "hbm_busy_cycles": final.hbm_busy_cycles,
        "hbm_max_queue_depth": final.hbm_max_queue_depth,
        "hbm_utilization": min(
            1.0,
            final.hbm_busy_cycles / hbm_span if hbm_span else 0.0,
        ),
        "workload_hash": _canonical_hash(_workload_payload(input_payload)),
        "config_hash": _canonical_hash(input_payload),
        "implementation_hash": implementation_digest(),
        "shadow_real_request_count": 0,
    }
    return PivotExecution(
        summary, tuple(detail_rows), tuple(epoch_rows), tuple(all_plans),
        detailed_workload.routes, detailed_workload.summary_row(),
        system_timeline.peer_rows, system_timeline.timeline_rows,
        system_timeline.combine_rows,
        tuple(guard_rows),
    )


def run_pivot_ca_file(config_path: Path, output_dir: Path) -> PivotExecution:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    execution = run_pivot_ca(load_runner_config(config_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", [execution.summary])
    _write_csv(output_dir / "decision_detail.csv", list(execution.decisions))
    _write_csv(output_dir / "quality_epochs.csv", [asdict(item) for item in execution.epochs])
    _write_csv(output_dir / "ep_routes.csv", [item.to_dict() for item in execution.routes])
    _write_csv(output_dir / "ep_local_workload.csv", [execution.local_workload])
    _write_csv(output_dir / "ep_peer_workloads.csv", list(execution.peer_workloads))
    _write_csv(output_dir / "ep_timeline.csv", list(execution.ep_timeline))
    _write_csv(output_dir / "ep_return_combine.csv", list(execution.combine_rows))
    _write_csv(output_dir / "online_incumbent_guard.csv", list(execution.guard_rows))
    metadata = {
        "config_path": str(config_path),
        "config_hash": execution.summary["config_hash"],
        "workload_hash": execution.summary["workload_hash"],
        "policy_name": POLICY_NAME,
        "implementation_hash": execution.summary["implementation_hash"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return execution
