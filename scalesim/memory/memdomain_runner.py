"""End-to-end runner for the canonical MemDomain baseline matrix."""

from __future__ import annotations

import json
import heapq
from dataclasses import dataclass, replace
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from scalesim.memory.chunk_residency import ChunkResidencyManager, WeightChunk, _intersection_cycles
from scalesim.memory.memdomain_experiment import (
    Baseline,
    ExperimentRow,
    derive_selected_row,
    validate_matrix,
    workload_digest,
    write_matrix,
)
from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.streaming_residency import (
    StreamingChunkResult, StreamingLoadPlan, StreamingResidencyEngine,
)
from scalesim.memory.prefetch_policy import (
    BankAwarePrefetchPolicy,
    BankSnapshot,
    NaivePrefetchPolicy,
    NoPrefetchPolicy,
    PrefetchAction,
    PrefetchDecision,
)
from scalesim.memory.unified_bank_domain import (
    UnifiedBankDomain, UnifiedDomainReport, UnifiedMemoryRequest,
)
from scalesim.memory.virtual_bank_mapping import (
    BankPressure,
    VirtualBankMappingTable,
)


@dataclass(frozen=True)
class RunnerConfig:
    experiment_id: str
    workload_name: str
    resources: ResourceBudget
    interleave_bytes: int
    compute_cycles: int
    compute_intervals: Tuple[Tuple[int, int], ...]
    mapping_overhead_per_object: int
    prefetch_window: int
    pressure_queue_threshold: int
    pressure_conflict_threshold: int
    pressure_busy_threshold: int
    static_weight_banks: Tuple[int, ...]
    chunks: Tuple[WeightChunk, ...]
    compute_requests: Tuple[UnifiedMemoryRequest, ...]
    payload: Mapping[str, object]
    adaptive_prefetch: bool = False
    max_prefetch_window: int = 8
    max_prefetch_capacity_fraction: float = 0.25


@dataclass(frozen=True)
class RawBaselineExecution:
    """Measured row plus the real P9 execution detail used to derive it."""
    row: ExperimentRow
    chunks: Tuple[StreamingChunkResult, ...]
    memory_report: UnifiedDomainReport


def load_runner_config(path: Path) -> RunnerConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    hardware = payload["hardware"]
    resources = ResourceBudget(
        int(hardware["bank_count"]),
        int(hardware["capacity_bytes"]),
        float(hardware["bandwidth_bytes_per_cycle"]),
        int(hardware["ports_per_bank"]),
        int(hardware["request_buffer_depth"]),
    )
    chunks = tuple(WeightChunk(
        str(item["chunk_id"]), int(item["expert_id"]), int(item["ffn_part"]),
        int(item["tile_id"]), int(item["size_bytes"]), int(item["use_cycle"]),
        int(item["logical_address"]), int(item.get("bank_group_size", 1)),
    ) for item in payload["chunks"])
    requests = tuple(UnifiedMemoryRequest(
        request_id=str(item["request_id"]),
        issue_cycle=int(item["issue_cycle"]),
        tensor_type=str(item["tensor_type"]),
        object_id=str(item["object_id"]),
        address=int(item["address"]),
        size_bytes=int(item["size_bytes"]),
        kind=str(item.get("kind", "read")),
        preferred_banks=tuple(int(bank) for bank in item.get("preferred_banks", ())),
    ) for item in payload.get("compute_requests", ()))
    policy = payload["policy"]
    capacity_fraction = float(
        policy.get("max_prefetch_capacity_fraction", 0.25)
    )
    if not 0.0 < capacity_fraction <= 1.0:
        raise ValueError("max_prefetch_capacity_fraction must be in (0, 1]")
    return RunnerConfig(
        experiment_id=str(payload["experiment_id"]),
        workload_name=str(payload["workload_name"]),
        resources=resources,
        interleave_bytes=int(hardware["interleave_bytes"]),
        compute_cycles=int(payload["compute_cycles"]),
        compute_intervals=tuple(
            (int(interval[0]), int(interval[1]))
            for interval in payload.get("compute_intervals", ((0, int(payload["compute_cycles"])),))
        ),
        mapping_overhead_per_object=int(policy["mapping_overhead_per_object"]),
        prefetch_window=int(policy["prefetch_window"]),
        pressure_queue_threshold=int(policy["queue_threshold"]),
        pressure_conflict_threshold=int(policy["conflict_threshold"]),
        pressure_busy_threshold=int(policy["busy_threshold"]),
        static_weight_banks=tuple(int(bank) for bank in policy["static_weight_banks"]),
        chunks=chunks,
        compute_requests=requests,
        payload=payload,
        adaptive_prefetch=bool(policy.get("adaptive_prefetch", False)),
        max_prefetch_window=max(
            int(policy["prefetch_window"]),
            int(policy.get("max_prefetch_window", 8)),
        ),
        max_prefetch_capacity_fraction=capacity_fraction,
    )


def _compute_pressure(
    config: RunnerConfig, domain: UnifiedBankDomain
) -> Mapping[int, BankPressure]:
    report = domain.simulate(config.compute_requests)
    return {
        bank: BankPressure(
            queue_depth=report.per_bank_max_queue_depth[bank],
            busy_cycles=report.per_bank_busy_cycles[bank],
            conflicts=report.per_bank_conflicts[bank],
        )
        for bank in range(config.resources.bank_count)
    }


def _pressure_snapshot(report, cycle: int, bank_count: int, horizon: int = 64):
    """Time-local pressure visible to an online prefetch decision.

    Aggregate whole-run busy/conflict counts made every Bank permanently hot
    and caused Bank-aware prefetch to cancel all requests.  This snapshot uses
    only compute services overlapping the decision horizon.
    """
    end = cycle + horizon
    result = {}
    for bank in range(bank_count):
        active = [service for service in report.services
                  if bank in service.banks and service.issue_cycle < end
                  and service.completion_cycle > cycle]
        result[bank] = BankPressure(
            queue_depth=len(active),
            busy_cycles=sum(max(0, min(end, item.completion_cycle) -
                                max(cycle, item.start_cycle)) for item in active),
            conflicts=sum(item.queue_wait_cycles > 0 for item in active),
        )
    return result


def _bank_metrics(report) -> Mapping[str, float]:
    accesses = list(report.per_bank_accesses.values())
    busy = list(report.per_bank_busy_cycles.values())
    mean = float(sum(busy)) / float(len(busy)) if busy else 0.0
    variance = sum((value - mean) ** 2 for value in busy) / len(busy) if busy else 0.0
    finish = max(1, report.finish_cycle)
    conflict_count = sum(report.per_bank_conflicts.values())
    return {
        "bank_conflict_count": conflict_count,
        "bank_conflict_rate": float(conflict_count) / float(report.total_beats) if report.total_beats else 0.0,
        "bank_imbalance": sqrt(variance) / mean if mean else 0.0,
        "hotspot_bank_ratio": (
            float(sum(value > mean * 1.5 for value in busy)) / len(busy) if mean and busy else 0.0
        ),
        "idle_bank_ratio": float(sum(value == 0 for value in accesses)) / len(accesses) if accesses else 0.0,
        "effective_bank_parallelism": min(float(len(busy)), float(sum(busy)) / finish),
        "max_queue_depth": max(report.per_bank_max_queue_depth.values(), default=0),
    }


def _compute_interference(config: RunnerConfig, full_report, compute_only_report) -> int:
    full = {item.request_id: item for item in full_report.services}
    solo = {item.request_id: item for item in compute_only_report.services}
    delays = [
        max(0, full[request.request_id].completion_cycle - solo[request.request_id].completion_cycle)
        for request in config.compute_requests
    ]
    return max(delays, default=0)


def _communication_stall(config: RunnerConfig) -> int:
    system = config.payload.get("system", {})
    if int(system.get("num_gpus", 1)) <= 1:
        return 0
    latency = int(system.get("communication_latency_cycles", 0))
    bandwidth = float(system.get("communication_bandwidth_bytes_per_cycle", 1))
    remote_fraction = float(system.get("remote_token_fraction", 0.0))
    payload_bytes = int(system.get("token_payload_bytes", 0))
    tokens = int(config.payload.get("topology_provenance", {}).get("total_tokens", 0))
    if latency < 0 or bandwidth <= 0 or not 0.0 <= remote_fraction <= 1.0:
        raise ValueError("invalid EP communication configuration")
    remote_bytes = int(tokens * remote_fraction * payload_bytes)
    return latency + int((remote_bytes + bandwidth - 1) // bandwidth) if remote_bytes else 0


def _effective_prefetch_window(
    config: RunnerConfig, baseline: Baseline
) -> int:
    if (
        baseline not in (Baseline.MEMDOMAIN_RAW, Baseline.MEMDOMAIN_SAFE)
        or not config.adaptive_prefetch
    ):
        return config.prefetch_window
    largest_chunk = max((chunk.size_bytes for chunk in config.chunks), default=1)
    capacity_window = max(
        config.prefetch_window,
        int(
            config.resources.capacity_bytes
            * config.max_prefetch_capacity_fraction
            // largest_chunk
        ),
    )
    return min(config.max_prefetch_window, capacity_window)


def _decisions(
    baseline: Baseline,
    config: RunnerConfig,
    manager: ChunkResidencyManager,
    pressure: Mapping[int, BankPressure],
    compute_report=None,
) -> Tuple[PrefetchDecision, ...]:
    if baseline in (Baseline.STATIC_NOPF, Baseline.DYNAMIC_NOPF):
        return tuple(NoPrefetchPolicy().decide(chunk) for chunk in config.chunks)
    if baseline in (Baseline.STATIC_NAIVEPF, Baseline.DYNAMIC_NAIVEPF):
        planned = NaivePrefetchPolicy(
            config.prefetch_window, config.static_weight_banks
        ).plan(config.chunks)
        if baseline == Baseline.DYNAMIC_NAIVEPF:
            planned = tuple(PrefetchDecision(
                item.chunk_id, item.action, item.decision_cycle, item.issue_cycle,
                (), item.redirected, item.reason,
            ) for item in planned)
        return planned

    policy = BankAwarePrefetchPolicy(
        config.pressure_queue_threshold,
        config.pressure_conflict_threshold,
        config.pressure_busy_threshold,
    )
    decisions = {}
    free = {
        bank: manager.mapping.bank_capacity[bank] - manager.mapping.bank_occupied[bank]
        for bank in range(config.resources.bank_count)
    }
    ordered_chunks = tuple(sorted(
        config.chunks, key=lambda item: (item.use_cycle, item.chunk_id)
    ))
    effective_window = _effective_prefetch_window(config, baseline)
    events = []
    reservations = []
    sequence = 0
    for index, chunk in enumerate(ordered_chunks):
        trigger_index = index - effective_window
        issue = 0 if trigger_index < 0 else ordered_chunks[trigger_index].use_cycle
        heapq.heappush(events, (issue, sequence, index, chunk))
        sequence += 1

    def release_through(cycle):
        while reservations and reservations[0][0] <= cycle:
            _, _, allocation = heapq.heappop(reservations)
            for bank, amount in allocation.items():
                free[bank] += amount

    def feasible_allocation(chunk, banks):
        banks = tuple(banks)
        if len(banks) != chunk.bank_group_size or chunk.size_bytes < len(banks):
            return None
        if any(free.get(bank, 0) <= 0 for bank in banks):
            return None
        if sum(free[bank] for bank in banks) < chunk.size_bytes:
            return None
        allocation = {bank: 1 for bank in banks}
        remaining = chunk.size_bytes - len(banks)
        while remaining:
            bank = max(banks, key=lambda item: (free[item]-allocation[item], -item))
            available = free[bank] - allocation[bank]
            if available <= 0:
                return None
            grant = min(remaining, available)
            allocation[bank] += grant
            remaining -= grant
        return allocation

    def fallback_group(chunk, local_pressure):
        candidates = []
        for group in combinations(
            range(config.resources.bank_count), chunk.bank_group_size
        ):
            allocation = feasible_allocation(chunk, group)
            if allocation is None:
                continue
            score = sum(local_pressure.get(bank, BankPressure()).score for bank in group)
            candidates.append((score, -sum(free[bank] for bank in group), group, allocation))
        return min(candidates, default=None)

    while events:
        issue, _, index, chunk = heapq.heappop(events)
        if (
            sum(free.values()) < chunk.size_bytes
            or sum(value > 0 for value in free.values()) < chunk.bank_group_size
        ):
            release_through(issue)
        local_pressure = (_pressure_snapshot(
            compute_report, issue, config.resources.bank_count
        ) if compute_report is not None else pressure)
        snapshot = BankSnapshot(
            issue, local_pressure, dict(free), dict(manager.mapping.bank_capacity)
        )
        estimate = max(1, int(
            (chunk.size_bytes + config.resources.bandwidth_bytes_per_cycle - 1)
            // config.resources.bandwidth_bytes_per_cycle
        ))
        decision = policy.decide(
            chunk, snapshot, estimate, config.static_weight_banks,
            guard_incumbent=(baseline == Baseline.MEMDOMAIN_SAFE),
            switching_cost_cycles=config.mapping_overhead_per_object,
        )
        if decision.action == PrefetchAction.DELAY:
            heapq.heappush(
                events, (int(decision.issue_cycle), sequence, index, chunk)
            )
            sequence += 1
            continue
        if decision.action == PrefetchAction.CANCEL and issue < chunk.use_cycle:
            # P4: pressure alone may not remove work present in the matched
            # NaivePF baseline. Preserve the prefetch and use the best currently
            # feasible group; execution remains capacity-safe if no group is
            # available yet and will retry on the real release timeline.
            fallback = fallback_group(chunk, local_pressure)
            target = fallback[2] if fallback is not None else ()
            decision = PrefetchDecision(
                chunk.chunk_id, PrefetchAction.PREFETCH, issue, issue, target,
                bool(target and target != config.static_weight_banks[:len(target)]),
                "naive_plan_fallback",
            )
        decisions[index] = decision
        if decision.action == PrefetchAction.PREFETCH and decision.target_banks:
            allocation = feasible_allocation(chunk, decision.target_banks)
            if allocation is not None:
                for bank, amount in allocation.items():
                    free[bank] -= amount
                heapq.heappush(
                    reservations,
                    (max(chunk.use_cycle, issue + estimate), sequence, allocation),
                )
                sequence += 1
    return tuple(decisions[index] for index in range(len(ordered_chunks)))


def run_raw_baseline_with_details(
    config: RunnerConfig, baseline: Baseline
) -> RawBaselineExecution:
    if baseline not in {
        Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF, Baseline.DYNAMIC_NOPF,
        Baseline.DYNAMIC_NAIVEPF, Baseline.MEMDOMAIN_RAW,
        Baseline.MEMDOMAIN_SAFE,
    }:
        raise ValueError("run_raw_baseline accepts only measured baseline kinds")
    dynamic = baseline in {
        Baseline.DYNAMIC_NOPF, Baseline.DYNAMIC_NAIVEPF, Baseline.MEMDOMAIN_RAW,
        Baseline.MEMDOMAIN_SAFE,
    }
    mapping = VirtualBankMappingTable(
        config.resources, "conflict_aware" if baseline in (
            Baseline.MEMDOMAIN_RAW, Baseline.MEMDOMAIN_SAFE
        ) else (
            "least_occupied" if dynamic else "round_robin"
        )
    )
    # Policy planning reads this mapping's capacity view; P9 owns execution.
    manager = ChunkResidencyManager(mapping)
    domain = UnifiedBankDomain(config.resources, config.interleave_bytes)
    compute_only = domain.simulate(config.compute_requests)
    pressure = {
        bank: BankPressure(
            queue_depth=compute_only.per_bank_max_queue_depth[bank],
            busy_cycles=compute_only.per_bank_busy_cycles[bank],
            conflicts=compute_only.per_bank_conflicts[bank],
        ) for bank in range(config.resources.bank_count)
    }
    decisions = _decisions(baseline, config, manager, pressure, compute_only)
    by_chunk = {chunk.chunk_id: chunk for chunk in config.chunks}
    plans = []
    for decision in decisions:
        mapping_latency = (
            config.mapping_overhead_per_object
            if dynamic and (
                baseline != Baseline.MEMDOMAIN_SAFE
                or decision.guard_committed
            )
            else 0
        )
        preferred = tuple(decision.target_banks)
        if baseline in (Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF):
            preferred = config.static_weight_banks
        chunk = by_chunk[decision.chunk_id]
        if decision.action == PrefetchAction.PREFETCH:
            plans.append(StreamingLoadPlan(
                chunk, int(decision.issue_cycle), "prefetch", preferred,
                mapping_latency,
            ))
        else:
            plans.append(StreamingLoadPlan(
                chunk, chunk.use_cycle, "demand", preferred, mapping_latency,
            ))

    residency = StreamingResidencyEngine(domain, mapping).run(
        plans, config.compute_requests, pressure
    )
    full = residency.memory_report
    mapping_stats = mapping.statistics()
    bank_metrics = _bank_metrics(full)
    exposed_by_chunk = {
        item.chunk_id: min(
            item.mapping_latency_cycles, item.miss_stall_cycles
        )
        for item in residency.chunks
    }
    demand_stall = sum(
        item.miss_stall_cycles - exposed_by_chunk[item.chunk_id]
        for item in residency.chunks
        if item.classification == "demand_miss"
    )
    late_stall = sum(
        item.miss_stall_cycles - exposed_by_chunk[item.chunk_id]
        for item in residency.chunks
        if item.classification == "late"
    )
    base_bank_stall = max(
        (service.queue_wait_cycles for service in compute_only.services), default=0
    )
    interference = _compute_interference(config, full, compute_only)
    committed_mappings = (
        sum(item.guard_committed for item in decisions)
        if baseline == Baseline.MEMDOMAIN_SAFE
        else mapping_stats.mapping_count
    )
    mapping_work = (
        committed_mappings * config.mapping_overhead_per_object if dynamic else 0
    )
    mapping_overhead = sum(exposed_by_chunk.values()) if dynamic else 0
    mapping_hidden = mapping_work - mapping_overhead
    components = {
        "compute_cycles": config.compute_cycles,
        "bank_stall_cycles": base_bank_stall,
        "weight_load_stall_cycles": demand_stall,
        "prefetch_miss_stall_cycles": late_stall,
        "prefetch_interference_stall_cycles": interference,
        "mapping_overhead_cycles": mapping_overhead,
        "communication_stall_cycles": _communication_stall(config),
        "other_stall_cycles": 0,
    }
    total = sum(components.values())
    prefetches = [item for item in residency.chunks if item.effective_kind == "prefetch"]
    timely = [item for item in prefetches if item.classification == "timely"]
    late = [item for item in prefetches if item.classification == "late"]
    transfer_intervals = [
        (service.issue_cycle, service.completion_cycle)
        for service in full.services if service.request_id.startswith("load:")
    ]
    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0
    row = ExperimentRow(
        schema_version=1,
        experiment_id=config.experiment_id,
        workload_name=config.workload_name,
        workload_hash=workload_digest(config.payload),
        baseline=baseline.value,
        candidate_source=(
            f"measured:adaptive_window={_effective_prefetch_window(config, baseline)}"
            if baseline in (Baseline.MEMDOMAIN_RAW, Baseline.MEMDOMAIN_SAFE)
            and config.adaptive_prefetch
            else "measured"
        ),
        bank_count=config.resources.bank_count,
        capacity_bytes=config.resources.capacity_bytes,
        bandwidth_bytes_per_cycle=config.resources.bandwidth_bytes_per_cycle,
        ports_per_bank=config.resources.ports_per_bank,
        request_buffer_depth=config.resources.request_buffer_depth,
        total_cycles=total,
        **components,
        **bank_metrics,
        prefetch_requests=len(prefetches),
        prefetch_bytes=sum(by_chunk[item.chunk_id].size_bytes for item in prefetches),
        prefetch_coverage=ratio(len(prefetches), len(residency.chunks)),
        prefetch_accuracy=ratio(len(prefetches), len(prefetches)),
        timely_prefetch_ratio=ratio(len(timely), len(prefetches)),
        late_prefetch_ratio=ratio(len(late), len(prefetches)),
        unused_prefetch_ratio=0.0,
        prefetch_occupancy_byte_cycles=sum(
            by_chunk[item.chunk_id].size_bytes * (item.release_cycle - item.actual_issue_cycle)
            for item in prefetches
        ),
        compute_transfer_overlap_cycles=_intersection_cycles(
            config.compute_intervals, transfer_intervals
        ),
        mapping_count=mapping_stats.mapping_count,
        mapping_failures=mapping_stats.allocation_failures,
        peak_occupied_bytes=mapping_stats.peak_occupied_bytes,
        mapping_work_cycles=mapping_work,
        mapping_hidden_cycles=mapping_hidden,
        fallback_used=(
            baseline == Baseline.MEMDOMAIN_SAFE
            and any(not item.guard_committed for item in decisions)
        ),
        selected_candidate=(
            "Online-Guarded-Full"
            if baseline == Baseline.MEMDOMAIN_SAFE else ""
        ),
    )
    return RawBaselineExecution(row, residency.chunks, full)


def run_raw_baseline(config: RunnerConfig, baseline: Baseline) -> ExperimentRow:
    return run_raw_baseline_with_details(config, baseline).row


def run_best_static_baseline(
    config: RunnerConfig, baseline: Baseline
) -> ExperimentRow:
    return run_best_static_baseline_with_details(config, baseline).row


def run_best_static_baseline_with_details(
    config: RunnerConfig, baseline: Baseline
) -> RawBaselineExecution:
    if baseline not in (Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF):
        raise ValueError("static search accepts only static baselines")
    width = len(config.static_weight_banks)
    candidates = []
    groups = []
    for start in range(config.resources.bank_count):
        group = tuple(
            (start + offset) % config.resources.bank_count for offset in range(width)
        )
        if group not in groups:
            groups.append(group)
    for group in groups:
        candidate_config = RunnerConfig(
            experiment_id=config.experiment_id,
            workload_name=config.workload_name,
            resources=config.resources,
            interleave_bytes=config.interleave_bytes,
            compute_cycles=config.compute_cycles,
            compute_intervals=config.compute_intervals,
            mapping_overhead_per_object=config.mapping_overhead_per_object,
            prefetch_window=config.prefetch_window,
            pressure_queue_threshold=config.pressure_queue_threshold,
            pressure_conflict_threshold=config.pressure_conflict_threshold,
            pressure_busy_threshold=config.pressure_busy_threshold,
            static_weight_banks=tuple(group),
            chunks=config.chunks,
            compute_requests=config.compute_requests,
            payload=config.payload,
        )
        candidates.append((
            run_raw_baseline_with_details(candidate_config, baseline), group
        ))
    selected, group = min(
        candidates, key=lambda item: (item[0].row.total_cycles, item[1])
    )
    return RawBaselineExecution(
        replace(
            selected.row,
            candidate_source=(
                "exhaustive_cyclic_static_weight_groups:"
                + ":".join(map(str, group))
            ),
        ),
        selected.chunks,
        selected.memory_report,
    )


def run_dominating_dynamic_baseline_with_details(
    config: RunnerConfig,
    baseline: Baseline,
    static_incumbent: RawBaselineExecution,
) -> RawBaselineExecution:
    """Evaluate dynamic placement with the matched static optimum as incumbent.

    Keeping the incumbent in the feasible set is the P1 containment property:
    a dynamic mapping is committed only if its measured end-to-end objective is
    no larger. Otherwise the address translator preserves the static mapping.
    """
    matched = {
        Baseline.DYNAMIC_NOPF: Baseline.STATIC_NOPF,
        Baseline.DYNAMIC_NAIVEPF: Baseline.STATIC_NAIVEPF,
    }
    if baseline not in matched or static_incumbent.row.baseline != matched[baseline].value:
        raise ValueError("dynamic baseline requires its matched static incumbent")
    candidate = run_raw_baseline_with_details(config, baseline)
    if candidate.row.total_cycles < static_incumbent.row.total_cycles:
        return candidate
    return RawBaselineExecution(
        replace(
            static_incumbent.row,
            baseline=baseline.value,
            candidate_source=(
                "incumbent_static_mapping|" + static_incumbent.row.candidate_source
            ),
        ),
        static_incumbent.chunks,
        static_incumbent.memory_report,
    )


def _assert_matched_naive_prefetch_plans(
    static: RawBaselineExecution, dynamic: RawBaselineExecution
) -> None:
    """P2 fairness: placement may differ; the planned prefetch work may not."""
    def plan(execution):
        return tuple(sorted(
            (item.chunk_id, item.planned_kind, item.planned_issue_cycle)
            for item in execution.chunks
        ))
    if plan(static) != plan(dynamic):
        raise AssertionError(
            "Static-NaivePF and Dynamic-NaivePF use different prefetch plans"
        )
    if (
        static.row.prefetch_requests != dynamic.row.prefetch_requests
        or static.row.prefetch_bytes != dynamic.row.prefetch_bytes
    ):
        raise AssertionError(
            "Static-NaivePF and Dynamic-NaivePF use different prefetch workloads"
        )


def run_matrix(config: RunnerConfig) -> Tuple[ExperimentRow, ...]:
    static = run_best_static_baseline_with_details(config, Baseline.STATIC_NOPF)
    static_pf = run_best_static_baseline_with_details(
        config, Baseline.STATIC_NAIVEPF
    )
    dynamic = run_dominating_dynamic_baseline_with_details(
        config, Baseline.DYNAMIC_NOPF, static
    )
    dynamic_pf = run_dominating_dynamic_baseline_with_details(
        config, Baseline.DYNAMIC_NAIVEPF, static_pf
    )
    _assert_matched_naive_prefetch_plans(static_pf, dynamic_pf)
    raw_execution = run_raw_baseline_with_details(config, Baseline.MEMDOMAIN_RAW)
    safe_execution = run_raw_baseline_with_details(
        config, Baseline.MEMDOMAIN_SAFE
    )
    raw = [
        static.row, static_pf.row, dynamic.row, dynamic_pf.row, raw_execution.row,
    ]
    by_name = {row.baseline: row for row in raw}
    safe = safe_execution.row
    oracle = derive_selected_row(
        Baseline.ORACLE, [*raw, safe],
        [Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF, Baseline.DYNAMIC_NOPF,
         Baseline.DYNAMIC_NAIVEPF, Baseline.MEMDOMAIN_RAW, Baseline.MEMDOMAIN_SAFE],
    )
    return validate_matrix([*raw, safe, oracle])


def run_matrix_file(config_path: Path, output_path: Path) -> Tuple[ExperimentRow, ...]:
    rows = run_matrix(load_runner_config(config_path))
    write_matrix(output_path, rows)
    return rows
