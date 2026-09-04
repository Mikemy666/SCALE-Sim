"""End-to-end runner for the canonical MemDomain baseline matrix."""

from __future__ import annotations

import json
import heapq
import re
from dataclasses import dataclass, replace
from itertools import combinations
from math import ceil, sqrt
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from scalesim.memory.chunk_residency import ChunkResidencyManager, WeightChunk, _intersection_cycles
from scalesim.memory.buckyball_memdomain import (
    CONTRACT, BankAllocation, STATIC_ALLOCATION,
)
from scalesim.memory.buckyball_compiler import (
    compile_workload_static_plan, evaluate_gemm_allocation,
)
from scalesim.memory.topology_workload import EXPERT_LAYER, load_moe_topology
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
    dynamic_static_bank_overrides: Tuple[
        Tuple[str, Tuple[int, ...]], ...
    ] = ()
    static_bank_groups: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()
    dynamic_weight_bank_pools: Tuple[
        Tuple[str, Tuple[int, ...]], ...
    ] = ()
    dynamic_honor_preferred_banks: bool = False


def is_nonstationary_multilayer(config: RunnerConfig) -> bool:
    """Whether one trace contains Router-delimited consecutive MoE blocks."""
    return bool(config.payload.get("multi_layer_prefetch", {}).get("enabled"))


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
    requests_list = []
    for item in payload.get("compute_requests", ()):
        repeat_count = int(item.get("repeat_count", 1))
        repeat_interval = int(item.get("repeat_interval", 0))
        address_stride = int(item.get("address_stride", 0))
        if repeat_count <= 0 or repeat_interval < 0 or address_stride < 0:
            raise ValueError("invalid compact request repetition")
        for repeat in range(repeat_count):
            suffix = f"_t{repeat}" if repeat_count > 1 else ""
            requests_list.append(UnifiedMemoryRequest(
                request_id=str(item["request_id"]) + suffix,
                issue_cycle=int(item["issue_cycle"]) + repeat * repeat_interval,
                tensor_type=str(item["tensor_type"]),
                object_id=str(item["object_id"]) + suffix,
                address=int(item["address"]) + repeat * address_stride,
                size_bytes=int(item["size_bytes"]),
                kind=str(item.get("kind", "read")),
                preferred_banks=tuple(
                    int(bank) for bank in item.get("preferred_banks", ())
                ),
                wmode=int(item.get("wmode", 0)),
                bank_group_size=int(item.get("bank_group_size", 0)),
            ))
    requests = tuple(requests_list)
    policy = payload["policy"]
    selector = policy.get("prefetch_policy")
    if selector is not None and selector not in {
        "none", "naive_fixed", "bank_aware_raw", "bank_aware_guarded",
        "coverage_accuracy_constrained",
    }:
        raise ValueError(f"unsupported prefetch_policy: {selector}")
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


def static_allocation_config(
    config: RunnerConfig, allocation: BankAllocation,
) -> RunnerConfig:
    """Freeze one contiguous IA/Weight/OA/ACC ownership for a complete run."""
    if allocation.total != config.resources.bank_count:
        raise ValueError("static allocation must conserve all physical Banks")
    cursor = 0
    groups = []
    for name, width in (
        ("ia", allocation.ia), ("weight", allocation.weight),
        ("oa", allocation.oa), ("accumulator", allocation.accumulator),
    ):
        if width <= 0:
            raise ValueError("every static tensor domain needs at least one Bank")
        group = tuple(range(cursor, cursor + width))
        groups.append((name, group))
        cursor += width
    return replace(
        config,
        static_weight_banks=dict(groups)["weight"],
        static_bank_groups=tuple(groups),
    )


def profiled_static_allocation(config: RunnerConfig) -> BankAllocation:
    """Compile one model-wide fixed allocation from every expert GEMM.

    The topology is part of the hashed workload provenance.  No runtime event
    or future prefetch outcome is consulted here.
    """
    provenance = config.payload.get("topology_provenance", {})
    source = provenance.get("source_path")
    if not source:
        return STATIC_ALLOCATION
    if is_nonstationary_multilayer(config):
        dimensions = tuple(
            (int(item["m"]), int(item["n"]), int(item["k"]))
            for item in config.payload.get("compiler_bank_plans", ())
            if all(name in item for name in ("m", "n", "k"))
        )
    else:
        topology = load_moe_topology(Path(str(source)))
        dimensions = tuple(
            (m, n, k) for name, m, n, k in topology["layers"]
            if EXPERT_LAYER.fullmatch(name)
        )
    if not dimensions:
        return STATIC_ALLOCATION
    return compile_workload_static_plan(dimensions)


_STAGE_ID = re.compile(r"(?:^|_)e(\d+)_ff([12])(?:_|$)")
_PLAN_ID = re.compile(r"MoE-E(\d+)-FF([12])$")


def _allocation_groups(
    allocation: BankAllocation, rotation: int = 0,
) -> Mapping[str, Tuple[int, ...]]:
    """Return disjoint physical candidate pools for one compiler epoch."""
    cursor = 0
    result = {}
    for name, width in (
        ("ia", allocation.ia), ("weight", allocation.weight),
        ("oa", allocation.oa), ("accumulator", allocation.accumulator),
    ):
        result[name] = tuple(
            (rotation + bank) % CONTRACT.bank_count
            for bank in range(cursor, cursor + width)
        )
        cursor += width
    if cursor != CONTRACT.bank_count:
        raise ValueError("dynamic compiler allocation must use all physical Banks")
    return result


def compiled_dynamic_config(
    config: RunnerConfig, *, prefetch_coordinated: bool = False,
) -> RunnerConfig:
    """Apply the real per-expert/per-FFN compiler plan to execution.

    The previous DATE3 path exported ``compiler_bank_plans`` for analysis but
    ignored it in the event simulator: dynamic compute mappings searched all
    thirty Banks and weight chunks had no stage pool.  Here each stage changes
    only its virtual-to-physical candidate pools.  Physical capacity, ports,
    request timing, tensor widths, and demand traffic remain unchanged.  Pure
    Dynamic independently rotates each stage.  PIVOT requests the coordinated
    form, which keeps a common physical origin so future prefetched weights do
    not unexpectedly alias a preceding stage's IA/OA/ACC pools.
    """
    plans = {}
    for item in config.payload.get("compiler_bank_plans", ()):
        if "expert_id" in item and "ffn_part" in item:
            expert, part = int(item["expert_id"]), int(item["ffn_part"])
        else:
            match = _PLAN_ID.fullmatch(str(item.get("layer", "")))
            if not match:
                continue
            expert, part = int(match.group(1)), int(match.group(2))
        allocation = BankAllocation(
            int(item["ia_banks"]), int(item["weight_banks"]),
            int(item["oa_banks"]), int(item["acc_banks"]),
        )
        if allocation.total != config.resources.bank_count:
            raise ValueError("stage compiler plan does not conserve physical Banks")
        # Reusing one physical origin for every stage accidentally preserved
        # global tensor-type partitions.  A virtual mapping architecture
        # instead remaps each stage; this deterministic compiler rotation
        # makes that address translation visible without consulting outcomes.
        rotation = (
            0 if prefetch_coordinated
            else (expert * 7 + (part - 1) * 13) % CONTRACT.bank_count
        )
        plans[(expert, part)] = _allocation_groups(allocation, rotation)
    if not plans:
        # Unit configurations without topology/compiler provenance retain the
        # existing dynamic mapper rather than inventing a paper-scale plan.
        return config

    requests = []
    for request in config.compute_requests:
        match = _STAGE_ID.search(request.object_id)
        if match is None:
            match = _STAGE_ID.search(request.request_id)
        if match is None:
            requests.append(request)
            continue
        identity = (int(match.group(1)), int(match.group(2)))
        pools = plans.get(identity)
        if pools is None:
            raise ValueError(f"missing compiler plan for active stage {identity}")
        pool = pools[request.tensor_type]
        group_size = request.bank_group_size or max(1, len(request.preferred_banks))
        if group_size > len(pool):
            raise ValueError(
                f"{request.tensor_type} vBank group exceeds compiler pool in {identity}"
            )
        requests.append(replace(request, preferred_banks=pool))

    weight_pools = []
    for chunk in config.chunks:
        identity = (chunk.expert_id, chunk.ffn_part)
        if identity not in plans:
            raise ValueError(f"missing compiler plan for weight stage {identity}")
        weight_pools.append((chunk.chunk_id, plans[identity]["weight"]))
    return replace(
        config,
        compute_requests=tuple(requests),
        dynamic_weight_bank_pools=tuple(weight_pools),
        dynamic_honor_preferred_banks=True,
    )


def compiler_bank_service_cycles(config: RunnerConfig) -> int:
    """Trace-visible expert Bank service on the GEMM critical paths.

    The request simulator accounts for port collisions, but its formerly used
    ``max(queue_wait)`` omits the compulsory IA/Weight/OA transfers and ACC
    read-modify-write service itself.  Consequently a conflict-free request
    trace incorrectly reported zero on-chip stall and made every allocation
    equivalent.  This function adds that missing service term using the same
    compiler objective that selected the legal allocation; observed queueing
    remains an additional term.
    """
    provenance = config.payload.get("topology_provenance", {})
    source = provenance.get("source_path")
    if not source or not config.payload.get("compiler_bank_plans"):
        return 0
    dimensions = {}
    if is_nonstationary_multilayer(config):
        for item in config.payload["compiler_bank_plans"]:
            if all(name in item for name in ("expert_id", "ffn_part", "m", "n", "k")):
                dimensions[(int(item["expert_id"]), int(item["ffn_part"]))] = (
                    int(item["m"]), int(item["n"]), int(item["k"])
                )
    else:
        topology = load_moe_topology(Path(str(source)))
        for name, m, n, k in topology["layers"]:
            match = _PLAN_ID.fullmatch(name)
            if match:
                dimensions[(int(match.group(1)), int(match.group(2)))] = (m, n, k)
    active = {(chunk.expert_id, chunk.ffn_part) for chunk in config.chunks}
    if config.dynamic_honor_preferred_banks:
        allocations = {}
        for item in config.payload["compiler_bank_plans"]:
            identity = None
            if "expert_id" in item and "ffn_part" in item:
                identity = (int(item["expert_id"]), int(item["ffn_part"]))
            else:
                match = _PLAN_ID.fullmatch(str(item.get("layer", "")))
                if match:
                    identity = (int(match.group(1)), int(match.group(2)))
            if identity is not None:
                allocations[identity] = (
                    BankAllocation(
                        int(item["ia_banks"]), int(item["weight_banks"]),
                        int(item["oa_banks"]), int(item["acc_banks"]),
                    )
                )
    else:
        if config.static_bank_groups:
            widths = {name: len(group) for name, group in config.static_bank_groups}
            fixed = BankAllocation(
                widths["ia"], widths["weight"], widths["oa"], widths["accumulator"]
            )
        else:
            fixed = STATIC_ALLOCATION
        allocations = {identity: fixed for identity in active}
    missing = active - dimensions.keys() | active - allocations.keys()
    if missing:
        raise ValueError(f"missing compiler service model for active stages {sorted(missing)}")
    return sum(
        evaluate_gemm_allocation(*dimensions[identity], allocations[identity])[1]
        for identity in active
    )


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
    # DATE3 EP supplies the exact number of remote Top-k route replicas.  Older
    # payloads retain the legacy aggregate-fraction behavior for compatibility.
    exact_remote_replicas = system.get("remote_route_replicas")
    remote_bytes = (
        int(exact_remote_replicas) * payload_bytes
        if exact_remote_replicas is not None
        else int(tokens * remote_fraction * payload_bytes)
    )
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


def offchip_load_cycles(config: RunnerConfig, size_bytes: int) -> int:
    """HBM startup plus serialization, shared by every paper scheme."""
    hardware = config.payload.get("hardware", {})
    startup = int(hardware.get("offchip_startup_cycles", 0))
    bits_per_cycle = float(hardware.get("offchip_bandwidth_bits_per_cycle", 8))
    if startup < 0 or bits_per_cycle <= 0 or size_bytes <= 0:
        raise ValueError("invalid off-chip transfer parameters")
    return startup + int(ceil(size_bytes / (bits_per_cycle / 8.0)))


def critical_path_miss_stalls(chunks, *, multi_layer: bool,
                              subtract_mapping: bool = False):
    """Return demand/late stalls without double-counting overlapping waits."""
    raw = {"demand_miss": 0, "late": 0}
    intervals = []
    for item in chunks:
        if item.classification not in raw:
            continue
        mapping = (
            min(item.mapping_latency_cycles, item.miss_stall_cycles)
            if subtract_mapping else 0
        )
        stall = max(0, item.miss_stall_cycles - mapping)
        raw[item.classification] += stall
        if stall:
            intervals.append((item.use_cycle, item.use_cycle + stall))
    if not multi_layer:
        return raw["demand_miss"], raw["late"]
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    exposed = sum(end - start for start, end in merged)
    total_raw = raw["demand_miss"] + raw["late"]
    if not total_raw:
        return 0, 0
    demand = int(round(exposed * raw["demand_miss"] / total_raw))
    return demand, exposed - demand


def _fixed_issue_schedule(config: RunnerConfig, window: int):
    """Fixed look-ahead constrained by per-layer Router visibility.

    A multi-layer trace must not let a request in layer L+1 be triggered by a
    Chunk in layer L: the L+1 expert set does not exist architecturally until
    that layer's Router has completed.  Single-layer DATE experiments retain
    their original global fixed-window schedule.
    """
    ordered = tuple(sorted(
        config.chunks, key=lambda item: (item.use_cycle, item.chunk_id)
    ))
    multi = config.payload.get("multi_layer_prefetch", {})
    if not multi.get("enabled"):
        return {
            chunk.chunk_id: (
                chunk.use_cycle if window == 0 else
                0 if index - window < 0 else ordered[index - window].use_cycle
            )
            for index, chunk in enumerate(ordered)
        }
    experts_per_layer = int(multi["experts_per_layer"])
    visibility = {
        int(item["layer_id"]): int(item["start_cycle"])
        for item in config.payload.get("topology_provenance", {}).get(
            "layer_profiles", ()
        )
    }
    by_layer = {}
    for chunk in ordered:
        layer = chunk.expert_id // experts_per_layer
        by_layer.setdefault(layer, []).append(chunk)
    schedule = {}
    for layer, chunks in sorted(by_layer.items()):
        if layer not in visibility:
            raise ValueError(f"missing Router visibility for layer {layer}")
        route_cycle = visibility[layer]
        for index, chunk in enumerate(chunks):
            if window == 0:
                issue = chunk.use_cycle
            elif index - window < 0:
                issue = route_cycle
            else:
                issue = max(route_cycle, chunks[index - window].use_cycle)
            schedule[chunk.chunk_id] = issue
    return schedule


def _atomic_noprefetch_config(config: RunnerConfig) -> RunnerConfig:
    """Use the architectural Weight-tile request for a NoPF control.

    Runtime Chunk coalescing is a prefetch mechanism.  Letting NoPF inherit a
    C=8 experiment seed silently grants it the same eight-tile HBM burst and
    makes a supposedly parameter-free control vary across the Chunk sweep.
    """
    provenance = config.payload.get("topology_provenance", {})
    if not is_nonstationary_multilayer(config):
        return config
    tile_size = int(provenance.get("tile_size", 16))
    bytes_per_element = int(provenance.get("weight_bytes_per_element", 1))
    atomic_bytes = tile_size * tile_size * bytes_per_element
    source_pools = dict(config.dynamic_weight_bank_pools)
    chunks = []
    pools = []
    by_stage = {}
    for item in config.chunks:
        by_stage.setdefault((item.expert_id, item.ffn_part), []).append(item)
    for identity in sorted(by_stage):
        source = sorted(
            by_stage[identity], key=lambda item: (item.use_cycle, item.tile_id)
        )
        for index, item in enumerate(source):
            pieces = int(ceil(item.size_bytes / atomic_bytes))
            if index + 1 < len(source):
                step = max(
                    1,
                    int(round(
                        max(1, source[index + 1].use_cycle - item.use_cycle)
                        / pieces
                    )),
                )
            elif index:
                step = max(
                    1,
                    int(round(
                        max(1, item.use_cycle - source[index - 1].use_cycle)
                        / pieces
                    )),
                )
            else:
                step = max(1, tile_size * 2)
            for offset in range(pieces):
                chunk = replace(
                    item,
                    chunk_id=f"{item.chunk_id}__demand_tile{offset}",
                    tile_id=item.tile_id * pieces + offset,
                    size_bytes=min(
                        atomic_bytes, item.size_bytes - offset * atomic_bytes
                    ),
                    use_cycle=max(
                        0, item.use_cycle - (pieces - 1 - offset) * step
                    ),
                    logical_address=(
                        item.logical_address + offset * atomic_bytes
                    ),
                )
                chunks.append(chunk)
                if item.chunk_id in source_pools:
                    pools.append((chunk.chunk_id, source_pools[item.chunk_id]))
    return replace(
        config,
        chunks=tuple(sorted(
            chunks,
            key=lambda item: (
                item.use_cycle, item.expert_id, item.ffn_part, item.tile_id
            ),
        )),
        dynamic_weight_bank_pools=tuple(pools),
    )


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
        if config.payload.get("multi_layer_prefetch", {}).get("enabled"):
            issue_schedule = _fixed_issue_schedule(
                config, config.prefetch_window
            )
            planned = tuple(
                NoPrefetchPolicy().decide(chunk)
                if config.prefetch_window == 0 else PrefetchDecision(
                    chunk.chunk_id, PrefetchAction.PREFETCH,
                    issue_schedule[chunk.chunk_id],
                    issue_schedule[chunk.chunk_id],
                    config.static_weight_banks, False,
                    "fixed_window_router_gated",
                )
                for chunk in sorted(
                    config.chunks,
                    key=lambda item: (item.use_cycle, item.chunk_id),
                )
            )
        else:
            planned = NaivePrefetchPolicy(
                config.prefetch_window, config.static_weight_banks
            ).plan(config.chunks)
        if baseline == Baseline.DYNAMIC_NAIVEPF:
            # Keep the Static-NaivePF issue plan exactly unchanged and optimize
            # only virtual-to-physical placement.  "Least occupied" ignored
            # the request deadline and repeatedly moved prefetches onto Banks
            # that completed later than the exhaustive static incumbent.
            # Rank an equal-width candidate pool using the pressure visible at
            # the common issue cycle; the mapping table performs the final
            # capacity-safe allocation at execution time.
            redirected = []
            static_overrides = dict(config.dynamic_static_bank_overrides)
            compiler_pools = dict(config.dynamic_weight_bank_pools)

            for item, chunk in zip(
                planned, sorted(config.chunks,
                                key=lambda value: (value.use_cycle,
                                                   value.chunk_id))
            ):
                issue = int(item.issue_cycle)
                local = (_pressure_snapshot(
                    compute_report, issue, config.resources.bank_count,
                    horizon=max(64, chunk.use_cycle - issue),
                ) if compute_report is not None else pressure)
                transfer = max(
                    1, (chunk.size_bytes
                        + config.resources.bandwidth_bytes_per_cycle - 1)
                    // config.resources.bandwidth_bytes_per_cycle,
                )
                # `target_banks` is a candidate pool, not the final physical
                # group. Preserve the same pool width as the static baseline
                # so both schemes have identical capacity/parallelism; only
                # the Bank identities change dynamically.
                pool_width = max(
                    chunk.bank_group_size, len(config.static_weight_banks)
                )
                allowed_banks = compiler_pools.get(
                    chunk.chunk_id, tuple(range(config.resources.bank_count))
                )
                ranked = sorted(
                    allowed_banks,
                    key=lambda bank: (
                        max(
                            0,
                            issue + transfer * (
                                1 + local.get(
                                    bank, BankPressure()
                                ).queue_depth
                            ) - chunk.use_cycle,
                        ),
                        local.get(bank, BankPressure()).conflicts,
                        local.get(bank, BankPressure()).queue_depth,
                        local.get(bank, BankPressure()).busy_cycles,
                        bank,
                    ),
                )
                target = tuple(ranked[:min(pool_width, len(ranked))])
                if chunk.chunk_id in static_overrides:
                    target = tuple(static_overrides[chunk.chunk_id])
                    reason = "layer_guard_static_incumbent"
                    guard_committed = False
                else:
                    reason = "fixed_window_dynamic_deadline_mapping"
                    guard_committed = True
                redirected.append(PrefetchDecision(
                    item.chunk_id, item.action, item.decision_cycle,
                    item.issue_cycle, tuple(target),
                    bool(target and target != tuple(
                        config.static_weight_banks[:len(target)]
                    )),
                    reason, guard_committed,
                ))
            planned = tuple(redirected)
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
    issue_schedule = _fixed_issue_schedule(config, effective_window)
    events = []
    reservations = []
    sequence = 0
    for index, chunk in enumerate(ordered_chunks):
        issue = issue_schedule[chunk.chunk_id]
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
    if baseline in (Baseline.STATIC_NOPF, Baseline.DYNAMIC_NOPF):
        config = _atomic_noprefetch_config(config)
    # Baseline ownership follows the DATE2 1:1:1:3 partition.  Derive the
    # ranges from the configured resource count instead of assuming 30 Banks:
    # for the paper configuration this remains exactly 5/5/5/15, while small
    # smoke/robustness configurations cannot produce out-of-range requests.
    bank_count = config.resources.bank_count
    if bank_count < 4:
        raise ValueError("DATE2 requires at least four physical Banks")
    sp_width = max(1, bank_count // 6)
    if 3 * sp_width >= bank_count:
        sp_width = 1
    acc_start = 3 * sp_width
    static_groups = {
        "ia": tuple(range(0, sp_width)),
        "weight": tuple(range(sp_width, 2 * sp_width)),
        "oa": tuple(range(2 * sp_width, 3 * sp_width)),
        "accumulator": tuple(range(acc_start, bank_count)),
    }
    if config.static_bank_groups:
        configured = dict(config.static_bank_groups)
        if set(configured) != set(static_groups):
            raise ValueError("static Bank groups must define IA/Weight/OA/ACC")
        flattened = tuple(bank for group in configured.values() for bank in group)
        if sorted(flattened) != list(range(bank_count)) or len(set(flattened)) != bank_count:
            raise ValueError("static Bank groups must partition every physical Bank once")
        static_groups = configured
    compute_requests = (
        config.compute_requests if dynamic else tuple(
            replace(
                request,
                preferred_banks=static_groups[request.tensor_type],
                bank_group_size=(
                    min(4, len(static_groups["accumulator"]))
                    if request.tensor_type == "accumulator"
                    else len(static_groups[request.tensor_type])
                ),
            )
            for request in config.compute_requests
        )
    )
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
    compute_only = domain.simulate(compute_requests)
    pressure = {
        bank: BankPressure(
            queue_depth=compute_only.per_bank_max_queue_depth[bank],
            busy_cycles=compute_only.per_bank_busy_cycles[bank],
            conflicts=compute_only.per_bank_conflicts[bank],
        ) for bank in range(config.resources.bank_count)
    }
    decisions = _decisions(baseline, config, manager, pressure, compute_only)
    by_chunk = {chunk.chunk_id: chunk for chunk in config.chunks}
    dynamic_weight_pools = dict(config.dynamic_weight_bank_pools)
    plans = []
    for decision in decisions:
        mapping_latency = (
            config.mapping_overhead_per_object
            if dynamic and decision.guard_committed else 0
        )
        preferred = tuple(decision.target_banks)
        if baseline in (Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF):
            preferred = config.static_weight_banks
        elif dynamic and not preferred and decision.chunk_id in dynamic_weight_pools:
            preferred = dynamic_weight_pools[decision.chunk_id]
        chunk = by_chunk[decision.chunk_id]
        if decision.action == PrefetchAction.PREFETCH:
            plans.append(StreamingLoadPlan(
                chunk, int(decision.issue_cycle), "prefetch", preferred,
                mapping_latency,
                offchip_latency_cycles=offchip_load_cycles(
                    config, chunk.size_bytes
                ),
            ))
        else:
            plans.append(StreamingLoadPlan(
                chunk, chunk.use_cycle, "demand", preferred, mapping_latency,
                offchip_latency_cycles=offchip_load_cycles(
                    config, chunk.size_bytes
                ),
            ))

    residency = StreamingResidencyEngine(domain, mapping).run(
        plans, compute_requests, pressure,
        dynamic_compute_mapping=(
            dynamic and not config.dynamic_honor_preferred_banks
        ),
        bind_prefetched_weight_reads=bool(
            config.payload.get("multi_layer_prefetch", {}).get("enabled")
        ),
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
    demand_stall, late_stall = critical_path_miss_stalls(
        residency.chunks,
        multi_layer=is_nonstationary_multilayer(config),
        subtract_mapping=True,
    )
    observed_bank_queue = max(
        (service.queue_wait_cycles for service in compute_only.services), default=0
    )
    base_bank_stall = compiler_bank_service_cycles(config) + observed_bank_queue
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
    # Fairness and timeliness are defined over the planned prefetch workload.
    # A request delayed beyond its use deadline remains a late planned
    # prefetch instead of disappearing from the workload as a demand request.
    prefetches = [
        item for item in residency.chunks if item.planned_kind == "prefetch"
    ]
    timely = [item for item in prefetches if item.classification == "timely"]
    late = [
        item for item in prefetches
        if item.classification in ("late", "demand_miss")
    ]
    transfer_intervals = [
        (service.issue_cycle, service.completion_cycle)
        for service in full.services if service.request_id.startswith("load:")
    ]
    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0
    hbm_active = [
        item for item in residency.chunks if item.offchip_latency_cycles > 0
    ]
    hbm_span = (
        max(item.hbm_complete_cycle for item in hbm_active)
        - min(item.hbm_issue_cycle for item in hbm_active)
        if hbm_active else 0
    )
    row = ExperimentRow(
        schema_version=1,
        experiment_id=config.experiment_id,
        workload_name=config.workload_name,
        workload_hash=workload_digest(config.payload),
        baseline=baseline.value,
        candidate_source=(
            "measured:atomic_demand_tiles"
            if baseline in (Baseline.STATIC_NOPF, Baseline.DYNAMIC_NOPF)
            and config.payload.get("multi_layer_prefetch", {}).get("enabled")
            else f"measured:adaptive_window={_effective_prefetch_window(config, baseline)}"
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
        hbm_queue_wait_cycles=residency.hbm_queue_wait_cycles,
        hbm_service_cycles=residency.hbm_service_cycles,
        hbm_busy_cycles=residency.hbm_busy_cycles,
        hbm_max_queue_depth=residency.hbm_max_queue_depth,
        hbm_utilization=min(
            1.0, ratio(residency.hbm_busy_cycles, hbm_span)
        ),
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
    if config.dynamic_honor_preferred_banks or is_nonstationary_multilayer(config):
        # DATE3 compiler-driven Dynamic is allowed to trade a small local
        # regression in one FFN for a lower complete-model critical path.
        # Static-Opt remains an exact feasible incumbent, but containment is
        # checked at the paper objective (model total), not independently for
        # every layer.  Router-delimited multi-block traces also use atomic
        # NoPF demand tiles whose IDs do not exist in the original coalesced
        # Chunk table; their correct containment scope is the complete trace.
        if candidate.row.total_cycles < static_incumbent.row.total_cycles:
            return RawBaselineExecution(
                replace(
                    candidate.row,
                    candidate_source="measured:model_guarded_stage_mapping",
                ),
                candidate.chunks,
                candidate.memory_report,
            )
        return RawBaselineExecution(
            replace(
                static_incumbent.row,
                baseline=baseline.value,
                candidate_source=(
                    "incumbent_static_mapping|"
                    + static_incumbent.row.candidate_source
                ),
            ),
            static_incumbent.chunks,
            static_incumbent.memory_report,
        )
    # The placement containment contract is independent of prefetching:
    # both dynamic variants must retain their matched static placement as a
    # feasible per-layer incumbent.  Previously this guard covered only the
    # naive-prefetch pair, allowing Dynamic-NoPF to regress locally under
    # highly skewed routing even when its model total still improved.
    by_chunk = {chunk.chunk_id: chunk for chunk in config.chunks}
    static_chunks = {
        item.chunk_id: item for item in static_incumbent.chunks
    }
    overrides = {}

    def layer_penalties(execution):
        penalties = {}
        for item in execution.chunks:
            chunk = by_chunk[item.chunk_id]
            key = (chunk.expert_id, chunk.ffn_part)
            penalties[key] = penalties.get(key, 0) + (
                item.miss_stall_cycles
                + item.allocation_wait_cycles
                + min(item.mapping_latency_cycles,
                      item.miss_stall_cycles)
            )
        return penalties

    static_penalties = layer_penalties(static_incumbent)
    # P11 Safe-by-construction dynamic baseline: retain profitable
    # per-layer mappings and pin only locally regressing expert FFNs to
    # their measured static physical placement. Re-evaluate after every
    # update because concurrent residency can change neighbouring layers.
    for _ in range(len(static_penalties) + 1):
        dynamic_penalties = layer_penalties(candidate)
        regressions = {
            key for key, value in dynamic_penalties.items()
            if value > static_penalties[key]
        }
        if not regressions:
            if overrides:
                candidate = RawBaselineExecution(
                    replace(
                        candidate.row,
                        candidate_source=(
                            "measured:layer_guarded_dynamic_mapping"
                        ),
                        fallback_used=True,
                    ),
                    candidate.chunks,
                    candidate.memory_report,
                )
            break
        previous = len(overrides)
        for chunk in config.chunks:
            if (chunk.expert_id, chunk.ffn_part) in regressions:
                overrides[chunk.chunk_id] = static_chunks[
                    chunk.chunk_id
                ].physical_banks
        if len(overrides) == previous:
            candidate = static_incumbent
            break
        guarded_config = replace(
            config,
            dynamic_static_bank_overrides=tuple(sorted(overrides.items())),
        )
        candidate = run_raw_baseline_with_details(
            guarded_config, baseline
        )
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
    if not config.adaptive_prefetch:
        # A zero prefetch window makes both dynamic controls demand-only, but
        # their placement engines can still differ.  Safe must retain the
        # measured best implementable fixed-point incumbent rather than
        # selecting by the policy label alone.
        incumbent = min(
            (dynamic, dynamic_pf),
            key=lambda execution: (
                execution.row.total_cycles, execution.row.baseline
            ),
        )
        safe_execution = RawBaselineExecution(
            replace(
                incumbent.row,
                baseline=Baseline.MEMDOMAIN_SAFE.value,
                candidate_source="measured:fixed_window_incumbent",
                fallback_used=True,
                selected_candidate="Online-Guarded-Full",
            ),
            incumbent.chunks,
            incumbent.memory_report,
        )
    else:
        safe_execution = run_raw_baseline_with_details(
            config, Baseline.MEMDOMAIN_SAFE
        )
        implementable = min(
            (static, static_pf, dynamic, dynamic_pf),
            key=lambda execution: (
                execution.row.total_cycles, execution.row.baseline
            ),
        )
        if safe_execution.row.total_cycles > implementable.row.total_cycles:
            # Final online guard: if accumulated local predictions still make
            # the complete guarded schedule slower, retain the already
            # measured implementable incumbent. This is a real fallback row,
            # not an Oracle choice over MemDomain-Raw.
            safe_execution = RawBaselineExecution(
                replace(
                    implementable.row,
                    baseline=Baseline.MEMDOMAIN_SAFE.value,
                    candidate_source=(
                        "measured:online_model_incumbent|"
                        + implementable.row.baseline
                    ),
                    fallback_used=True,
                    selected_candidate="Online-Guarded-Full",
                ),
                implementable.chunks,
                implementable.memory_report,
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


def run_paper_control_executions(
    config: RunnerConfig,
) -> Mapping[str, RawBaselineExecution]:
    """Run the six implementable paper controls plus a NoPF lower bound.

    Internal seven-row matrices are retained for backwards-compatible
    diagnostics.  This separate public contract prevents the legacy
    ``Static-NoPF`` (a cyclic Weight placement search) from being relabelled
    as either the literal 5/5/5/15 baseline or the model-wide four-domain
    static optimum.
    """
    static_555_config = static_allocation_config(config, STATIC_ALLOCATION)
    static_555 = run_raw_baseline_with_details(
        static_555_config, Baseline.STATIC_NOPF
    )

    compiled_allocation = profiled_static_allocation(config)
    compiled_config = static_allocation_config(config, compiled_allocation)
    compiled_static = run_raw_baseline_with_details(
        compiled_config, Baseline.STATIC_NOPF
    )
    if compiled_static.row.total_cycles < static_555.row.total_cycles:
        static_opt_config, static_opt = compiled_config, compiled_static
        allocation_source = compiled_allocation.as_tuple()
    else:
        static_opt_config, static_opt = static_555_config, static_555
        allocation_source = STATIC_ALLOCATION.as_tuple()
    static_opt = RawBaselineExecution(
        replace(
            static_opt.row,
            candidate_source=(
                "model_wide_static_allocation:"
                + ":".join(map(str, allocation_source))
            ),
        ),
        static_opt.chunks,
        static_opt.memory_report,
    )

    static_opt_pf = run_raw_baseline_with_details(
        static_opt_config, Baseline.STATIC_NAIVEPF
    )
    static_opt_pf = RawBaselineExecution(
        replace(
            static_opt_pf.row,
            candidate_source=static_opt.row.candidate_source,
        ),
        static_opt_pf.chunks,
        static_opt_pf.memory_report,
    )
    dynamic_config = compiled_dynamic_config(config)
    dynamic = run_dominating_dynamic_baseline_with_details(
        dynamic_config, Baseline.DYNAMIC_NOPF, static_opt
    )
    dynamic_pf = run_dominating_dynamic_baseline_with_details(
        dynamic_config, Baseline.DYNAMIC_NAIVEPF, static_opt_pf
    )
    _assert_matched_naive_prefetch_plans(static_opt_pf, dynamic_pf)

    # Conflict-free NoPF reference: same demand traffic and off-chip service,
    # but no exposed on-chip Bank queueing.  It is a lower-bound reference,
    # never an implementable proposed scheme.
    ideal_row = replace(
        dynamic.row,
        baseline=Baseline.ORACLE.value,
        candidate_source="conflict_free_nopf_lower_bound",
        bank_stall_cycles=0,
        bank_conflict_count=0,
        bank_conflict_rate=0.0,
        total_cycles=dynamic.row.total_cycles - dynamic.row.bank_stall_cycles,
        selected_candidate=Baseline.DYNAMIC_NOPF.value,
    )
    ideal = RawBaselineExecution(
        ideal_row, dynamic.chunks, dynamic.memory_report
    )
    return {
        "Static-555-NoPF": static_555,
        "Static-Opt-NoPF": static_opt,
        "Dynamic-NoPF": dynamic,
        "Static-Opt-FixedPF": static_opt_pf,
        "Dynamic-FixedPF": dynamic_pf,
        "Ideal-NoPF": ideal,
    }


def run_matrix_file(config_path: Path, output_path: Path) -> Tuple[ExperimentRow, ...]:
    rows = run_matrix(load_runner_config(config_path))
    write_matrix(output_path, rows)
    return rows
