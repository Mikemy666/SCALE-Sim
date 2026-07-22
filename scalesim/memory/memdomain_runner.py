"""End-to-end runner for the canonical MemDomain baseline matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from scalesim.memory.streaming_residency import StreamingLoadPlan, StreamingResidencyEngine
from scalesim.memory.prefetch_policy import (
    BankAwarePrefetchPolicy,
    BankSnapshot,
    NaivePrefetchPolicy,
    NoPrefetchPolicy,
    PrefetchAction,
    PrefetchDecision,
)
from scalesim.memory.unified_bank_domain import UnifiedBankDomain, UnifiedMemoryRequest
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


def _decisions(
    baseline: Baseline,
    config: RunnerConfig,
    manager: ChunkResidencyManager,
    pressure: Mapping[int, BankPressure],
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
    decisions = []
    free = {
        bank: manager.mapping.bank_capacity[bank] - manager.mapping.bank_occupied[bank]
        for bank in range(config.resources.bank_count)
    }
    ordered_chunks = tuple(sorted(
        config.chunks, key=lambda item: (item.use_cycle, item.chunk_id)
    ))
    for index, chunk in enumerate(ordered_chunks):
        trigger_index = index - config.prefetch_window
        issue = 0 if trigger_index < 0 else ordered_chunks[trigger_index].use_cycle
        while True:
            snapshot = BankSnapshot(issue, pressure, free)
            estimate = max(1, int(
                (chunk.size_bytes + config.resources.bandwidth_bytes_per_cycle - 1)
                // config.resources.bandwidth_bytes_per_cycle
            ))
            decision = policy.decide(chunk, snapshot, estimate, config.static_weight_banks)
            if decision.action != PrefetchAction.DELAY:
                break
            issue = int(decision.issue_cycle)
        decisions.append(decision)
        if decision.action == PrefetchAction.PREFETCH:
            for bank in decision.target_banks:
                # Mirror P3's conservative reservation for subsequent online decisions.
                free[bank] = max(0, free[bank] - (chunk.size_bytes // len(decision.target_banks)))
    return tuple(decisions)


def run_raw_baseline(config: RunnerConfig, baseline: Baseline) -> ExperimentRow:
    if baseline not in {
        Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF, Baseline.DYNAMIC_NOPF,
        Baseline.DYNAMIC_NAIVEPF, Baseline.MEMDOMAIN_RAW,
    }:
        raise ValueError("run_raw_baseline accepts only measured baseline kinds")
    dynamic = baseline in {
        Baseline.DYNAMIC_NOPF, Baseline.DYNAMIC_NAIVEPF, Baseline.MEMDOMAIN_RAW,
    }
    mapping = VirtualBankMappingTable(
        config.resources, "conflict_aware" if baseline == Baseline.MEMDOMAIN_RAW else (
            "least_occupied" if dynamic else "round_robin"
        )
    )
    # Policy planning reads this mapping's capacity view; P9 owns execution.
    manager = ChunkResidencyManager(mapping)
    domain = UnifiedBankDomain(config.resources, config.interleave_bytes)
    pressure = _compute_pressure(config, domain)
    decisions = _decisions(baseline, config, manager, pressure)
    by_chunk = {chunk.chunk_id: chunk for chunk in config.chunks}
    plans = []
    for decision in decisions:
        preferred = tuple(decision.target_banks)
        if baseline in (Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF):
            preferred = config.static_weight_banks
        chunk = by_chunk[decision.chunk_id]
        if decision.action == PrefetchAction.PREFETCH:
            plans.append(StreamingLoadPlan(
                chunk, int(decision.issue_cycle), "prefetch", preferred,
            ))
        else:
            plans.append(StreamingLoadPlan(
                chunk, chunk.use_cycle, "demand", preferred,
            ))

    compute_only = domain.simulate(config.compute_requests)
    residency = StreamingResidencyEngine(domain, mapping).run(
        plans, config.compute_requests, pressure
    )
    full = residency.memory_report
    mapping_stats = mapping.statistics()
    bank_metrics = _bank_metrics(full)
    demand_stall = sum(
        item.miss_stall_cycles for item in residency.chunks
        if item.classification == "demand_miss"
    )
    late_stall = sum(
        item.miss_stall_cycles for item in residency.chunks
        if item.classification == "late"
    )
    base_bank_stall = max(
        (service.queue_wait_cycles for service in compute_only.services), default=0
    )
    interference = _compute_interference(config, full, compute_only)
    mapping_overhead = mapping_stats.mapping_count * config.mapping_overhead_per_object if dynamic else 0
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
    return ExperimentRow(
        schema_version=1,
        experiment_id=config.experiment_id,
        workload_name=config.workload_name,
        workload_hash=workload_digest(config.payload),
        baseline=baseline.value,
        candidate_source="measured",
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
    )


def run_best_static_baseline(
    config: RunnerConfig, baseline: Baseline
) -> ExperimentRow:
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
        candidates.append((run_raw_baseline(candidate_config, baseline), group))
    selected, group = min(
        candidates, key=lambda item: (item[0].total_cycles, item[1])
    )
    from dataclasses import replace
    return replace(
        selected,
        candidate_source="exhaustive_cyclic_static_weight_groups:" + ":".join(map(str, group)),
    )


def run_matrix(config: RunnerConfig) -> Tuple[ExperimentRow, ...]:
    raw = [
        run_best_static_baseline(config, Baseline.STATIC_NOPF),
        run_best_static_baseline(config, Baseline.STATIC_NAIVEPF),
        run_raw_baseline(config, Baseline.DYNAMIC_NOPF),
        run_raw_baseline(config, Baseline.DYNAMIC_NAIVEPF),
        run_raw_baseline(config, Baseline.MEMDOMAIN_RAW),
    ]
    by_name = {row.baseline: row for row in raw}
    safe = derive_selected_row(
        Baseline.MEMDOMAIN_SAFE,
        [by_name[Baseline.STATIC_NOPF.value], by_name[Baseline.DYNAMIC_NOPF.value],
         by_name[Baseline.MEMDOMAIN_RAW.value]],
        [Baseline.STATIC_NOPF, Baseline.DYNAMIC_NOPF, Baseline.MEMDOMAIN_RAW],
    )
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
