"""Analytical Peer-NPU and dependency timeline for DATE3 Expert Parallelism."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Mapping, Tuple

from scalesim.memory.date3_ep_model import DetailedNPUWorkload
from scalesim.memory.date3_ep_model import localize_detailed_npu
from scalesim.memory.memdomain_experiment import ExperimentRow, validate_matrix
from scalesim.memory.memdomain_runner import RunnerConfig


@dataclass(frozen=True)
class EPSystemTimeline:
    detailed_ready_cycle: int
    peer_ready_cycle: int
    result_ready_cycle: int
    combine_cycles: int
    total_cycles: int
    exposed_remote_wait_cycles: int
    dispatch_bytes: int
    dispatch_cycles: int
    return_bytes: int
    return_cycles: int
    peer_rows: Tuple[Mapping[str, object], ...]
    timeline_rows: Tuple[Mapping[str, object], ...]
    combine_rows: Tuple[Mapping[str, object], ...]


def _transfer_cycles(byte_count: int, startup: int, bandwidth: float) -> int:
    return startup + int(ceil(byte_count / bandwidth)) if byte_count > 0 else 0


def build_ep_system_timeline(
    original: RunnerConfig,
    detailed: DetailedNPUWorkload,
    detailed_ready_cycle: int,
) -> EPSystemTimeline:
    """Compose detailed and analytical Peer paths by dependencies, never sums."""
    contract = detailed.contract
    system = original.payload.get("system", {})
    hardware = original.payload.get("hardware", {})
    payload_bytes = int(system.get("token_payload_bytes", 0))
    return_payload_bytes = int(system.get("result_payload_bytes", payload_bytes))
    comm_startup = int(system.get("communication_latency_cycles", 0))
    comm_bw = float(system.get("communication_bandwidth_bytes_per_cycle", 1))
    hbm_startup = int(hardware.get("offchip_startup_cycles", 0))
    hbm_bw_bits = float(hardware.get("offchip_bandwidth_bits_per_cycle", 8))
    hbm_bw = max(1.0, hbm_bw_bits / 8.0)
    combine_per_token = int(system.get("combine_cycles_per_token", 1))
    if min(payload_bytes, return_payload_bytes, comm_startup, hbm_startup,
           combine_per_token) < 0 or comm_bw <= 0:
        raise ValueError("invalid DATE3 EP system parameters")

    stages = {(item.expert_id, item.ffn_part): item for item in contract.stages}
    chunks_by_stage = {}
    for chunk in original.chunks:
        chunks_by_stage.setdefault((chunk.expert_id, chunk.ffn_part), []).append(chunk)
    routes_by_owner = {npu: [] for npu in range(contract.num_npus)}
    for route in detailed.routes:
        routes_by_owner[route.owner_npu].append(route)

    router_ready = min(
        (stage.original_start_cycle for stage in contract.stages), default=0
    )
    peer_rows = []
    timeline = [{
        "event": "detailed_npu_ready", "npu_id": contract.detailed_npu_id,
        "start_cycle": 0, "finish_cycle": detailed_ready_cycle,
        "detail": "PIVOT-CA local active experts",
    }]
    peer_finishes = []
    total_dispatch_bytes = total_dispatch_cycles = 0
    total_return_bytes = total_return_cycles = 0
    for npu in range(contract.num_npus):
        if npu == contract.detailed_npu_id:
            continue
        owner_routes = routes_by_owner[npu]
        active_experts = sorted({route.global_expert_id for route in owner_routes})
        remote_routes = [route for route in owner_routes if route.is_remote]
        dispatch_bytes = len(remote_routes) * payload_bytes
        dispatch_cycles = _transfer_cycles(dispatch_bytes, comm_startup, comm_bw)
        cursor = router_ready + dispatch_cycles
        total_dispatch_bytes += dispatch_bytes
        total_dispatch_cycles += dispatch_cycles
        if dispatch_cycles:
            timeline.append({
                "event": "token_dispatch", "npu_id": npu,
                "start_cycle": router_ready,
                "finish_cycle": router_ready + dispatch_cycles,
                "detail": f"replicas={len(remote_routes)};bytes={dispatch_bytes}",
            })

        npu_weight_bytes = npu_weight_cycles = npu_compute_cycles = 0
        for expert in active_experts:
            expert_start = cursor
            expert_weight = expert_weight_cycles = expert_compute = 0
            for part in (1, 2):
                stage = stages.get((expert, part))
                if stage is None:
                    raise ValueError(f"Peer expert {expert} lacks FF{part} metadata")
                weight_bytes = sum(
                    chunk.size_bytes for chunk in chunks_by_stage.get((expert, part), ())
                )
                weight_cycles = _transfer_cycles(weight_bytes, hbm_startup, hbm_bw)
                cursor += weight_cycles + stage.compute_cycles
                expert_weight += weight_bytes
                expert_weight_cycles += weight_cycles
                expert_compute += stage.compute_cycles
            npu_weight_bytes += expert_weight
            npu_weight_cycles += expert_weight_cycles
            npu_compute_cycles += expert_compute
            peer_rows.append({
                "npu_id": npu, "expert_id": expert,
                "token_count": contract.token_counts[expert],
                "weight_bytes": expert_weight,
                "weight_load_cycles": expert_weight_cycles,
                "compute_cycles": expert_compute,
                "start_cycle": expert_start, "finish_cycle": cursor,
                "execution_mode": "analytical_peer_owner_local",
            })

        return_bytes = len(remote_routes) * return_payload_bytes
        return_cycles = _transfer_cycles(return_bytes, comm_startup, comm_bw)
        compute_finish = cursor
        cursor += return_cycles
        total_return_bytes += return_bytes
        total_return_cycles += return_cycles
        if active_experts:
            timeline.append({
                "event": "peer_owner_work", "npu_id": npu,
                "start_cycle": router_ready + dispatch_cycles,
                "finish_cycle": compute_finish,
                "detail": (
                    f"experts={'|'.join(map(str, active_experts))};"
                    f"weight_bytes={npu_weight_bytes};"
                    f"weight_cycles={npu_weight_cycles};compute_cycles={npu_compute_cycles}"
                ),
            })
        if return_cycles:
            timeline.append({
                "event": "result_return", "npu_id": npu,
                "start_cycle": compute_finish, "finish_cycle": cursor,
                "detail": f"results={len(remote_routes)};bytes={return_bytes}",
            })
        peer_finishes.append(cursor)

    peer_ready = max(peer_finishes, default=0)
    result_ready = max(detailed_ready_cycle, peer_ready)
    token_count = len({route.token_id for route in detailed.routes})
    combine_cycles = token_count * combine_per_token
    total = result_ready + combine_cycles
    timeline.append({
        "event": "token_combine", "npu_id": contract.detailed_npu_id,
        "start_cycle": result_ready, "finish_cycle": total,
        "detail": f"tokens={token_count};top_k={contract.top_k}",
    })
    combine_rows = []
    by_token = {}
    for route in detailed.routes:
        by_token.setdefault(route.token_id, []).append(route)
    for token_id, routes in sorted(by_token.items()):
        combine_rows.append({
            "token_id": token_id,
            "source_npu": routes[0].source_npu,
            "expert_ids": "|".join(str(item.global_expert_id) for item in routes),
            "owner_npus": "|".join(str(item.owner_npu) for item in routes),
            "routing_weights": "|".join(str(item.routing_weight) for item in routes),
            "expected_results": len(routes),
            "combine_start_cycle": result_ready,
            "combine_complete_cycle": total,
        })
    return EPSystemTimeline(
        detailed_ready_cycle, peer_ready, result_ready, combine_cycles, total,
        max(0, result_ready - detailed_ready_cycle),
        total_dispatch_bytes, total_dispatch_cycles,
        total_return_bytes, total_return_cycles,
        tuple(peer_rows), tuple(timeline), tuple(combine_rows),
    )


def run_date3_ep_baseline_matrix(
    original: RunnerConfig,
) -> Tuple[ExperimentRow, ...]:
    """Run every DATE3 control inside the same EP system envelope as PIVOT."""
    # Local import keeps the EP timeline module independent of the baseline
    # implementation during module initialization.
    from scalesim.memory.memdomain_runner import run_matrix

    detailed = localize_detailed_npu(original)
    local_rows = run_matrix(detailed.config)
    rows = []
    for row in local_rows:
        detailed_ready = row.total_cycles - row.communication_stall_cycles
        timeline = build_ep_system_timeline(original, detailed, detailed_ready)
        rows.append(replace(
            row,
            communication_stall_cycles=timeline.exposed_remote_wait_cycles,
            other_stall_cycles=timeline.combine_cycles,
            total_cycles=timeline.total_cycles,
        ))
    return validate_matrix(rows)


def run_date3_ep_paper_controls(original: RunnerConfig):
    """Run named paper controls inside the same DATE3 EP envelope."""
    from scalesim.memory.memdomain_runner import (
        RawBaselineExecution, run_paper_control_executions,
    )

    detailed = localize_detailed_npu(original)
    local = run_paper_control_executions(detailed.config)
    result = {}
    for name, execution in local.items():
        row = execution.row
        detailed_ready = (
            row.total_cycles - row.communication_stall_cycles
            - row.other_stall_cycles
        )
        timeline = build_ep_system_timeline(original, detailed, detailed_ready)
        system_row = replace(
            row,
            communication_stall_cycles=timeline.exposed_remote_wait_cycles,
            other_stall_cycles=timeline.combine_cycles,
            total_cycles=timeline.total_cycles,
        )
        result[name] = RawBaselineExecution(
            system_row, execution.chunks, execution.memory_report
        )
    return result
