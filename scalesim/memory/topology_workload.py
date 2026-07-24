"""Convert SCALE-Sim MoE topology CSVs into P7/P9 runner workloads."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Mapping

from scalesim.memory.buckyball_memdomain import CONTRACT
from scalesim.memory.buckyball_compiler import compile_gemm_bank_plan

EXPERT_LAYER = re.compile(r"MoE-E(\d+)-FF([12])$")


def load_moe_topology(path: Path) -> Mapping[str, object]:
    layers = []
    experts = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or not row[0] or row[0] == "Layer":
                continue
            if len(row) < 4:
                raise ValueError(f"malformed topology row: {row}")
            name = row[0].strip()
            m, n, k = (int(row[index]) for index in range(1, 4))
            layers.append((name, m, n, k))
            match = EXPERT_LAYER.fullmatch(name)
            if match:
                expert_id, part = int(match.group(1)), int(match.group(2))
                experts[(expert_id, part)] = (m, n, k)
    expert_ids = sorted({expert for expert, _ in experts})
    if not expert_ids or expert_ids != list(range(len(expert_ids))):
        raise ValueError("expert IDs must be contiguous from zero")
    for expert in expert_ids:
        if (expert, 1) not in experts or (expert, 2) not in experts:
            raise ValueError(f"expert {expert} must contain FF1 and FF2")
        if experts[(expert, 1)][0] != experts[(expert, 2)][0]:
            raise ValueError(f"expert {expert} FF1/FF2 token counts differ")
    return {"layers": tuple(layers), "experts": experts, "expert_ids": tuple(expert_ids)}


def generate_topology_runner_payload(
    path: Path,
    model_class: str,
    chunk_size_bytes: int = 2 * 1024,
    ia_bytes_per_element: int = 1,
    weight_bytes_per_element: int = 1,
    accumulator_bytes_per_element: int = 4,
    output_bytes_per_element: int = 1,
    accumulator_mode: str = "banked_rmw",
    top_k: int = 1,
    num_gpus: int = 1,
) -> Mapping[str, object]:
    if model_class not in {"homogeneous", "heterogeneous"}:
        raise ValueError("model_class must be homogeneous or heterogeneous")
    if (chunk_size_bytes <= 0 or min(
            ia_bytes_per_element, weight_bytes_per_element,
            accumulator_bytes_per_element, output_bytes_per_element,
            top_k, num_gpus) <= 0):
        raise ValueError("chunk and precision sizes must be positive")
    if accumulator_mode != "banked_rmw":
        raise ValueError("DATE2 target architecture requires banked_rmw")
    topology = load_moe_topology(path)
    experts = topology["experts"]
    token_counts = tuple(experts[(expert, 1)][0] for expert in topology["expert_ids"])
    if sum(token_counts) % top_k:
        raise ValueError("expert assignments must be divisible by Top-K")
    total_tokens = sum(token_counts) // top_k

    # Schedule weight use from the analytical work of the owning FFN stage.
    # The old fixed +8-cycle spacing made a 16 KiB transfer physically unable
    # to meet any paper Window and forced every prefetch to be late.
    array_macs_per_cycle = CONTRACT.tile_size * CONTRACT.tile_size
    non_expert_cycles = sum(
        max(1, (m * n * k + array_macs_per_cycle - 1) // array_macs_per_cycle)
        for name, m, n, k in topology["layers"] if not EXPERT_LAYER.fullmatch(name)
    )
    chunks = []
    stage_requests = []
    compiler_plans = []
    address = 0
    use_cycle = non_expert_cycles + 32
    for expert in topology["expert_ids"]:
        for part in (1, 2):
            m, n, k = experts[(expert, part)]
            weight_bytes = n * k * weight_bytes_per_element
            chunk_count = max(1, (weight_bytes + chunk_size_bytes - 1) // chunk_size_bytes)
            stage_cycles = max(
                1, (m * n * k + array_macs_per_cycle - 1) // array_macs_per_cycle
            )
            tile_spacing = max(1, stage_cycles // chunk_count)
            stage_start = use_cycle
            compiled = compile_gemm_bank_plan(m, n, k)
            compiler_plans.append({
                "layer": f"MoE-E{expert}-FF{part}",
                "ia_banks": compiled.allocation.ia,
                "weight_banks": compiled.allocation.weight,
                "oa_banks": compiled.allocation.oa,
                "acc_banks": compiled.allocation.accumulator,
                "predicted_cycles": compiled.objective.total_cycles,
                "predicted_exposed_stall":
                    compiled.objective.exposed_memory_stall_cycles,
                "static_predicted_cycles": compiled.static_cycles,
                "predicted_gain": compiled.predicted_gain,
                "fallback_used": compiled.fallback_used,
            })
            ia_start = (expert * 3 + (part - 1) * 5) % CONTRACT.bank_count
            oa_start = (ia_start + 10) % CONTRACT.bank_count
            stage_requests.extend((
                {
                    "request_id": f"compute_e{expert}_ff{part}_ia",
                    "issue_cycle": stage_start,
                    "tensor_type": "ia", "object_id": f"ia_e{expert}_ff{part}",
                    "address": expert * 4096 + part * 1024,
                    "size_bytes": max(1024, m * k * ia_bytes_per_element),
                    "kind": "read",
                    "preferred_banks": [
                        (ia_start + offset) % CONTRACT.bank_count
                        for offset in range(CONTRACT.bank_count)
                    ],
                    "bank_group_size": compiled.allocation.ia,
                },
                {
                    "request_id": f"compute_e{expert}_ff{part}_oa",
                    "issue_cycle": stage_start + stage_cycles // 2,
                    "tensor_type": "oa",
                    "object_id": f"oa_e{expert}_ff{part}",
                    "address": expert * 4096 + part * 1024,
                    "size_bytes": max(1024, m * n * output_bytes_per_element),
                    "kind": "write",
                    "preferred_banks": [
                        (oa_start + offset) % CONTRACT.bank_count
                        for offset in range(CONTRACT.bank_count)
                    ],
                    "bank_group_size": compiled.allocation.oa,
                },
            ))
            # One representative 16x16 output tile follows the confirmed
            # Buckyball target protocol.  The first K tile overwrites; every
            # later K tile performs an atomic 3-cycle-per-row AccPipe RMW.
            # Full tile multiplicity is recorded in provenance and expanded
            # by the trace-driven scheduler, avoiding enormous JSON files.
            k_tiles = max(1, (k + CONTRACT.tile_size - 1) // CONTRACT.tile_size)
            output_tiles = (
                max(1, (m + CONTRACT.tile_size - 1) // CONTRACT.tile_size)
                * max(1, (n + CONTRACT.tile_size - 1) // CONTRACT.tile_size)
            )
            tile_step = max(1, stage_cycles // (output_tiles * k_tiles))
            acc_banks = list(range(15, 30))
            for k_tile in range(k_tiles):
                acc_cycle = stage_start + (k_tile + 1) * tile_step
                stage_requests.append({
                    "request_id": f"acc_e{expert}_ff{part}_k{k_tile}",
                    "issue_cycle": acc_cycle,
                    "tensor_type": "accumulator",
                    "object_id": f"acc_e{expert}_ff{part}_tile0",
                    "address": expert * 8192 + part * 2048,
                    "size_bytes": CONTRACT.accumulator_tile_bytes,
                    "kind": "write",
                    "wmode": int(k_tile > 0),
                    "preferred_banks": acc_banks,
                    "bank_group_size": CONTRACT.acc_stripe_banks,
                    "repeat_count": output_tiles,
                    "repeat_interval": k_tiles * tile_step,
                    "address_stride": CONTRACT.accumulator_tile_bytes,
                })
            remaining = weight_bytes
            tile = 0
            while remaining:
                size = min(chunk_size_bytes, remaining)
                chunks.append({
                    "chunk_id": f"e{expert}_ff{part}_c{tile}",
                    "expert_id": expert,
                    "ffn_part": part,
                    "tile_id": tile,
                    "size_bytes": size,
                    "use_cycle": stage_start + (tile + 1) * tile_spacing,
                    "logical_address": address,
                    "bank_group_size": max(
                        1, (size + CONTRACT.bank_bytes - 1)
                        // CONTRACT.bank_bytes
                    ),
                })
                remaining -= size
                address += size
                tile += 1
            use_cycle = stage_start + stage_cycles

    compute_cycles = use_cycle + 32
    name = Path(path).stem
    return {
        "experiment_id": f"p10-overall-{name.lower()}",
        "workload_name": name,
        "compute_cycles": compute_cycles,
        "compute_intervals": [[0, compute_cycles]],
        "hardware": {
            "bank_count": CONTRACT.bank_count,
            "capacity_bytes": CONTRACT.capacity_bytes,
            "bandwidth_bytes_per_cycle":
                CONTRACT.aggregate_bank_bandwidth_bytes_per_cycle,
            "ports_per_bank": CONTRACT.ports_per_bank,
            "request_buffer_depth": 32,
            "interleave_bytes": CONTRACT.bank_width_bits // 8,
            "bank_width_bits": CONTRACT.bank_width_bits,
            "bank_entries": CONTRACT.bank_entries,
            "offchip_bandwidth_bits_per_cycle":
                CONTRACT.offchip_bandwidth_bits_per_cycle,
            "offchip_startup_cycles": CONTRACT.offchip_startup_cycles,
        },
        "policy": {
            "mapping_overhead_per_object": 0,
            "prefetch_window": 2,
            "queue_threshold": 2,
            "conflict_threshold": 4,
            "busy_threshold": 32,
            "static_ia_banks": list(range(0, 5)),
            "static_weight_banks": list(range(5, 10)),
            "static_oa_banks": list(range(10, 15)),
            "static_acc_banks": list(range(15, 30)),
            "acc_stripe_banks": CONTRACT.acc_stripe_banks,
        },
        "topology_provenance": {
            "source_path": str(Path(path)),
            "model_class": model_class,
            "top_k": top_k,
            "routing_mode": "topology_counts",
            "token_counts": list(token_counts),
            "total_tokens": total_tokens,
            "chunk_size_bytes": chunk_size_bytes,
            "original_model_format": "FP32",
            "compute_format": "INT8xINT8_INT32",
            "ia_bytes_per_element": ia_bytes_per_element,
            "weight_bytes_per_element": weight_bytes_per_element,
            "accumulator_bytes_per_element": accumulator_bytes_per_element,
            "output_bytes_per_element": output_bytes_per_element,
            "accumulator_mode": accumulator_mode,
            "tile_size": CONTRACT.tile_size,
            "accumulator_rmw_cycles": CONTRACT.rmw_cycles,
            "requant_tile_cycles": CONTRACT.requant_tile_cycles,
            "acc_fragmentation_banks": CONTRACT.static_acc_fragmentation_banks,
            "weight_scale_divisor": 1,
            "paper_scale_performance_claim": False,
            "streaming_fixed_capacity": True,
        },
        "system": {
            "num_gpus": num_gpus,
            "communication_latency_cycles": 20,
            "communication_bandwidth_bytes_per_cycle": 128,
            "remote_token_fraction": 0.0 if num_gpus == 1 else 0.5,
            "token_payload_bytes": 384 * ia_bytes_per_element,
        },
        "chunks": chunks,
        "compiler_bank_plans": compiler_plans,
        "compute_requests": [
            {"request_id": "compute_frontend_ia", "issue_cycle": 0, "tensor_type": "ia",
             "object_id": "ia", "address": 0,
             "size_bytes": 256 * 384 * ia_bytes_per_element,
             "kind": "read", "preferred_banks": list(range(5))},
            *stage_requests,
        ],
    }
