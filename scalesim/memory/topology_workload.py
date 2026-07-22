"""Convert SCALE-Sim MoE topology CSVs into P7/P9 runner workloads."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Mapping


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
    chunk_size_bytes: int = 16 * 1024,
    precision_bytes: int = 2,
    weight_scale_divisor: int = 8,
    top_k: int = 1,
    num_gpus: int = 1,
) -> Mapping[str, object]:
    if model_class not in {"homogeneous", "heterogeneous"}:
        raise ValueError("model_class must be homogeneous or heterogeneous")
    if (chunk_size_bytes <= 0 or precision_bytes <= 0 or
            weight_scale_divisor <= 0 or top_k <= 0 or num_gpus <= 0):
        raise ValueError("chunk and precision sizes must be positive")
    topology = load_moe_topology(path)
    experts = topology["experts"]
    token_counts = tuple(experts[(expert, 1)][0] for expert in topology["expert_ids"])
    if sum(token_counts) % top_k:
        raise ValueError("expert assignments must be divisible by Top-K")
    total_tokens = sum(token_counts) // top_k

    chunks = []
    address = 0
    use_cycle = 32
    for expert in topology["expert_ids"]:
        for part in (1, 2):
            _, n, k = experts[(expert, part)]
            raw_weight_bytes = n * k * precision_bytes
            remaining = (raw_weight_bytes + weight_scale_divisor - 1) // weight_scale_divisor
            tile = 0
            while remaining:
                size = min(chunk_size_bytes, remaining)
                chunks.append({
                    "chunk_id": f"e{expert}_ff{part}_c{tile}",
                    "expert_id": expert,
                    "ffn_part": part,
                    "tile_id": tile,
                    "size_bytes": size,
                    "use_cycle": use_cycle,
                    "logical_address": address,
                    "bank_group_size": max(1, (size + 64 * 1024 - 1) // (64 * 1024)),
                })
                remaining -= size
                address += size
                use_cycle += 8
                tile += 1

    # The analytical compute envelope is deterministic and common across all
    # baseline policies; P10 compares memory behavior under the same envelope.
    macs = sum(m * n * k for _, m, n, k in topology["layers"])
    compute_cycles = max(use_cycle + 32, (macs + 255) // 256)
    name = Path(path).stem
    return {
        "experiment_id": f"p10-overall-{name.lower()}",
        "workload_name": name,
        "compute_cycles": compute_cycles,
        "compute_intervals": [[0, compute_cycles]],
        "hardware": {
            "bank_count": 24,
            "capacity_bytes": 24 * 64 * 1024,
            "bandwidth_bytes_per_cycle": 384,
            "ports_per_bank": 1,
            "request_buffer_depth": 32,
            "interleave_bytes": 1024,
        },
        "policy": {
            "mapping_overhead_per_object": 1,
            "prefetch_window": 2,
            "queue_threshold": 2,
            "conflict_threshold": 4,
            "busy_threshold": 32,
            "static_weight_banks": list(range(8, 16)),
        },
        "topology_provenance": {
            "source_path": str(Path(path)),
            "model_class": model_class,
            "top_k": top_k,
            "routing_mode": "topology_counts",
            "token_counts": list(token_counts),
            "total_tokens": total_tokens,
            "chunk_size_bytes": chunk_size_bytes,
            "precision_bytes": precision_bytes,
            "weight_scale_divisor": weight_scale_divisor,
            "paper_scale_performance_claim": False,
            "streaming_fixed_capacity": True,
        },
        "system": {
            "num_gpus": num_gpus,
            "communication_latency_cycles": 20,
            "communication_bandwidth_bytes_per_cycle": 128,
            "remote_token_fraction": 0.0 if num_gpus == 1 else 0.5,
            "token_payload_bytes": 384 * precision_bytes,
        },
        "chunks": chunks,
        "compute_requests": [
            {"request_id": "compute_ia", "issue_cycle": 0, "tensor_type": "ia",
             "object_id": "ia", "address": 0, "size_bytes": 256 * 384 * precision_bytes,
             "kind": "read", "preferred_banks": list(range(8))},
            {"request_id": "compute_accumulator", "issue_cycle": compute_cycles // 2,
             "tensor_type": "accumulator", "object_id": "accumulator", "address": 0,
             "size_bytes": 256 * 384 * precision_bytes, "kind": "write",
             "preferred_banks": list(range(16, 24))},
        ],
    }
