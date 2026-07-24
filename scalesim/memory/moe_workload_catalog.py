"""Literature-backed MoE model catalog and scaled runner-workload generator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple


ARCHITECTURE_CLASSES = {"homogeneous", "heterogeneous_size", "shared_routed"}


@dataclass(frozen=True)
class MoEModelSpec:
    model_id: str
    display_name: str
    architecture_class: str
    source_title: str
    source_url: str
    source_locator: str
    hidden_size: int
    routed_expert_intermediate_sizes: Tuple[int, ...]
    top_k: int
    projection_count: int
    shared_expert_intermediate_sizes: Tuple[int, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.architecture_class not in ARCHITECTURE_CLASSES:
            raise ValueError(f"unsupported architecture class: {self.architecture_class}")
        if not self.model_id or not self.source_title or not self.source_locator:
            raise ValueError("model and source identity must not be empty")
        if not self.source_url.startswith("https://"):
            raise ValueError("source_url must be an HTTPS primary source")
        if self.hidden_size <= 0 or self.projection_count not in (2, 3):
            raise ValueError("invalid hidden size or FFN projection count")
        if not self.routed_expert_intermediate_sizes:
            raise ValueError("at least one routed expert is required")
        if any(value <= 0 for value in self.routed_expert_intermediate_sizes):
            raise ValueError("expert intermediate sizes must be positive")
        if not 1 <= self.top_k <= len(self.routed_expert_intermediate_sizes):
            raise ValueError("top_k exceeds routed expert count")
        unique = set(self.routed_expert_intermediate_sizes)
        if self.architecture_class == "homogeneous" and len(unique) != 1:
            raise ValueError("homogeneous model contains unequal routed experts")
        if self.architecture_class == "heterogeneous_size" and len(unique) < 2:
            raise ValueError("heterogeneous-size model requires unequal experts")
        if self.architecture_class == "shared_routed" and not self.shared_expert_intermediate_sizes:
            raise ValueError("shared-routed model requires shared expert dimensions")


def load_catalog(path: Path) -> Tuple[MoEModelSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported catalog schema")
    specs = []
    for item in payload["models"]:
        specs.append(MoEModelSpec(
            model_id=str(item["model_id"]),
            display_name=str(item["display_name"]),
            architecture_class=str(item["architecture_class"]),
            source_title=str(item["source_title"]),
            source_url=str(item["source_url"]),
            source_locator=str(item["source_locator"]),
            hidden_size=int(item["hidden_size"]),
            routed_expert_intermediate_sizes=tuple(
                int(value) for value in item["routed_expert_intermediate_sizes"]
            ),
            top_k=int(item["top_k"]),
            projection_count=int(item["projection_count"]),
            shared_expert_intermediate_sizes=tuple(
                int(value) for value in item.get("shared_expert_intermediate_sizes", ())
            ),
            notes=str(item.get("notes", "")),
        ))
    identifiers = [spec.model_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate model_id in catalog")
    return tuple(specs)


def _scaled(value: int, divisor: int, alignment: int) -> int:
    raw = max(alignment, int(round(float(value) / float(divisor))))
    return max(alignment, int(round(float(raw) / alignment)) * alignment)


def generate_runner_payload(
    spec: MoEModelSpec,
    dimension_divisor: int = 64,
    alignment: int = 4,
    precision_bytes: int = 1,
    chunk_size_bytes: int = 16384,
    tokens: int = 32,
) -> Mapping[str, object]:
    if min(dimension_divisor, alignment, precision_bytes, chunk_size_bytes, tokens) <= 0:
        raise ValueError("generation parameters must be positive")
    scaled_hidden = _scaled(spec.hidden_size, dimension_divisor, alignment)
    scaled_routed = tuple(
        _scaled(value, dimension_divisor, alignment)
        for value in spec.routed_expert_intermediate_sizes
    )
    scaled_shared = tuple(
        _scaled(value, dimension_divisor, alignment)
        for value in spec.shared_expert_intermediate_sizes
    )

    chunks = []
    address = 0
    use_cycle = 20
    chunk_sequence = 0

    def add_expert(expert_id: int, intermediate: int, expert_kind: str) -> None:
        nonlocal address, use_cycle, chunk_sequence
        projection_shapes = [(scaled_hidden, intermediate)] * (spec.projection_count - 1)
        projection_shapes.append((intermediate, scaled_hidden))
        for projection, (rows, columns) in enumerate(projection_shapes):
            weight_bytes = rows * columns * precision_bytes
            remaining = weight_bytes
            while remaining:
                size = min(chunk_size_bytes, remaining)
                chunks.append({
                    "chunk_id": f"{expert_kind}_e{expert_id}_p{projection}_c{chunk_sequence}",
                    "expert_id": expert_id,
                    "expert_kind": expert_kind,
                    "ffn_part": 1 if projection < spec.projection_count - 1 else 2,
                    "tile_id": chunk_sequence,
                    "size_bytes": size,
                    "use_cycle": use_cycle,
                    "logical_address": address,
                    "bank_group_size": 1,
                })
                address += size
                remaining -= size
                use_cycle += 8
                chunk_sequence += 1

    for expert_id, intermediate in enumerate(scaled_routed):
        add_expert(expert_id, intermediate, "routed")
    for shared_id, intermediate in enumerate(scaled_shared):
        add_expert(len(scaled_routed) + shared_id, intermediate, "shared")

    bank_count = 24
    # Fixed 64 KiB per physical Bank, independent of total model weights.
    capacity = bank_count * 64 * 1024
    compute_cycles = max(use_cycle + 20, tokens * len(scaled_routed))
    payload = {
        "experiment_id": f"catalog-{spec.model_id}",
        "workload_name": spec.model_id,
        "compute_cycles": compute_cycles,
        "compute_intervals": [[0, compute_cycles]],
        "hardware": {
            "bank_count": bank_count,
            "capacity_bytes": capacity,
            "bandwidth_bytes_per_cycle": 384,
            "ports_per_bank": 1,
            "request_buffer_depth": 32,
            "interleave_bytes": 64,
        },
        "policy": {
            "mapping_overhead_per_object": 1,
            "prefetch_window": 2,
            "queue_threshold": 2,
            "conflict_threshold": 4,
            "busy_threshold": 32,
            "static_weight_banks": list(range(8, 16)),
        },
        "model_provenance": {
            "model_id": spec.model_id,
            "architecture_class": spec.architecture_class,
            "source_title": spec.source_title,
            "source_url": spec.source_url,
            "source_locator": spec.source_locator,
            "original_hidden_size": spec.hidden_size,
            "original_routed_expert_intermediate_sizes": list(spec.routed_expert_intermediate_sizes),
            "original_shared_expert_intermediate_sizes": list(spec.shared_expert_intermediate_sizes),
            "original_top_k": spec.top_k,
            "projection_count": spec.projection_count,
            "dimension_divisor": dimension_divisor,
            "alignment": alignment,
            "scaled_hidden_size": scaled_hidden,
            "scaled_routed_expert_intermediate_sizes": list(scaled_routed),
            "scaled_shared_expert_intermediate_sizes": list(scaled_shared),
            "derived_workload": True,
            "paper_scale_performance_claim": False,
            "streaming_fixed_capacity": True,
            "original_model_format": "FP32",
            "compute_format": "INT8xINT8_INT32",
            "ia_bytes_per_element": 1,
            "weight_bytes_per_element": 1,
            "accumulator_bytes_per_element": 4,
            "output_bytes_per_element": 1,
            "accumulator_mode": "local",
        },
        "routing": {
            "tokens": tokens,
            "top_k": spec.top_k,
            "mode": "balanced",
            "seed": 0,
        },
        "chunks": chunks,
        "compute_requests": [
            {
                "request_id": "compute_ia",
                "issue_cycle": 0,
                "tensor_type": "ia",
                "object_id": "ia",
                "address": 0,
                "size_bytes": max(64, tokens * scaled_hidden * precision_bytes),
                "kind": "read",
                "preferred_banks": list(range(0, 8)),
            },
            {
                "request_id": "compute_oa",
                "issue_cycle": max(1, compute_cycles // 2),
                "tensor_type": "oa",
                "object_id": "oa",
                "address": 0,
                "size_bytes": max(64, tokens * scaled_hidden * precision_bytes),
                "kind": "write",
                "preferred_banks": list(range(16, 24)),
            },
        ],
    }
    return payload


def write_runner_payload(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
