"""Uniform DATE2 network scaling and Buckyball compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scalesim.memory.buckyball_memdomain import CONTRACT
from scalesim.memory.topology_workload import load_moe_topology


@dataclass(frozen=True)
class NetworkCompatibility:
    model: str
    expert_count: int
    total_tokens: int
    hidden_size: int
    padded_rows: int
    homogeneous_expert_weights: bool
    compatible: bool


def round_up(value: int, alignment: int = CONTRACT.tile_size) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def validate_network_set(paths: Sequence[Path]) -> Mapping[str, NetworkCompatibility]:
    """Require one Top-1 token distribution and one scale across all models.

    M may require edge padding. N/K must be tile aligned except Router_logits,
    whose N=expert_count is deliberately padded by the array.
    """
    results = {}
    reference_tokens = None
    reference_hidden = None
    for path in paths:
        topology = load_moe_topology(path)
        experts = topology["experts"]
        expert_ids = topology["expert_ids"]
        token_counts = tuple(experts[(expert, 1)][0] for expert in expert_ids)
        hidden_candidates = {
            experts[(expert, 1)][2] for expert in expert_ids
        } | {
            experts[(expert, 2)][1] for expert in expert_ids
        }
        if len(hidden_candidates) != 1:
            raise ValueError(f"{path.stem}: inconsistent scaled hidden size")
        hidden = next(iter(hidden_candidates))
        if hidden % CONTRACT.tile_size:
            raise ValueError(f"{path.stem}: hidden size is not tile aligned")
        if reference_tokens is None:
            reference_tokens = token_counts
            reference_hidden = hidden
        if token_counts != reference_tokens:
            raise ValueError("all DATE2 models must use the same Top-1 imbalance")
        if hidden != reference_hidden:
            raise ValueError("all DATE2 models must use the same dimension scale")

        projection_shapes = tuple(
            (experts[(expert, 1)][1], experts[(expert, 2)][2])
            for expert in expert_ids
        )
        for name, _m, n, k in topology["layers"]:
            if name != "Router_logits" and (n % 16 or k % 16):
                raise ValueError(f"{path.stem}:{name} is not 16-wide aligned")
        results[path.stem] = NetworkCompatibility(
            model=path.stem,
            expert_count=len(expert_ids),
            total_tokens=sum(token_counts),
            hidden_size=hidden,
            padded_rows=sum(round_up(value) - value for value in token_counts),
            homogeneous_expert_weights=len(set(projection_shapes)) == 1,
            compatible=True,
        )
    return results
