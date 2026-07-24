"""Implementable analytical Bank-col search for Buckyball GEMMs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Tuple

from scalesim.memory.buckyball_memdomain import (
    CONTRACT, STATIC_ALLOCATION, BankAllocation, CompilerObjective,
    legal_allocations, select_compiler_allocation,
)


@dataclass(frozen=True)
class GemmBankPlan:
    allocation: BankAllocation
    objective: CompilerObjective
    static_cycles: int
    predicted_gain: float
    fallback_used: bool


def _cycles(m: int, n: int, k: int, allocation: BankAllocation) -> Tuple[int, int, int]:
    dim = CONTRACT.tile_size
    mt, nt, kt = ceil(m / dim), ceil(n / dim), ceil(k / dim)
    compute = max(1, ceil(m * n * k / (dim * dim)))
    line = CONTRACT.per_bank_bandwidth_bytes_per_cycle
    ia = ceil((m * k) / (max(1, allocation.ia) * line))
    weight = ceil((n * k) / (max(1, allocation.weight) * line))
    oa = ceil((m * n) / (max(1, allocation.oa) * line))
    acc_groups = max(1, allocation.accumulator // CONTRACT.acc_stripe_banks)
    acc_per_output = dim + max(0, kt - 1) * CONTRACT.accumulator_rmw_tile_cycles
    acc = ceil(mt * nt * acc_per_output / acc_groups)
    requant = ceil(mt * nt * CONTRACT.requant_tile_cycles / max(1, allocation.oa))
    exposed = max(ia, weight) + acc + oa + requant
    # A deterministic conflict proxy: tile streams beyond available parallel
    # Banks serialize. It is used only after the two performance objectives.
    conflicts = (
        max(0, mt * kt - allocation.ia)
        + max(0, nt * kt - allocation.weight)
        + max(0, mt * nt - acc_groups)
    )
    return compute + exposed, exposed, conflicts


@lru_cache(maxsize=None)
def compile_gemm_bank_plan(m: int, n: int, k: int) -> GemmBankPlan:
    if min(m, n, k) <= 0:
        raise ValueError("GEMM dimensions must be positive")
    candidates = []
    # The static allocation is evaluated with only 12 usable ACC Banks; its
    # remaining three Banks are fixed-boundary fragmentation.
    for allocation in legal_allocations():
        total, exposed, conflicts = _cycles(m, n, k, allocation)
        objective = CompilerObjective(
            total, exposed, conflicts, allocation.total, allocation.as_tuple()
        )
        candidates.append((allocation, objective))
    selected, objective = select_compiler_allocation(candidates)
    static_objective = next(
        item for allocation, item in candidates if allocation == STATIC_ALLOCATION
    )
    return GemmBankPlan(
        selected, objective, static_objective.total_cycles,
        (static_objective.total_cycles / objective.total_cycles - 1.0)
        if objective.total_cycles else 0.0,
        selected == STATIC_ALLOCATION,
    )
