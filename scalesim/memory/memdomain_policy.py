"""Policy comparison contracts for the MemDomain simulator.

This module deliberately contains no bank simulator.  It defines the common
resource and objective contract used to compare static, oracle, runtime, and
safe-dynamic results.  Keeping the contract independent prevents individual
policies from changing the accounting rules to obtain a preferred result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple


@dataclass(frozen=True)
class ResourceBudget:
    """Hardware resources that must be identical in a fair comparison."""

    bank_count: int
    capacity_bytes: int
    bandwidth_bytes_per_cycle: float
    ports_per_bank: int
    request_buffer_depth: int

    def __post_init__(self) -> None:
        for name in ("bank_count", "capacity_bytes", "ports_per_bank", "request_buffer_depth"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(self.bandwidth_bytes_per_cycle) <= 0:
            raise ValueError("bandwidth_bytes_per_cycle must be positive")
        if self.request_buffer_depth < self.ports_per_bank:
            raise ValueError("request_buffer_depth must cover all Bank ports")


@dataclass(frozen=True)
class CycleBreakdown:
    """One end-to-end objective shared by every policy."""

    compute: int
    bank_stall: int = 0
    weight_load_stall: int = 0
    prefetch_miss_stall: int = 0
    prefetch_interference_stall: int = 0
    mapping_overhead: int = 0
    communication_stall: int = 0
    other_stall: int = 0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if int(value) < 0:
                raise ValueError(f"cycle component {name} must be non-negative")

    def as_dict(self) -> Mapping[str, int]:
        return {
            "compute": int(self.compute),
            "bank_stall": int(self.bank_stall),
            "weight_load_stall": int(self.weight_load_stall),
            "prefetch_miss_stall": int(self.prefetch_miss_stall),
            "prefetch_interference_stall": int(self.prefetch_interference_stall),
            "mapping_overhead": int(self.mapping_overhead),
            "communication_stall": int(self.communication_stall),
            "other_stall": int(self.other_stall),
        }

    @property
    def total(self) -> int:
        return sum(self.as_dict().values())


@dataclass(frozen=True)
class PolicyResult:
    name: str
    kind: str
    resources: ResourceBudget
    cycles: CycleBreakdown
    allocation: Tuple[int, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy result name must not be empty")
        if self.kind not in {"static", "runtime_dynamic", "oracle_dynamic", "safe_dynamic"}:
            raise ValueError(f"unsupported policy kind: {self.kind}")
        if self.allocation and sum(self.allocation) != self.resources.bank_count:
            raise ValueError("allocation must conserve the physical bank count")
        if self.allocation and any(int(count) <= 0 for count in self.allocation):
            raise ValueError("every allocated bank group must be positive")

    @property
    def total_cycles(self) -> int:
        return self.cycles.total


def _materialize(results: Iterable[PolicyResult], label: str) -> Tuple[PolicyResult, ...]:
    values = tuple(results)
    if not values:
        raise ValueError(f"{label} candidate set must not be empty")
    expected = values[0].resources
    mismatched = [item.name for item in values if item.resources != expected]
    if mismatched:
        raise ValueError(f"resource mismatch in {label}: {', '.join(mismatched)}")
    return values


def best_static(results: Iterable[PolicyResult]) -> PolicyResult:
    candidates = _materialize(results, "static")
    if any(item.kind != "static" for item in candidates):
        raise ValueError("best_static accepts only static candidates")
    return min(candidates, key=lambda item: (item.total_cycles, item.name))


def select_oracle_dynamic(
    static_results: Iterable[PolicyResult],
    dynamic_results: Iterable[PolicyResult],
) -> PolicyResult:
    """Select the common-objective oracle from a superset containing statics."""

    statics = _materialize(static_results, "static")
    dynamics = _materialize(dynamic_results, "dynamic")
    if statics[0].resources != dynamics[0].resources:
        raise ValueError("static and dynamic candidates use different resources")
    if any(item.kind not in {"runtime_dynamic", "oracle_dynamic"} for item in dynamics):
        raise ValueError("invalid dynamic candidate kind")
    selected = min((*statics, *dynamics), key=lambda item: (item.total_cycles, item.name))
    return PolicyResult(
        name=f"oracle:{selected.name}",
        kind="oracle_dynamic",
        resources=selected.resources,
        cycles=selected.cycles,
        allocation=selected.allocation,
        metadata={"selected_candidate": selected.name, "fallback_to_static": selected.kind == "static"},
    )


def select_safe_dynamic(
    static_results: Iterable[PolicyResult], runtime_result: PolicyResult
) -> PolicyResult:
    """Return runtime dynamic only when it beats the best static end-to-end."""

    static = best_static(static_results)
    if runtime_result.kind != "runtime_dynamic":
        raise ValueError("safe dynamic requires one runtime_dynamic candidate")
    if runtime_result.resources != static.resources:
        raise ValueError("runtime dynamic and static baseline use different resources")
    use_runtime = runtime_result.total_cycles <= static.total_cycles
    selected = runtime_result if use_runtime else static
    return PolicyResult(
        name=f"safe:{selected.name}",
        kind="safe_dynamic",
        resources=selected.resources,
        cycles=selected.cycles,
        allocation=selected.allocation,
        metadata={
            "selected_candidate": selected.name,
            "fallback_to_static": not use_runtime,
            "runtime_total_cycles": runtime_result.total_cycles,
            "best_static_total_cycles": static.total_cycles,
        },
    )


def assert_theoretical_order(
    static_results: Iterable[PolicyResult],
    oracle_result: PolicyResult,
    safe_result: PolicyResult,
) -> None:
    static = best_static(static_results)
    for label, result in (("oracle", oracle_result), ("safe", safe_result)):
        if result.resources != static.resources:
            raise AssertionError(f"{label} result resource mismatch")
        if result.total_cycles > static.total_cycles:
            raise AssertionError(
                f"{label} total {result.total_cycles} exceeds best static {static.total_cycles}"
            )
