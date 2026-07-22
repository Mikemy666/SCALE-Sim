"""Adapters between simulator reports and the P1 comparison contract."""

from __future__ import annotations

from typing import Mapping, Sequence

from scalesim.memory.memdomain_policy import (
    CycleBreakdown,
    PolicyResult,
    ResourceBudget,
)


def _non_negative(report: Mapping[str, object], key: str) -> int:
    value = int(report.get(key, 0))
    if value < 0:
        raise ValueError(f"report field {key} must be non-negative")
    return value


def policy_result_from_report(
    name: str,
    kind: str,
    resources: ResourceBudget,
    report: Mapping[str, object],
    allocation: Sequence[int] = (),
) -> PolicyResult:
    """Convert an explicit common-schema simulator report to PolicyResult.

    Callers must provide components rather than a pre-computed TotalCycles so
    the adapter can reject accounting drift.
    """

    cycles = CycleBreakdown(
        compute=_non_negative(report, "ComputeCycles"),
        bank_stall=_non_negative(report, "BankStallCycles"),
        weight_load_stall=_non_negative(report, "WeightLoadStallCycles"),
        prefetch_miss_stall=_non_negative(report, "PrefetchMissStallCycles"),
        prefetch_interference_stall=_non_negative(report, "PrefetchInterferenceStallCycles"),
        mapping_overhead=_non_negative(report, "MappingOverheadCycles"),
        communication_stall=_non_negative(report, "CommunicationStallCycles"),
        other_stall=_non_negative(report, "OtherStallCycles"),
    )
    if "TotalCycles" in report and int(report["TotalCycles"]) != cycles.total:
        raise ValueError(
            f"TotalCycles accounting mismatch: report={int(report['TotalCycles'])}, "
            f"components={cycles.total}"
        )
    return PolicyResult(
        name=name,
        kind=kind,
        resources=resources,
        cycles=cycles,
        allocation=tuple(int(value) for value in allocation),
        metadata={"source": "simulator_common_schema"},
    )
