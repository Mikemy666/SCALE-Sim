"""Canonical MemDomain baseline matrix and deterministic CSV schema."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

from scalesim.memory.memdomain_policy import ResourceBudget


class Baseline(str, Enum):
    STATIC_NOPF = "Static-NoPF"
    STATIC_NAIVEPF = "Static-NaivePF"
    DYNAMIC_NOPF = "Dynamic-NoPF"
    DYNAMIC_NAIVEPF = "Dynamic-NaivePF"
    MEMDOMAIN_RAW = "MemDomain-Raw"
    MEMDOMAIN_SAFE = "MemDomain-Safe"
    ORACLE = "Oracle"


REQUIRED_BASELINES = tuple(item.value for item in Baseline)


class TheoreticalContractViolation(ValueError):
    """A measured matrix violates a required MemDomain dominance relation."""


@dataclass(frozen=True)
class ExperimentRow:
    schema_version: int
    experiment_id: str
    workload_name: str
    workload_hash: str
    baseline: str
    candidate_source: str
    bank_count: int
    capacity_bytes: int
    bandwidth_bytes_per_cycle: float
    ports_per_bank: int
    request_buffer_depth: int
    compute_cycles: int
    bank_stall_cycles: int
    weight_load_stall_cycles: int
    prefetch_miss_stall_cycles: int
    prefetch_interference_stall_cycles: int
    mapping_overhead_cycles: int
    communication_stall_cycles: int
    other_stall_cycles: int
    total_cycles: int
    bank_conflict_count: int = 0
    bank_conflict_rate: float = 0.0
    bank_imbalance: float = 0.0
    hotspot_bank_ratio: float = 0.0
    idle_bank_ratio: float = 0.0
    effective_bank_parallelism: float = 0.0
    max_queue_depth: int = 0
    prefetch_requests: int = 0
    prefetch_bytes: int = 0
    prefetch_coverage: float = 0.0
    prefetch_accuracy: float = 0.0
    timely_prefetch_ratio: float = 0.0
    late_prefetch_ratio: float = 0.0
    unused_prefetch_ratio: float = 0.0
    prefetch_occupancy_byte_cycles: int = 0
    compute_transfer_overlap_cycles: int = 0
    mapping_count: int = 0
    mapping_failures: int = 0
    peak_occupied_bytes: int = 0
    fallback_used: bool = False
    selected_candidate: str = ""
    mapping_work_cycles: int = 0
    mapping_hidden_cycles: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported experiment schema version")
        if self.baseline not in REQUIRED_BASELINES:
            raise ValueError(f"unsupported baseline: {self.baseline}")
        if not self.experiment_id or not self.workload_name or not self.workload_hash:
            raise ValueError("experiment/workload identity must not be empty")
        non_negative = (
            "bank_count", "capacity_bytes", "ports_per_bank", "request_buffer_depth",
            "compute_cycles", "bank_stall_cycles", "weight_load_stall_cycles",
            "prefetch_miss_stall_cycles", "prefetch_interference_stall_cycles",
            "mapping_overhead_cycles", "communication_stall_cycles", "other_stall_cycles",
            "total_cycles", "bank_conflict_count", "max_queue_depth", "prefetch_requests",
            "prefetch_bytes", "prefetch_occupancy_byte_cycles",
            "compute_transfer_overlap_cycles", "mapping_count", "mapping_failures",
            "peak_occupied_bytes", "mapping_work_cycles", "mapping_hidden_cycles",
        )
        if any(int(getattr(self, name)) < 0 for name in non_negative):
            raise ValueError("integer result fields must be non-negative")
        if self.bandwidth_bytes_per_cycle <= 0:
            raise ValueError("bandwidth must be positive")
        ratios = (
            "bank_conflict_rate", "hotspot_bank_ratio", "idle_bank_ratio",
            "prefetch_coverage", "prefetch_accuracy", "timely_prefetch_ratio",
            "late_prefetch_ratio", "unused_prefetch_ratio",
        )
        if any(not 0.0 <= float(getattr(self, name)) <= 1.0 for name in ratios):
            raise ValueError("ratio fields must be in [0, 1]")
        if self.bank_imbalance < 0 or self.effective_bank_parallelism < 0:
            raise ValueError("Bank imbalance/parallelism must be non-negative")
        expected = sum((
            self.compute_cycles,
            self.bank_stall_cycles,
            self.weight_load_stall_cycles,
            self.prefetch_miss_stall_cycles,
            self.prefetch_interference_stall_cycles,
            self.mapping_overhead_cycles,
            self.communication_stall_cycles,
            self.other_stall_cycles,
        ))
        if self.total_cycles != expected:
            raise ValueError(
                f"total cycle accounting mismatch for {self.baseline}: "
                f"reported={self.total_cycles}, components={expected}"
            )
        if self.mapping_work_cycles != (
            self.mapping_hidden_cycles + self.mapping_overhead_cycles
        ):
            raise ValueError(
                "mapping work must equal hidden plus exposed mapping cycles"
            )

    @property
    def resources(self) -> ResourceBudget:
        return ResourceBudget(
            self.bank_count, self.capacity_bytes, self.bandwidth_bytes_per_cycle,
            self.ports_per_bank, self.request_buffer_depth,
        )


def workload_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_experiment(rows: Sequence[ExperimentRow]) -> None:
    identities = {(row.experiment_id, row.workload_name, row.workload_hash) for row in rows}
    if len(identities) != 1:
        raise ValueError("matrix rows do not share one experiment/workload identity")
    resources = {row.resources for row in rows}
    if len(resources) != 1:
        raise ValueError("matrix rows do not share one hardware resource budget")


def derive_selected_row(
    baseline: Baseline,
    candidates: Sequence[ExperimentRow],
    allowed_baselines: Sequence[Baseline],
) -> ExperimentRow:
    if baseline not in (Baseline.MEMDOMAIN_SAFE, Baseline.ORACLE):
        raise ValueError("only Safe and Oracle rows may be derived")
    if not candidates:
        raise ValueError("derived row requires candidates")
    _same_experiment(candidates)
    allowed = {item.value for item in allowed_baselines}
    actual = {row.baseline for row in candidates}
    if actual != allowed:
        raise ValueError(f"candidate provenance mismatch: expected={sorted(allowed)}, actual={sorted(actual)}")
    selected = min(candidates, key=lambda row: (row.total_cycles, row.baseline))
    return replace(
        selected,
        baseline=baseline.value,
        candidate_source="|".join(sorted(actual)),
        fallback_used=(
            baseline == Baseline.MEMDOMAIN_SAFE
            and selected.baseline != Baseline.MEMDOMAIN_RAW.value
        ),
        selected_candidate=selected.baseline,
    )


def validate_matrix(rows: Iterable[ExperimentRow]) -> Tuple[ExperimentRow, ...]:
    values = tuple(rows)
    if len(values) != len(REQUIRED_BASELINES):
        raise ValueError(f"matrix must contain exactly {len(REQUIRED_BASELINES)} rows")
    _same_experiment(values)
    by_baseline = {row.baseline: row for row in values}
    if len(by_baseline) != len(values) or set(by_baseline) != set(REQUIRED_BASELINES):
        raise ValueError("matrix must contain every required baseline exactly once")

    safe = by_baseline[Baseline.MEMDOMAIN_SAFE.value]
    oracle = by_baseline[Baseline.ORACLE.value]
    static = by_baseline[Baseline.STATIC_NOPF.value]
    dynamic = by_baseline[Baseline.DYNAMIC_NOPF.value]
    raw = by_baseline[Baseline.MEMDOMAIN_RAW.value]
    implementable_safe_candidates = {
        Baseline.STATIC_NOPF.value,
        Baseline.STATIC_NAIVEPF.value,
        Baseline.DYNAMIC_NOPF.value,
        Baseline.DYNAMIC_NAIVEPF.value,
        Baseline.MEMDOMAIN_RAW.value,
    }
    if safe.selected_candidate not in implementable_safe_candidates:
        raise ValueError("Safe row has invalid selected-candidate provenance")
    selected_safe = by_baseline[safe.selected_candidate]
    _assert_derived_copy(safe, selected_safe, "Safe")
    if oracle.selected_candidate not in set(REQUIRED_BASELINES) - {Baseline.ORACLE.value}:
        raise ValueError("Oracle row has invalid selected-candidate provenance")
    selected_oracle = by_baseline[oracle.selected_candidate]
    _assert_derived_copy(oracle, selected_oracle, "Oracle")
    if oracle.total_cycles > min(row.total_cycles for row in values if row is not oracle):
        raise ValueError("Oracle is not the best candidate")
    return tuple(sorted(values, key=lambda row: REQUIRED_BASELINES.index(row.baseline)))


def validate_theoretical_contract(
    rows: Iterable[ExperimentRow],
) -> Tuple[ExperimentRow, ...]:
    """Enforce the DATE2 design-space containment contract.

    Structural validation remains separate so historical matrices can still be
    read and diagnosed. Every newly written matrix must additionally pass this
    function: dynamic placement contains the corresponding static placement,
    dynamic prefetch uses the same prefetch workload as static prefetch, and
    the final Safe policy is no worse than any implementable measured baseline.
    """
    values = validate_matrix(rows)
    by_baseline = {row.baseline: row for row in values}
    static = by_baseline[Baseline.STATIC_NOPF.value]
    static_pf = by_baseline[Baseline.STATIC_NAIVEPF.value]
    dynamic = by_baseline[Baseline.DYNAMIC_NOPF.value]
    dynamic_pf = by_baseline[Baseline.DYNAMIC_NAIVEPF.value]
    raw = by_baseline[Baseline.MEMDOMAIN_RAW.value]
    safe = by_baseline[Baseline.MEMDOMAIN_SAFE.value]

    violations = []
    if dynamic.total_cycles > static.total_cycles:
        violations.append(
            "Dynamic-NoPF must not exceed Static-NoPF "
            f"({dynamic.total_cycles} > {static.total_cycles})"
        )

    prefetch_identity_fields = ("prefetch_requests", "prefetch_bytes")
    mismatched = [
        name for name in prefetch_identity_fields
        if getattr(dynamic_pf, name) != getattr(static_pf, name)
    ]
    if mismatched:
        violations.append(
            "Dynamic-NaivePF and Static-NaivePF must issue the same prefetch "
            f"workload; mismatched fields: {', '.join(mismatched)}"
        )
    if dynamic_pf.total_cycles > static_pf.total_cycles:
        violations.append(
            "Dynamic-NaivePF must not exceed Static-NaivePF "
            f"({dynamic_pf.total_cycles} > {static_pf.total_cycles})"
        )

    implementable = (static, static_pf, dynamic, dynamic_pf, raw)
    best = min(implementable, key=lambda row: (row.total_cycles, row.baseline))
    if safe.total_cycles > best.total_cycles:
        violations.append(
            "MemDomain-Safe must not exceed the best implementable candidate "
            f"({safe.total_cycles} > {best.total_cycles}, best={best.baseline})"
        )

    if violations:
        raise TheoreticalContractViolation("; ".join(violations))
    return values


def _assert_derived_copy(
    derived: ExperimentRow, selected: ExperimentRow, label: str
) -> None:
    excluded = {"baseline", "candidate_source", "fallback_used", "selected_candidate"}
    derived_values = asdict(derived)
    selected_values = asdict(selected)
    mismatched = [
        name for name in derived_values
        if name not in excluded and derived_values[name] != selected_values[name]
    ]
    if mismatched:
        raise ValueError(
            f"{label} row metrics do not match selected candidate: {', '.join(mismatched)}"
        )


def write_matrix(path: Path, rows: Iterable[ExperimentRow]) -> None:
    # P0 contract gate: invalid dynamic results must never become paper output.
    ordered = validate_theoretical_contract(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [item.name for item in fields(ExperimentRow)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow(asdict(row))


def read_matrix(path: Path) -> Tuple[ExperimentRow, ...]:
    integer_fields = {
        item.name for item in fields(ExperimentRow)
        if item.type in (int, "int")
    }
    float_fields = {
        item.name for item in fields(ExperimentRow)
        if item.type in (float, "float")
    }
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            values = dict(raw)
            for name in integer_fields:
                if name not in values or values[name] == "":
                    values[name] = (
                        int(values.get("mapping_overhead_cycles", 0))
                        if name == "mapping_work_cycles" else 0
                    )
                else:
                    values[name] = int(values[name])
            for name in float_fields:
                values[name] = float(values.get(name, 0.0) or 0.0)
            values["fallback_used"] = values["fallback_used"].strip().lower() == "true"
            rows.append(ExperimentRow(**values))
    return validate_matrix(rows)
