"""Audit DATE2 outputs against theory and paper-level performance contracts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.memdomain_experiment import (  # noqa: E402
    Baseline,
    read_matrix,
    validate_theoretical_contract,
    workload_digest,
)

MATRIX_SUITES = ("overall", "window_chunk", "robustness")
BASELINE_NAMES = (
    Baseline.STATIC_NOPF.value,
    Baseline.STATIC_NAIVEPF.value,
    Baseline.DYNAMIC_NOPF.value,
    Baseline.DYNAMIC_NAIVEPF.value,
)


def _matrix_paths(root: Path, suite: str):
    return sorted((root / "outputs/DATE2" / suite).glob("*.csv"))


def _config_path(root: Path, suite: str, stem: str) -> Path:
    return root / "configs/MoE/DATE2" / suite / f"{stem}.json"


def audit_matrix(root: Path, suite: str, path: Path):
    rows = validate_theoretical_contract(read_matrix(path))
    config_path = _config_path(root, suite, path.stem)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = workload_digest(payload)
    actual_hashes = {row.workload_hash for row in rows}
    if actual_hashes != {expected_hash}:
        raise ValueError(f"stale workload hash: {path}")
    by_name = {row.baseline: row for row in rows}
    best_baseline = min(by_name[name].total_cycles for name in BASELINE_NAMES)
    static_pf = by_name[Baseline.STATIC_NAIVEPF.value]
    dynamic_pf = by_name[Baseline.DYNAMIC_NAIVEPF.value]
    safe = by_name[Baseline.MEMDOMAIN_SAFE.value]
    return {
        "suite": suite,
        "variant": path.stem,
        "best_baseline_cycles": best_baseline,
        "safe_cycles": safe.total_cycles,
        "gain": (best_baseline / safe.total_cycles) - 1.0,
        "dynamic_pf_gain": (
            static_pf.total_cycles / dynamic_pf.total_cycles
        ) - 1.0,
        "dynamic_pf_source": dynamic_pf.candidate_source,
        "safe_source": safe.candidate_source,
        "safe_policy": safe.selected_candidate,
        "fallback_used": safe.fallback_used,
    }


def audit_layer_reports(root: Path, suites, require=False):
    failures = []
    experiment_dirs = {
        "overall": "exp4", "window_chunk": "exp5", "robustness": "exp6",
    }
    reports = []
    expected = 0
    for suite in suites:
        configs = sorted(
            (root / "configs/MoE/DATE2" / suite).glob("*.json")
        )
        expected += len(configs)
        reports.extend(
            root / "outputs/DATE2" / experiment_dirs[suite]
            / config.stem / "LAYER_DOMINANCE_REPORT.csv"
            for config in configs
            if (
                root / "outputs/DATE2" / experiment_dirs[suite]
                / config.stem / "LAYER_DOMINANCE_REPORT.csv"
            ).exists()
        )
    if require and len(reports) != expected:
        raise ValueError(
            f"expected {expected} layer dominance reports, found {len(reports)}"
        )
    for path in reports:
        with path.open(newline="", encoding="utf-8") as stream:
            for line, row in enumerate(csv.DictReader(stream), start=2):
                if row["contract_pass"].strip().lower() != "true":
                    failures.append(f"{path}:{line}")
    if failures:
        raise ValueError(
            "per-stage dynamic dominance failed: " + ", ".join(failures)
        )
    return len(reports)


def audit(
    root: Path = ROOT,
    suites=MATRIX_SUITES,
    min_model_gain: float = 0.05,
    min_geomean_gain: float = 0.10,
    require_layer_reports: bool = False,
):
    records = []
    for suite in suites:
        paths = _matrix_paths(root, suite)
        expected = len(list((root / "configs/MoE/DATE2" / suite).glob("*.json")))
        if len(paths) != expected:
            raise ValueError(
                f"{suite}: expected {expected} matrices, found {len(paths)}"
            )
        records.extend(audit_matrix(root, suite, path) for path in paths)

    overall = [item for item in records if item["suite"] == "overall"]
    if "overall" in suites:
        non_strict_dynamic = [
            item for item in overall
            if item["dynamic_pf_gain"] <= 0.0
            or "incumbent_static_mapping" in item["dynamic_pf_source"]
        ]
        if non_strict_dynamic:
            details = ", ".join(
                f"{item['variant']}={item['dynamic_pf_gain']:.2%}"
                for item in non_strict_dynamic
            )
            raise ValueError(
                "overall Dynamic-NaivePF must strictly beat matched "
                f"Static-NaivePF without whole-model fallback: {details}"
            )
        weak = [
            item for item in overall if item["gain"] + 1e-12 < min_model_gain
        ]
        if weak:
            details = ", ".join(
                f"{item['variant']}={item['gain']:.2%}" for item in weak
            )
            raise ValueError(
                f"overall per-model gain below {min_model_gain:.2%}: {details}"
            )
        geomean_gain = (
            math.prod(1.0 + item["gain"] for item in overall)
            ** (1.0 / len(overall)) - 1.0
        )
        if geomean_gain + 1e-12 < min_geomean_gain:
            raise ValueError(
                f"overall geomean gain {geomean_gain:.2%} is below "
                f"{min_geomean_gain:.2%}"
            )
    else:
        geomean_gain = None

    layer_reports = audit_layer_reports(
        root, suites, require=require_layer_reports
    )
    return {
        "schema_version": 1,
        "matrix_count": len(records),
        "layer_report_count": layer_reports,
        "min_model_gain": min((item["gain"] for item in overall), default=None),
        "geomean_gain": geomean_gain,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--suite", choices=(*MATRIX_SUITES, "all"), default="all")
    parser.add_argument("--min-model-gain", type=float, default=0.05)
    parser.add_argument("--min-geomean-gain", type=float, default=0.10)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--allow-missing-layer-reports", action="store_true")
    args = parser.parse_args()
    suites = MATRIX_SUITES if args.suite == "all" else (args.suite,)
    result = audit(
        args.root.resolve(), suites, args.min_model_gain,
        args.min_geomean_gain,
        require_layer_reports=not args.allow_missing_layer_reports,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
