"""Validate the DATE3 Exp1--Exp7 paper contract without running simulation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.DATE3.experiment_contract import (
    CONFIG_ROOT, EXP4_MAPPING_SCHEMES, FIG_ROOT, OUTPUT_ROOT,
    PUBLIC_BASELINES, ROBUSTNESS_SCHEMES,
)

EXPECTED_CONFIGS = {
    "overall": 4, "end_to_end": 4, "ablation": 8, "window_chunk": 32,
    "prefetch_calibration": 56, "joint_prefetch": 32, "quality_sensitivity": 19,
    "robustness_factorial": 96, "unit_cases": 1,
}
EXPECTED_ROWS = {
    "exp3/naive_prefetch_interference.csv": 64,
    "exp4/overall_comparison.csv": 4 * len(EXP4_MAPPING_SCHEMES),
    "exp4/mapping_comparison.csv": 4 * len(EXP4_MAPPING_SCHEMES),
    "exp5/joint_prefetch.csv": 32 * len(PUBLIC_BASELINES),
    "exp5/prefetch_calibration.csv": 56 * 2,
    "exp5/deployable_selection.csv": 2,
    "exp5/pivot_ca_ablation.csv": 8,
    "exp6/robustness_comparison.csv": 96 * len(ROBUSTNESS_SCHEMES),
    "exp7/end_to_end_summary.csv": 4 * len(PUBLIC_BASELINES),
    "exp7/non_moe_layer_breakdown.csv": 4 * 4 * 7,
}
REQUIRED_INPUTS = (
    "exp1/exp1.json", "exp2/exp2.json", "exp3/exp3.json",
    "architecture.json", "manifest.json",
)
REQUIRED_CHARACTERIZATION = (
    "exp1/layer_characterization.csv", "exp1/accumulator_sensitivity.csv",
    "exp1/temporal_bank_demand.csv", "exp2/static_bank_sweep.csv",
    "exp2/per_stage_best.csv",
)
REQUIRED_FIGURES = (
    "exp1_cycle_breakdown.pdf", "exp1_flow_and_ideal_banks.pdf",
    "exp2_per_stage_best_ratio.pdf", "exp2_static_ratio_heatmap.pdf",
    "exp3_performance_heatmap.pdf", "exp3_timeliness_occupancy_conflict.pdf",
    "exp4_mapping_four_way.pdf", "exp4_mapping_benefit_decomposition.pdf",
    "exp4_mapping_stall_and_critical_path.pdf",
    "exp5_prefetch_tradeoff.pdf", "exp5_public_sensitivity.pdf",
    "exp5_calibration_selection.pdf",
    "exp5_online_adaptation.pdf", "exp5_online_guard.pdf",
    "exp5_hbm_mechanism.pdf",
    "exp5_pivot_ca_ablation.pdf",
    "exp6_failure_diagnosis.pdf",
    "exp7_end_to_end_speedup.pdf", "exp7_block_decomposition.pdf",
    "exp7_non_moe_layer_breakdown.pdf",
)


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_configs() -> None:
    for relative in REQUIRED_INPUTS:
        assert (CONFIG_ROOT / relative).exists(), f"missing DATE3 input: {relative}"
    for suite, expected in EXPECTED_CONFIGS.items():
        actual = len(list((CONFIG_ROOT / suite).glob("*.json")))
        assert actual == expected, f"{suite}: expected {expected} configs, got {actual}"
    manifest = json.loads((CONFIG_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suites"] == EXPECTED_CONFIGS
    assert manifest["paper_experiments"]["exp7"] == "end_to_end"
    assert manifest["date2_modified"] is False
    for config in (CONFIG_ROOT / "overall").glob("*.json"):
        payload = json.loads(config.read_text(encoding="utf-8"))
        assert payload["ep"]["num_npus"] == 2
        assert payload["hardware"]["bank_count"] == 30
        assert payload["hardware"]["bank_width_bits"] == 128
        assert payload["hardware"]["ports_per_bank"] == 1
    for config in (CONFIG_ROOT / "end_to_end").glob("*.json"):
        payload = json.loads(config.read_text(encoding="utf-8"))
        contract = payload["end_to_end_approximation"]
        assert contract["scope"] == "four_complete_moe_transformer_blocks"
        assert contract["block_count"] == 4
        assert len(contract["non_moe_layers"]) == 7
        assert payload["multi_layer_prefetch"]["layer_count"] == 4
        assert payload["multi_layer_prefetch"]["router_boundary_prefetch"] == "forbidden"
        assert payload["ep"]["num_npus"] == 2
    for exp in ("exp1", "exp2"):
        payload = json.loads((CONFIG_ROOT / exp / f"{exp}.json").read_text(encoding="utf-8"))
        assert payload["ep_degree"] == 1
        assert payload["hardware"]["bank_count"] == 30
        assert payload["precision"]["accumulator"] == "INT32"
    plot_source = (ROOT / "scripts/DATE3/plot_experiments.py").read_text(encoding="utf-8")
    compat_source = (ROOT / "scripts/DATE3/build_experiment_compat.py").read_text(encoding="utf-8")
    assert "outputs/DATE2" not in plot_source + compat_source
    assert "fig/DATE2" not in plot_source + compat_source


def validate_outputs() -> None:
    for relative in REQUIRED_CHARACTERIZATION:
        assert (OUTPUT_ROOT / relative).exists(), f"missing result: {relative}"
    for relative, expected in EXPECTED_ROWS.items():
        path = OUTPUT_ROOT / relative
        assert path.exists(), f"missing result: {relative}"
        rows = _rows(path)
        assert len(rows) == expected, f"{relative}: expected {expected} rows, got {len(rows)}"
        if rows and "policy_name" in rows[0]:
            names = {row["policy_name"] for row in rows}
            assert not names.intersection({
                "MemDomain", "MemDomain-Raw", "MemDomain-Safe",
                "PIVOT-CA", "Oracle",
            })
            if "PIVOT" in names:
                keys = [
                    tuple(row.get(field, "") for field in (
                        "model", "window", "chunk_tiles", "variable", "value",
                    ))
                    for row in rows if row["policy_name"] == "PIVOT"
                ]
                assert len(keys) == len(set(keys)), (
                    f"{relative}: duplicate public PIVOT rows"
                )
    exp2 = _rows(OUTPUT_ROOT / "exp2/static_bank_sweep.csv")
    stages = {row["layer"] for row in exp2}
    assert len(exp2) == len(stages) * 91
    exp7 = _rows(OUTPUT_ROOT / "exp7/end_to_end_summary.csv")
    for row in exp7:
        assert int(row["approx_block_total_cycles"]) == int(
            row["composition_check_cycles"]
        )
        assert abs(float(row["end_to_end_speedup_vs_static"])
                   - float(row["amdahl_speedup"])) < 1e-9


def validate_figures() -> None:
    for name in REQUIRED_FIGURES:
        assert (FIG_ROOT / name).exists(), f"missing figure: {name}"
    exp6 = _rows(OUTPUT_ROOT / "exp6/robustness_comparison.csv")
    variables = {row["variable"] for row in exp6}
    for variable in variables:
        assert (FIG_ROOT / f"exp6_{variable}.pdf").exists(), (
            f"missing Exp6 single-variable figure: {variable}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-outputs", action="store_true")
    parser.add_argument("--require-figures", action="store_true")
    args = parser.parse_args()
    validate_configs()
    if args.require_outputs or args.require_figures:
        validate_outputs()
    if args.require_figures:
        validate_figures()
    print("DATE3 Exp1--Exp7 contract validated")


if __name__ == "__main__":
    main()
