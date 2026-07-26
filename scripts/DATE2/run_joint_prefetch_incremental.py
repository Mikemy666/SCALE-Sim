"""Run exp5 joint paths while reusing unchanged validated exp3 baselines."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.memdomain_experiment import (  # noqa: E402
    Baseline,
    derive_selected_row,
    read_matrix,
    validate_matrix,
    workload_digest,
    write_matrix,
)
from scalesim.memory.memdomain_runner import (  # noqa: E402
    load_runner_config,
    run_raw_baseline_with_details,
)

CONVENTIONAL = (
    Baseline.STATIC_NOPF,
    Baseline.STATIC_NAIVEPF,
    Baseline.DYNAMIC_NOPF,
    Baseline.DYNAMIC_NAIVEPF,
)


def comparable_payload(payload):
    value = json.loads(json.dumps(payload))
    for key in ("experiment_id", "date2_suite", "date2_variant"):
        value.pop(key, None)
    policy = value["policy"]
    for key in (
        "adaptive_prefetch",
        "max_prefetch_window",
        "max_prefetch_capacity_fraction",
    ):
        policy.pop(key, None)
    return value


def run_variant(stem: str):
    old_config = ROOT / "configs/MoE/DATE2/window_chunk" / f"{stem}.json"
    new_config = ROOT / "configs/MoE/DATE2/joint_prefetch" / f"{stem}.json"
    old_output = ROOT / "outputs/DATE2/window_chunk" / f"{stem}.csv"
    output = ROOT / "outputs/DATE2/joint_prefetch" / f"{stem}.csv"
    old_payload = json.loads(old_config.read_text(encoding="utf-8"))
    new_payload = json.loads(new_config.read_text(encoding="utf-8"))
    if comparable_payload(old_payload) != comparable_payload(new_payload):
        raise ValueError(f"{stem}: exp3/exp5 workloads differ beyond joint policy")
    if not new_payload["policy"].get("adaptive_prefetch", False):
        raise ValueError(f"{stem}: adaptive_prefetch is not enabled")

    old_rows = {row.baseline: row for row in validate_matrix(read_matrix(old_output))}
    config = load_runner_config(new_config)
    digest = workload_digest(new_payload)
    conventional = [
        replace(
            old_rows[item.value],
            experiment_id=config.experiment_id,
            workload_hash=digest,
        )
        for item in CONVENTIONAL
    ]
    raw = run_raw_baseline_with_details(config, Baseline.MEMDOMAIN_RAW).row
    safe = run_raw_baseline_with_details(config, Baseline.MEMDOMAIN_SAFE).row
    incumbent = min(conventional, key=lambda row: (row.total_cycles, row.baseline))
    if safe.total_cycles > incumbent.total_cycles:
        safe = replace(
            incumbent,
            baseline=Baseline.MEMDOMAIN_SAFE.value,
            candidate_source=(
                "measured:online_model_incumbent|" + incumbent.baseline
            ),
            fallback_used=True,
            selected_candidate="Online-Guarded-Full",
        )
    raw_rows = [*conventional, raw]
    oracle = derive_selected_row(
        Baseline.ORACLE,
        [*raw_rows, safe],
        [*CONVENTIONAL, Baseline.MEMDOMAIN_RAW, Baseline.MEMDOMAIN_SAFE],
    )
    rows = validate_matrix([*raw_rows, safe, oracle])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    write_matrix(temporary, rows)
    temporary.replace(output)
    print(f"completed: joint_prefetch/{stem}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant")
    args = parser.parse_args()
    configs = sorted(
        (ROOT / "configs/MoE/DATE2/joint_prefetch").glob("w*_c*.json")
    )
    stems = [path.stem for path in configs]
    if args.variant:
        if args.variant not in stems:
            parser.error(f"unknown joint-prefetch variant: {args.variant}")
        stems = [args.variant]
    for stem in stems:
        run_variant(stem)


if __name__ == "__main__":
    main()
