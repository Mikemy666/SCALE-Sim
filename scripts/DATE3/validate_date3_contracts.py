"""Validate DATE3 summary/detail/epoch contracts without touching DATE2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/MoE/DATE3"
OUTPUT_ROOT = ROOT / "outputs/DATE3"
CYCLE_FIELDS = (
    "compute_cycles", "bank_stall_cycles", "weight_load_stall_cycles",
    "prefetch_miss_stall_cycles", "prefetch_interference_stall_cycles",
    "mapping_overhead_cycles", "communication_stall_cycles", "other_stall_cycles",
)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def load_one(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise AssertionError(f"expected one summary row: {path}")
    return rows[0]


def valid_ratio(row, name, valid_name):
    valid = row[valid_name].lower() == "true"
    if valid:
        value = float(row[name])
        assert 0.0 <= value <= 1.0
    else:
        assert row[name] in ("", "nan", "NaN")


def validate_variant(config: Path, output: Path):
    summary = load_one(output / "summary.csv")
    payload = json.loads(config.read_text(encoding="utf-8"))
    assert summary["config_hash"] == canonical_hash(payload)
    assert len(summary["implementation_hash"]) == 64
    values = [int(float(summary[name])) for name in CYCLE_FIELDS]
    assert min(values) >= 0
    assert int(float(summary["total_cycles"])) == sum(values)
    assert int(summary["result_ready_cycle"]) == max(
        int(summary["detailed_ready_cycle"]), int(summary["peer_ready_cycle"])
    )
    assert int(summary["total_cycles"]) == (
        int(summary["result_ready_cycle"]) + int(summary["combine_cycles"])
    )
    ep = payload["ep"]
    system = payload["system"]
    remote = int(summary["remote_route_replicas"])
    assert int(summary["dispatch_bytes"]) == (
        remote * int(system["token_payload_bytes"])
    )
    assert int(summary["return_bytes"]) == (
        remote * int(system["result_payload_bytes"])
    )
    valid_ratio(summary, "coverage", "coverage_valid")
    valid_ratio(summary, "accuracy", "accuracy_valid")
    useful = int(summary["useful_timely_bytes"])
    assert useful <= int(summary["required_bytes"])
    assert useful <= int(summary["prefetched_bytes"])
    assert min(int(summary[name]) for name in (
        "late_bytes", "unused_bytes", "evicted_before_use_bytes",
        "occupancy_byte_cycles")) >= 0
    assert int(summary["shadow_real_request_count"]) == 0
    assert float(summary["selected_chunk_min"]) <= float(summary["selected_chunk_max"])
    assert float(summary["selected_window_min"]) <= float(summary["selected_window_max"])
    detail_path = output / "decision_detail.csv"
    epoch_path = output / "quality_epochs.csv"
    route_path = output / "ep_routes.csv"
    local_path = output / "ep_local_workload.csv"
    peer_path = output / "ep_peer_workloads.csv"
    timeline_path = output / "ep_timeline.csv"
    combine_path = output / "ep_return_combine.csv"
    guard_path = output / "online_incumbent_guard.csv"
    assert all(path.exists() for path in (
        detail_path, epoch_path, route_path, local_path, peer_path,
        timeline_path, combine_path,
        guard_path,
    ))
    with detail_path.open(newline="", encoding="utf-8") as stream:
        detail = list(csv.DictReader(stream))
    assert detail
    for row in detail:
        assert "oracle" not in json.dumps(row).lower()
        if row["selected"].lower() == "true" and row["fallback_used"].lower() != "true":
            assert float(row["predicted_coverage"]) >= float(row["coverage_threshold"])
            assert float(row["predicted_accuracy"]) >= float(row["accuracy_threshold"])
    with route_path.open(newline="", encoding="utf-8") as stream:
        routes = list(csv.DictReader(stream))
    assert len(routes) == sum(int(value) for value in ep["token_counts"])
    by_token = {}
    observed = [0] * int(ep["num_experts"])
    for route in routes:
        token = int(route["token_id"])
        expert = int(route["global_expert_id"])
        owner = int(route["owner_npu"])
        assert owner == int(ep["expert_owner_map"][expert])
        observed[expert] += 1
        by_token.setdefault(token, []).append(route)
    assert observed == [int(value) for value in ep["token_counts"]]
    assert all(len(items) == int(ep["top_k"]) for items in by_token.values())
    with combine_path.open(newline="", encoding="utf-8") as stream:
        combine = list(csv.DictReader(stream))
    assert len(combine) == len(by_token)
    assert all(int(row["expected_results"]) == int(ep["top_k"])
               for row in combine)
    with peer_path.open(newline="", encoding="utf-8") as stream:
        peer = list(csv.DictReader(stream))
    for row in peer:
        expert = int(row["expert_id"])
        assert int(row["npu_id"]) == int(ep["expert_owner_map"][expert])
        assert int(row["npu_id"]) != int(ep["detailed_npu_id"])
    with guard_path.open(newline="", encoding="utf-8") as stream:
        guards = list(csv.DictReader(stream))
    assert guards
    for row in guards:
        proposal = int(row["proposal_prefix_cost_cycles"])
        fixed = int(row["fixed_prefix_cost_cycles"])
        noprefetch = int(row["noprefetch_prefix_cost_cycles"])
        incumbent = int(row["incumbent_prefix_cost_cycles"])
        applied = int(row["applied_prefix_cost_cycles"])
        assert incumbent == min(proposal, fixed, noprefetch)
        assert applied == incumbent
        assert row["applied_action"] == row["incumbent_action"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="unit_cases")
    parser.add_argument("--require-runtime-variation", action="store_true")
    args = parser.parse_args()
    checked = []
    for config in sorted((CONFIG_ROOT / args.suite).glob("*.json")):
        output = OUTPUT_ROOT / args.suite / config.stem
        if not (output / "summary.csv").exists():
            continue
        checked.append(validate_variant(config, output))
    if not checked:
        raise SystemExit("no DATE3 outputs found to validate")
    if args.require_runtime_variation:
        assert any(
            float(row["selected_chunk_min"]) < float(row["selected_chunk_max"])
            or float(row["selected_window_min"]) < float(row["selected_window_max"])
            for row in checked
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "outputs/DATE2", "fig/DATE2"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise AssertionError(f"DATE2 outputs/figures changed:\n{dirty}")
    print(f"validated {len(checked)} DATE3 variant(s); DATE2 outputs unchanged")


if __name__ == "__main__":
    main()
