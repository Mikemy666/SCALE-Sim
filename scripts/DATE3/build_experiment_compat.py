"""Build DATE2-compatible paper tables from completed DATE3 variant outputs.

This module performs aggregation only.  It never launches a simulator and it
never reads DATE2 outputs or figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.DATE3.experiment_contract import (
    CHUNKS, CONFIG_ROOT, EXP4_MAPPING_SCHEMES, MODELS, OUTPUT_ROOT,
    LOCAL_MEMORY_COMPONENTS, PUBLIC_BASELINES, ROBUSTNESS_SCHEMES, WINDOWS,
)
from scalesim.memory.buckyball_memdomain import CONTRACT
from scalesim.memory.topology_workload import EXPERT_LAYER, load_moe_topology
from scripts.DATE2.run_date2_characterization import simulate


def _read(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing completed DATE3 artifact: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _one(path: Path):
    rows = _read(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row: {path}")
    return rows[0]


def _write(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError(f"refusing to write empty compatibility table: {path}")
    fields = []
    for row in values:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _baseline_quality(detail_dir: Path) -> dict[str, dict[str, object]]:
    """Measure control quality from unique Tile lifetimes, never proxies."""
    chunks = _read(detail_dir / "CHUNK_REPORT.csv")
    result = {}
    for baseline in sorted({row["baseline"] for row in chunks}):
        selected = [row for row in chunks if row["baseline"] == baseline]
        required = sum(int(row["size_bytes"]) for row in selected)
        prefetched = [row for row in selected if row["planned_kind"] == "prefetch"]
        prefetched_bytes = sum(int(row["size_bytes"]) for row in prefetched)
        timely = sum(int(row["size_bytes"]) for row in prefetched
                     if row["classification"] == "timely")
        late = sum(int(row["size_bytes"]) for row in prefetched
                   if row["classification"] != "timely")
        result[baseline] = {
            "required_bytes": required,
            "prefetched_bytes": prefetched_bytes,
            "useful_timely_bytes": timely,
            "late_bytes": late,
            "unused_bytes": 0,
            "evicted_before_use_bytes": 0,
            "coverage": timely / required if required else "",
            "accuracy": timely / prefetched_bytes if prefetched_bytes else "",
            "coverage_valid": bool(required),
            "accuracy_valid": bool(prefetched_bytes),
            "quality_metric_scope": "unique_tile_lifetime_bytes",
        }
    return result


def _with_exact_quality(row: Mapping[str, object], quality) -> dict[str, object]:
    value = dict(row)
    internal = value["policy_name"]
    if internal == "MemDomain":
        raise ValueError(
            "legacy MemDomain-Safe must remain internal; PIVOT is the sole "
            "public proposed scheme"
        )
    value["policy_name"] = "PIVOT" if internal in ("PIVOT", "PIVOT-CA") else internal
    if all(name in value for name in (
        "required_bytes", "prefetched_bytes", "useful_timely_bytes", "late_bytes"
    )):
        return value
    if value["policy_name"] != "PIVOT":
        aliases = {
            "Static-Opt-NoPF": "Static-NoPF",
            "Static-Opt-FixedPF": "Static-NaivePF",
            "Dynamic-FixedPF": "Dynamic-NaivePF",
            "Static-555-NoPF": "Static-NoPF",
            "Ideal-NoPF": "Dynamic-NoPF",
        }
        value.update(quality[aliases.get(internal, internal)])
    return value


def _public_comparison_rows(path: Path):
    """Normalize old/new comparison files without rerunning simulation."""
    for row in _read(path):
        # Old DATE3 comparison files exposed MemDomain-Safe as MemDomain and
        # appended PIVOT-CA, accidentally creating two names for one design.
        if row["policy_name"] == "MemDomain":
            continue
        value = dict(row)
        if value["policy_name"] == "PIVOT-CA":
            value["policy_name"] = "PIVOT"
        yield value


def build_exp3() -> None:
    rows = []
    for window in WINDOWS:
        for chunk in CHUNKS:
            variant = f"w{window}_c{chunk}"
            matrix = _read(
                OUTPUT_ROOT / "window_chunk" / variant / "baseline_matrix.csv"
            )
            by_name = {row["baseline"]: row for row in matrix}
            chunks = _read(OUTPUT_ROOT / "exp3" / variant / "CHUNK_REPORT.csv")
            for baseline in ("Static-NoPF", "Static-NaivePF"):
                row = by_name[baseline]
                selected = [item for item in chunks if item["baseline"] == baseline]
                required_bytes = sum(int(item["size_bytes"]) for item in selected)
                prefetched = [item for item in selected
                              if item["planned_kind"] == "prefetch"]
                prefetched_bytes = sum(int(item["size_bytes"]) for item in prefetched)
                useful_bytes = sum(int(item["size_bytes"]) for item in prefetched
                                   if item["classification"] == "timely")
                late_bytes = sum(int(item["size_bytes"]) for item in prefetched
                                 if item["classification"] != "timely")
                rows.append({
                    "window": window,
                    "chunk_tiles": chunk,
                    "baseline": baseline,
                    "total_cycles": row["total_cycles"],
                    "prefetch_requests": row["prefetch_requests"],
                    "prefetch_bytes": row["prefetch_bytes"],
                    "bank_conflict_count": row["bank_conflict_count"],
                    "bank_conflict_rate": row["bank_conflict_rate"],
                    "prefetch_interference_stall_cycles":
                        row["prefetch_interference_stall_cycles"],
                    "timely_prefetch_ratio": (
                        useful_bytes / prefetched_bytes if prefetched_bytes else 0.0
                    ),
                    "late_prefetch_ratio": (
                        late_bytes / prefetched_bytes if prefetched_bytes else 0.0
                    ),
                    "unused_prefetch_ratio": row["unused_prefetch_ratio"],
                    "prefetch_occupancy_byte_cycles":
                        row["prefetch_occupancy_byte_cycles"],
                    "compute_transfer_overlap_cycles":
                        row["compute_transfer_overlap_cycles"],
                    "local_memory_stall_cycles": sum(int(row[name]) for name in (
                        "bank_stall_cycles", "weight_load_stall_cycles",
                        "prefetch_miss_stall_cycles",
                        "prefetch_interference_stall_cycles",
                        "mapping_overhead_cycles",
                    )),
                    "communication_exposed_wait_cycles":
                        row["communication_stall_cycles"],
                    "combine_cycles": row["other_stall_cycles"],
                    "required_bytes": required_bytes,
                    "prefetched_unique_bytes": prefetched_bytes,
                    "useful_timely_bytes": useful_bytes,
                    "late_bytes": late_bytes,
                    "coverage": useful_bytes / required_bytes if required_bytes else "",
                    "accuracy": useful_bytes / prefetched_bytes if prefetched_bytes else "",
                    "coverage_valid": bool(required_bytes),
                    "accuracy_valid": bool(prefetched_bytes),
                    "quality_metric_scope": "unique_tile_lifetime_bytes",
                    "workload_hash": row["workload_hash"],
                })
    _write(OUTPUT_ROOT / "exp3/naive_prefetch_interference.csv", rows)


def build_exp4() -> None:
    overall = []
    for model in MODELS:
        for row in _public_comparison_rows(
            OUTPUT_ROOT / "overall" / model / "comparison.csv"
        ):
            if row["policy_name"] in EXP4_MAPPING_SCHEMES:
                overall.append({"model": model, **_with_exact_quality(row, {})})
    _write(OUTPUT_ROOT / "exp4/overall_comparison.csv", overall)
    _write(OUTPUT_ROOT / "exp4/system_breakdown.csv", ({
        "model": row["model"], "policy_name": row["policy_name"],
        "compute_cycles": row["compute_cycles"],
        "local_memory_stall_cycles": row["local_memory_stall_cycles"],
        "communication_exposed_wait_cycles":
            row["communication_exposed_wait_cycles"],
        "combine_cycles": row["combine_cycles"],
        "detailed_ready_cycle": row["detailed_ready_cycle"],
        "peer_ready_cycle": row["peer_ready_cycle"],
        "result_ready_cycle": row["result_ready_cycle"],
        "total_cycles": row["total_cycles"],
    } for row in overall))
    _write(OUTPUT_ROOT / "exp4/mapping_comparison.csv", (
        row for row in overall if row["policy_name"] in EXP4_MAPPING_SCHEMES
    ))


def _build_pivot_ablation(target: Path) -> None:
    ablation = []
    for config in sorted((CONFIG_ROOT / "ablation").glob("*.json")):
        summary = _one(OUTPUT_ROOT / "ablation" / config.stem / "summary.csv")
        ablation.append({"variant": config.stem.split("__", 1)[-1], **summary})
    _write(target, ablation)


def _build_exp5_calibration() -> dict[str, tuple[int, int]]:
    """Select deployable fixed knobs without observing the test trace."""
    rows = []
    for config in sorted((CONFIG_ROOT / "prefetch_calibration").glob("*.json")):
        payload = json.loads(config.read_text(encoding="utf-8"))
        sweep = payload["sweep"]
        directory = OUTPUT_ROOT / "prefetch_calibration" / config.stem
        for row in _public_comparison_rows(directory / "comparison.csv"):
            if row["policy_name"] not in (
                "Static-Opt-FixedPF", "Dynamic-FixedPF"
            ):
                continue
            rows.append({
                "trace": sweep["trace"],
                "window": int(sweep["window"]),
                "chunk_tiles": int(sweep["chunk_tiles"]),
                "policy_name": row["policy_name"],
                "total_cycles": int(row["total_cycles"]),
                "coverage": row["coverage"],
                "accuracy": row["accuracy"],
                "hbm_max_queue_depth": row["hbm_max_queue_depth"],
                "prefetch_requests": row["prefetch_requests"],
                "selection_role": "independent_calibration",
            })
    _write(OUTPUT_ROOT / "exp5/prefetch_calibration.csv", rows)

    policies = ("Static-Opt-FixedPF", "Dynamic-FixedPF")
    selected = {}
    selection_rows = []
    for policy in policies:
        candidates = [row for row in rows if row["policy_name"] == policy]
        traces = sorted({row["trace"] for row in candidates})
        trace_min = {
            trace: min(row["total_cycles"] for row in candidates
                       if row["trace"] == trace)
            for trace in traces
        }
        scored = []
        for window in WINDOWS:
            if window == 0:
                continue
            for chunk in CHUNKS:
                points = [row for row in candidates
                          if row["window"] == window
                          and row["chunk_tiles"] == chunk]
                if len(points) != len(traces):
                    continue
                normalized = [
                    row["total_cycles"] / trace_min[row["trace"]]
                    for row in points
                ]
                scored.append((
                    sum(normalized) / len(normalized),
                    max(normalized), window * chunk, window, chunk,
                ))
        if not scored:
            raise ValueError(f"incomplete Exp5 calibration grid for {policy}")
        mean_norm, worst_norm, _, window, chunk = min(scored)
        selected[policy] = (window, chunk)
        selection_rows.append({
            "policy_name": policy,
            "selected_window": window,
            "selected_chunk_tiles": chunk,
            "calibration_trace_count": len(traces),
            "selection_objective": "mean_normalized_cycles",
            "mean_normalized_cycles": mean_norm,
            "worst_normalized_cycles": worst_norm,
            "test_trace_visible_during_selection": False,
            "deployment_semantics": "one_global_pair_frozen_before_test",
        })
    _write(OUTPUT_ROOT / "exp5/deployable_selection.csv", selection_rows)
    return selected


def build_exp5() -> None:
    _build_exp5_calibration()
    pivot = _one(OUTPUT_ROOT / "ablation/MoDSE__full/summary.csv")
    rows = []
    for window in WINDOWS:
        for chunk in CHUNKS:
            variant = f"w{window}_c{chunk}"
            directory = OUTPUT_ROOT / "joint_prefetch" / variant
            for row in _public_comparison_rows(directory / "comparison.csv"):
                if row["policy_name"] not in PUBLIC_BASELINES:
                    continue
                value = {
                    "window": window, "chunk_tiles": chunk, "variant": variant,
                    **_with_exact_quality(row, {}),
                }
                rows.append(value)
            rows.append({
                "window": window, "chunk_tiles": chunk, "variant": variant,
                "policy_name": "PIVOT",
                "total_cycles": pivot["total_cycles"],
                "compute_cycles": pivot["compute_cycles"],
                "local_memory_stall_cycles": sum(
                    int(float(pivot[name])) for name in LOCAL_MEMORY_COMPONENTS
                ),
                "communication_exposed_wait_cycles": pivot["communication_stall_cycles"],
                "combine_cycles": pivot["combine_cycles"],
                "detailed_ready_cycle": pivot["detailed_ready_cycle"],
                "peer_ready_cycle": pivot["peer_ready_cycle"],
                "result_ready_cycle": pivot["result_ready_cycle"],
                "hbm_queue_wait_cycles": pivot["hbm_queue_wait_cycles"],
                "hbm_service_cycles": pivot["hbm_service_cycles"],
                "hbm_busy_cycles": pivot["hbm_busy_cycles"],
                "hbm_max_queue_depth": pivot["hbm_max_queue_depth"],
                "hbm_utilization": pivot["hbm_utilization"],
                "prefetch_requests": pivot["prefetch_requests"],
                "peak_occupied_bytes": pivot["peak_occupied_bytes"],
                "required_bytes": pivot["required_bytes"],
                "prefetched_bytes": pivot["prefetched_bytes"],
                "useful_timely_bytes": pivot["useful_timely_bytes"],
                "late_bytes": pivot["late_bytes"],
                "coverage": pivot["coverage"], "accuracy": pivot["accuracy"],
                "quality_metric_scope": "unique_useful_timely_bytes",
                "analysis_only": False,
                "candidate_source": "persistent_online_joint_mapping_prefetch",
                "unused_bytes": pivot["unused_bytes"],
                "evicted_before_use_bytes": pivot["evicted_before_use_bytes"],
                "occupancy_byte_cycles": pivot["occupancy_byte_cycles"],
                "fallback_rate": pivot["fallback_rate"],
                "online_incumbent_guard_rate": pivot["online_incumbent_guard_rate"],
                "admission_rejection_count": pivot.get(
                    "admission_rejection_count", 0
                ),
                "selected_chunk_mean": pivot["selected_chunk_mean"],
                "selected_window_mean": pivot["selected_window_mean"],
            })
    _write(OUTPUT_ROOT / "exp5/joint_prefetch.csv", rows)
    _build_pivot_ablation(OUTPUT_ROOT / "exp5/pivot_ca_ablation.csv")


def build_exp6() -> None:
    paper = []
    all_controls = []
    for config in sorted((CONFIG_ROOT / "robustness_factorial").glob("*.json")):
        payload = json.loads(config.read_text(encoding="utf-8"))
        sweep = payload["sweep"]
        comparison = OUTPUT_ROOT / "robustness_factorial" / config.stem / "comparison.csv"
        if not comparison.exists():
            continue
        for row in _public_comparison_rows(comparison):
            if row["policy_name"] not in PUBLIC_BASELINES:
                continue
            value = {
                "variable": sweep["variable"],
                "value": sweep["value"],
                "model": sweep["model"],
                "variant": config.stem,
                **_with_exact_quality(row, {}),
            }
            all_controls.append(value)
            if row["policy_name"] in ROBUSTNESS_SCHEMES:
                paper.append(value)
    _write(OUTPUT_ROOT / "exp6/robustness_all_controls.csv", all_controls)
    _write(OUTPUT_ROOT / "exp6/robustness_comparison.csv", paper)


def _non_moe_block(model: str, block_count: int = 1):
    topology = load_moe_topology(
        ROOT / "topologies/MoE/DATE3/end_to_end" / f"{model}.csv"
    )
    base_layers = []
    embedded_per_block = 0
    for name, m, n, k in topology["layers"]:
        if EXPERT_LAYER.fullmatch(name):
            continue
        analytical_compute = max(1, (
            m * n * k + CONTRACT.tile_size ** 2 - 1
        ) // (CONTRACT.tile_size ** 2))
        report, _, _ = simulate(
            name, (m * k, n * k, m * n), (5, 5, 5),
            analytical_compute, m, n, k,
        )
        memory_stall = sum(report.breakdown[field] for field in (
            "ia_stall_cycles", "weight_stall_cycles",
            "shared_operand_stall_cycles", "accumulator_stall_cycles",
            "oa_stall_cycles",
        ))
        base_layers.append({
            "model": model, "base_layer": name, "category": (
                "Router" if name == "Router_logits" else "Attention"
            ),
            "M": m, "N": n, "K": k,
            "compute_cycles": report.compute_cycles,
            "memory_stall_cycles": memory_stall,
            "total_cycles": report.finish_cycle,
            "memory_model": "Static-5:5:5-SP-plus-15-ACC",
        })
        embedded_per_block += analytical_compute
    layers = []
    for block in range(block_count):
        for row in base_layers:
            layers.append({
                **row, "block": block,
                "layer": f"L{block}__{row['base_layer']}",
            })
    return layers, embedded_per_block * block_count, len(topology["experts"])


def build_exp7() -> None:
    rows = []
    layer_rows = []
    for model in MODELS:
        config_path = ROOT / "configs/MoE/DATE3/end_to_end" / f"{model}.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        approximation = config_payload["end_to_end_approximation"]
        block_count = int(approximation.get("block_count", 1))
        layers, embedded_non_moe_compute, expert_stages = _non_moe_block(
            model, block_count
        )
        layer_rows.extend(layers)
        non_moe_full = sum(int(row["total_cycles"]) for row in layers)
        non_moe_stall = sum(int(row["memory_stall_cycles"]) for row in layers)
        for row in _public_comparison_rows(
            OUTPUT_ROOT / "end_to_end" / model / "comparison.csv"
        ):
            if row["policy_name"] not in PUBLIC_BASELINES:
                continue
            policy_total = int(float(row["total_cycles"]))
            moe_ep_cycles = policy_total - embedded_non_moe_compute
            if moe_ep_cycles <= 0:
                raise ValueError(
                    f"{model}/{row['policy_name']}: invalid MoE+EP residual"
                )
            rows.append({
                "model": model,
                "policy_name": row["policy_name"],
                "block_count": block_count,
                "non_moe_layer_count": len(layers),
                "moe_expert_stage_count": expert_stages,
                "embedded_non_moe_compute_cycles": embedded_non_moe_compute,
                "non_moe_full_cycles": non_moe_full,
                "non_moe_memory_stall_cycles": non_moe_stall,
                "moe_ep_cycles": moe_ep_cycles,
                "approx_block_total_cycles": non_moe_full + moe_ep_cycles,
                "composition_check_cycles": (
                    policy_total + non_moe_full - embedded_non_moe_compute
                ),
                "approximation_scope": approximation["scope"],
                "ignored_operations": (
                    "embedding|normalization|softmax|residual|sampling"
                ),
            })
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["policy_name"]] = row
    for row in rows:
        static = by_model[row["model"]]["Static-555-NoPF"]
        row["end_to_end_speedup_vs_static"] = (
            static["approx_block_total_cycles"]
            / row["approx_block_total_cycles"]
        )
        row["moe_ep_speedup_vs_static"] = (
            static["moe_ep_cycles"] / row["moe_ep_cycles"]
        )
        row["end_to_end_cycle_reduction_pct"] = 100.0 * (
            1.0 - row["approx_block_total_cycles"]
            / static["approx_block_total_cycles"]
        )
        row["moe_ep_cycle_reduction_pct"] = 100.0 * (
            1.0 - row["moe_ep_cycles"] / static["moe_ep_cycles"]
        )
        row["static_moe_ep_fraction"] = (
            static["moe_ep_cycles"] / static["approx_block_total_cycles"]
        )
        fraction = row["static_moe_ep_fraction"]
        row["amdahl_speedup"] = 1.0 / (
            (1.0 - fraction)
            + fraction / row["moe_ep_speedup_vs_static"]
        )
    _write(OUTPUT_ROOT / "exp7/end_to_end_summary.csv", rows)
    _write(OUTPUT_ROOT / "exp7/non_moe_layer_breakdown.csv", layer_rows)


BUILDERS = {
    "exp3": build_exp3,
    "exp4": build_exp4,
    "exp5": build_exp5,
    "exp6": build_exp6,
    "exp7": build_exp7,
}


def build_experiment(exp: str) -> None:
    if exp in ("exp1", "exp2"):
        return
    BUILDERS[exp]()
    print(f"built DATE3 {exp} compatibility tables")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=tuple(BUILDERS), required=True)
    args = parser.parse_args()
    build_experiment(args.exp)


if __name__ == "__main__":
    main()
