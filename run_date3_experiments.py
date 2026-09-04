"""Run PIVOT (the DATE3 MemDomain design) with resumable hash validation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from scalesim.memory.memdomain_experiment import write_matrix
from scalesim.memory.memdomain_runner import load_runner_config
from scalesim.memory.date3_ep_system import run_date3_ep_baseline_matrix
from scalesim.memory.date3_ep_system import run_date3_ep_paper_controls
from scalesim.memory.date3_ep_system import build_ep_system_timeline
from scalesim.memory.date3_ep_model import localize_detailed_npu
from scalesim.memory.pivot_ca_runner import implementation_digest, run_pivot_ca_file
from scripts.DATE3.experiment_contract import (
    ANALYSIS_ONLY, EXP_TO_SUITE, LOCAL_MEMORY_COMPONENTS, public_name,
)

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs/MoE/DATE3"
OUTPUT_ROOT = ROOT / "outputs/DATE3"
SUITES = ("overall", "end_to_end", "ablation", "window_chunk", "prefetch_calibration",
          "joint_prefetch", "quality_sensitivity",
          "robustness_factorial", "unit_cases")
BASELINE_SUITES = {"overall", "end_to_end", "prefetch_calibration",
                   "joint_prefetch", "robustness_factorial"}
DETAIL_SUITES = {"overall", "end_to_end", "joint_prefetch", "robustness_factorial"}
EXP6_VARIABLES = (
    "expert_count", "token_count", "top_k", "expert_parallel",
    "routing_severity", "routing_seed",
)


def config_hash(config: Path) -> str:
    return __import__("hashlib").sha256(json.dumps(
        json.loads(config.read_text(encoding="utf-8")),
        sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def current(config: Path, output: Path) -> bool:
    summary = output / "summary.csv"
    metadata = output / "metadata.json"
    required = [
        summary, metadata, output / "decision_detail.csv",
        output / "quality_epochs.csv", output / "ep_routes.csv",
        output / "ep_local_workload.csv", output / "ep_peer_workloads.csv",
        output / "ep_timeline.csv", output / "ep_return_combine.csv",
        output / "online_incumbent_guard.csv",
    ]
    if output.parent.name in BASELINE_SUITES:
        required.append(output / "comparison.csv")
    if not all(path.exists() for path in required):
        return False
    try:
        expected = config_hash(config)
        value = json.loads(metadata.read_text(encoding="utf-8"))
        return (value["config_hash"] == expected
                and value["implementation_hash"] == implementation_digest())
    except (OSError, ValueError, KeyError):
        return False


def current_matrix(config: Path, output: Path) -> bool:
    matrix = output / "baseline_matrix.csv"
    metadata = output / "baseline_metadata.json"
    if not matrix.exists() or not metadata.exists():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        return (value["config_hash"] == config_hash(config)
                and value["execution_model"] == "date3_ep_baseline_matrix"
                and value["implementation_hash"] == implementation_digest())
    except (OSError, ValueError, KeyError):
        return False


def details_current(config: Path, output: Path) -> bool:
    required = (
        "CHUNK_REPORT.csv", "EXPERT_REPORT.csv", "FFN_STAGE_REPORT.csv",
        "BANK_REPORT.csv", "REQUEST_REPORT.csv", "EXPERT_INPUT_REPORT.csv",
        "MEASURED_SELECTIONS.csv", "EP_ROUTE_REPORT.csv",
        "NPU_WORKLOAD_REPORT.csv", "EP_COMMUNICATION_REPORT.csv",
    )
    metadata = output / "DETAILS_META.json"
    if not metadata.exists() or not all((output / name).exists() for name in required):
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        return (value["workload_hash"] == (
            __import__("scalesim.memory.memdomain_experiment", fromlist=["workload_digest"])
            .workload_digest(json.loads(config.read_text(encoding="utf-8")))
        ) and value["implementation_hash"] == implementation_digest())
    except (OSError, ValueError, KeyError):
        return False


def _local_memory_stall(row) -> int:
    return sum(int(getattr(row, name)) for name in LOCAL_MEMORY_COMPONENTS)


def _control_quality(execution, chunk_sizes):
    chunks = execution.chunks
    prefetched = [item for item in chunks if item.planned_kind == "prefetch"]
    if not prefetched:
        required = sum(chunk_sizes.values())
        return {
            "required_bytes": required,
            "prefetched_bytes": 0,
            "useful_timely_bytes": 0,
            "late_bytes": 0,
            "coverage": 0.0,
            "accuracy": 0.0,
            "quality_metric_scope": "unique_tile_lifetime_bytes",
        }
    required = sum(chunk_sizes[item.chunk_id] for item in chunks)
    prefetched_bytes = sum(chunk_sizes[item.chunk_id] for item in prefetched)
    timely = sum(
        chunk_sizes[item.chunk_id] for item in prefetched
        if item.classification == "timely"
    )
    late = prefetched_bytes - timely
    return {
        "required_bytes": required,
        "prefetched_bytes": prefetched_bytes,
        "useful_timely_bytes": timely,
        "late_bytes": late,
        "coverage": timely / required if required else 0.0,
        "accuracy": timely / prefetched_bytes if prefetched_bytes else 0.0,
        "quality_metric_scope": "unique_tile_lifetime_bytes",
    }


def write_comparison(config_path: Path, output: Path, controls, execution=None) -> None:
    """Write public, EP-fair comparison rows with unambiguous stall scopes."""
    expected = {
        "Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF",
        "Static-Opt-FixedPF", "Dynamic-FixedPF", "Ideal-NoPF",
    }
    if set(controls) != expected:
        raise ValueError("paper controls are incomplete or duplicated")
    cycles = {name: value.row.total_cycles for name, value in controls.items()}
    if not (
        cycles["Static-Opt-NoPF"] <= cycles["Static-555-NoPF"]
        and cycles["Dynamic-NoPF"] <= cycles["Static-Opt-NoPF"]
        and cycles["Dynamic-FixedPF"] <= cycles["Static-Opt-FixedPF"]
        and cycles["Ideal-NoPF"] <= cycles["Dynamic-NoPF"]
    ):
        raise ValueError(f"paper control dominance contract failed: {cycles}")
    original = load_runner_config(config_path)
    detailed = localize_detailed_npu(original)
    chunk_sizes = {item.chunk_id: item.size_bytes for item in detailed.config.chunks}
    rows = []
    for name, control in controls.items():
        row = control.row
        detailed_ready = row.total_cycles - row.communication_stall_cycles - row.other_stall_cycles
        timeline = build_ep_system_timeline(original, detailed, detailed_ready)
        quality = _control_quality(control, chunk_sizes)
        rows.append({
            "policy_name": name,
            "total_cycles": row.total_cycles,
            "compute_cycles": row.compute_cycles,
            "local_memory_stall_cycles": _local_memory_stall(row),
            "communication_exposed_wait_cycles": row.communication_stall_cycles,
            "combine_cycles": row.other_stall_cycles,
            "detailed_ready_cycle": timeline.detailed_ready_cycle,
            "peer_ready_cycle": timeline.peer_ready_cycle,
            "result_ready_cycle": timeline.result_ready_cycle,
            "hbm_queue_wait_cycles": row.hbm_queue_wait_cycles,
            "hbm_service_cycles": row.hbm_service_cycles,
            "hbm_busy_cycles": row.hbm_busy_cycles,
            "hbm_max_queue_depth": row.hbm_max_queue_depth,
            "hbm_utilization": row.hbm_utilization,
            "prefetch_requests": row.prefetch_requests,
            "peak_occupied_bytes": row.peak_occupied_bytes,
            **quality,
            "analysis_only": name == "Ideal-NoPF",
            "candidate_source": row.candidate_source,
        })
    if execution is None:
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return
    summary = execution.summary
    if config_path.parent.name == "overall":
        strict = {
            "Dynamic-NoPF<Static-Opt-NoPF": (
                cycles["Dynamic-NoPF"] < cycles["Static-Opt-NoPF"]
            ),
            "Dynamic-FixedPF<Static-Opt-FixedPF": (
                cycles["Dynamic-FixedPF"] < cycles["Static-Opt-FixedPF"]
            ),
            "PIVOT<Dynamic-NoPF": (
                summary["total_cycles"] < cycles["Dynamic-NoPF"]
            ),
            "PIVOT<Dynamic-FixedPF": (
                summary["total_cycles"] < cycles["Dynamic-FixedPF"]
            ),
        }
        if not all(strict.values()):
            raise ValueError(
                f"overall strict paper ordering failed: {strict}; "
                f"controls={cycles}; PIVOT={summary['total_cycles']}"
            )
    rows.append({
        "policy_name": "PIVOT",
        "total_cycles": summary["total_cycles"],
        "compute_cycles": summary["compute_cycles"],
        "local_memory_stall_cycles": sum(
            int(summary[name]) for name in LOCAL_MEMORY_COMPONENTS
        ),
        "communication_exposed_wait_cycles": summary["communication_stall_cycles"],
        "combine_cycles": summary["combine_cycles"],
        "detailed_ready_cycle": summary["detailed_ready_cycle"],
        "peer_ready_cycle": summary["peer_ready_cycle"],
        "result_ready_cycle": summary["result_ready_cycle"],
        "hbm_queue_wait_cycles": summary["hbm_queue_wait_cycles"],
        "hbm_service_cycles": summary["hbm_service_cycles"],
        "hbm_busy_cycles": summary["hbm_busy_cycles"],
        "hbm_max_queue_depth": summary["hbm_max_queue_depth"],
        "hbm_utilization": summary["hbm_utilization"],
        "prefetch_requests": summary["prefetch_requests"],
        "peak_occupied_bytes": summary["peak_occupied_bytes"],
        "coverage": summary["coverage"],
        "accuracy": summary["accuracy"],
        "required_bytes": summary["required_bytes"],
        "prefetched_bytes": summary["prefetched_bytes"],
        "useful_timely_bytes": summary["useful_timely_bytes"],
        "late_bytes": summary["late_bytes"],
        "quality_metric_scope": "unique_useful_timely_bytes",
        "analysis_only": False,
        "candidate_source": "online_joint_mapping_prefetch",
    })
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=SUITES, default="unit_cases")
    parser.add_argument("--exp", choices=("exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"))
    parser.add_argument("--variant")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-details", action="store_true")
    parser.add_argument("--exp6-variable", choices=EXP6_VARIABLES)
    parser.add_argument("--exp6-value")
    parser.add_argument("--exp6-model", choices=("HMoE", "Mixtral", "MoDSE", "Switchtrans"))
    args = parser.parse_args()
    if any((args.exp6_variable, args.exp6_value, args.exp6_model)) and args.exp != "exp6":
        parser.error("--exp6-* filters require --exp exp6")
    if args.exp6_value and not args.exp6_variable:
        parser.error("--exp6-value requires --exp6-variable")
    if args.exp in ("exp1", "exp2"):
        command = [sys.executable, str(ROOT / "scripts/DATE3/run_date3_characterization.py"),
                   "--exp", args.exp]
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if args.exp == "exp7":
        selection = OUTPUT_ROOT / "exp5/deployable_selection.csv"
        if not selection.exists():
            parser.error(
                "Exp7 requires outputs/DATE3/exp5/deployable_selection.csv"
            )
        if not args.dry_run:
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/DATE3/prepare_exp7_multilayer.py"),
            ], cwd=ROOT, check=True)
    suites = (
        ("window_chunk",) if args.exp == "exp3" else
        ("overall",) if args.exp == "exp4" else
        # Exp5 calibration is a one-time, frozen deployment artifact.  The
        # multi-layer test must not silently recalibrate after seeing a new
        # evaluation trace.
        ("ablation", "joint_prefetch") if args.exp == "exp5" else
        ("robustness_factorial",) if args.exp == "exp6" else
        ("end_to_end",) if args.exp == "exp7" else
        (args.suite,)
    )
    if args.exp == "exp5" and not (
        OUTPUT_ROOT / "exp5/deployable_selection.csv"
    ).exists():
        parser.error(
            "Exp5 requires a frozen deployable_selection.csv; run "
            "--suite prefetch_calibration first and aggregate calibration"
        )
    matched_any = False
    for suite in suites:
        configs = sorted((CONFIG_ROOT / suite).glob("*.json"))
        if args.variant:
            configs = [item for item in configs if item.stem == args.variant]
        if suite == "robustness_factorial" and any(
            (args.exp6_variable, args.exp6_value, args.exp6_model)
        ):
            selected = []
            for item in configs:
                sweep = json.loads(item.read_text(encoding="utf-8")).get("sweep", {})
                if args.exp6_variable and sweep.get("variable") != args.exp6_variable:
                    continue
                if args.exp6_value and str(sweep.get("value")) != args.exp6_value:
                    continue
                if args.exp6_model and sweep.get("model") != args.exp6_model:
                    continue
                selected.append(item)
            configs = selected
        if not configs:
            if args.variant and len(suites) > 1:
                print(f"skip: variant {args.variant} is not in suite={suite}")
                continue
            parser.error(f"no DATE3 configuration matched suite={suite}")
        matched_any = True
        summaries = []
        for config in configs:
            output = OUTPUT_ROOT / suite / config.stem
            controls_only = suite == "joint_prefetch" and args.skip_details
            if controls_only:
                metadata = output / "metadata.json"
                valid = False
                if metadata.exists() and (output / "comparison.csv").exists():
                    try:
                        saved = json.loads(metadata.read_text(encoding="utf-8"))
                        valid = (
                            saved.get("config_hash") == config_hash(config)
                            and saved.get("implementation_hash") == implementation_digest()
                            and saved.get("execution_model")
                                == "date3_exp5_fixed_controls_only"
                        )
                    except (OSError, ValueError):
                        valid = False
                if args.dry_run:
                    print(f"{'resume valid' if valid and not args.force else 'run controls'}: "
                          f"{config} -> {output / 'comparison.csv'}")
                    continue
                if valid and not args.force:
                    print(f"resume: fixed controls {suite}/{config.stem}")
                    continue
                original = load_runner_config(config)
                controls = run_date3_ep_paper_controls(original)
                output.mkdir(parents=True, exist_ok=True)
                write_comparison(config, output, controls)
                metadata.write_text(json.dumps({
                    "config_hash": config_hash(config),
                    "implementation_hash": implementation_digest(),
                    "execution_model": "date3_exp5_fixed_controls_only",
                    "pivot_source": "outputs/DATE3/ablation/MoDSE__full",
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"completed: fixed controls {suite}/{config.stem}")
                continue
            if suite == "window_chunk":
                matrix_path = output / "baseline_matrix.csv"
                if args.dry_run:
                    print(f"{'resume valid' if current_matrix(config, output) and not args.force else 'run'}: {config} -> {matrix_path}")
                    detail = OUTPUT_ROOT / "exp3" / config.stem
                    state = "skipped" if args.skip_details else (
                        "resume valid" if details_current(config, detail) and not args.force
                        else "run missing/stale"
                    )
                    print(f"detail [{state}]: {config} -> {detail}")
                    continue
                matrix_valid = current_matrix(config, output) and not args.force
                if matrix_valid:
                    print(f"resume: {suite}/{config.stem}")
                else:
                    rows = run_date3_ep_baseline_matrix(load_runner_config(config))
                    output.mkdir(parents=True, exist_ok=True)
                    write_matrix(matrix_path, rows)
                    (output / "baseline_metadata.json").write_text(json.dumps({
                        "config_hash": config_hash(config),
                        "execution_model": "date3_ep_baseline_matrix",
                        "implementation_hash": implementation_digest(),
                    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    summaries.extend({"variant": config.stem, **asdict(row)} for row in rows)
                    print(f"completed: {suite}/{config.stem}")
                if not args.skip_details:
                    detail = OUTPUT_ROOT / "exp3" / config.stem
                    if args.force or not details_current(config, detail):
                        subprocess.run([
                            sys.executable,
                            str(ROOT / "scripts/DATE3/export_date3_details.py"),
                            str(config), str(detail),
                        ], cwd=ROOT, check=True)
                        print(f"details completed: {suite}/{config.stem}")
                    else:
                        print(f"resume: valid details exist for {suite}/{config.stem}")
                continue
            if args.dry_run:
                print(f"{'resume valid' if current(config, output) and not args.force else 'run'}: {config} -> {output}")
                if suite in DETAIL_SUITES:
                    exp = {value: key for key, value in EXP_TO_SUITE.items()}[suite]
                    detail = OUTPUT_ROOT / exp / config.stem
                    state = "skipped" if args.skip_details else (
                        "resume valid" if details_current(config, detail) and not args.force
                        else "run missing/stale"
                    )
                    print(f"detail [{state}]: {config} -> {detail}")
                continue
            if current(config, output) and not args.force:
                with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
                    summaries.extend(csv.DictReader(stream))
                print(f"resume: {suite}/{config.stem}")
            else:
                execution = run_pivot_ca_file(config, output)
                summaries.append(dict(execution.summary))
                if suite in BASELINE_SUITES:
                    original = load_runner_config(config)
                    controls = run_date3_ep_paper_controls(original)
                    write_comparison(config, output, controls, execution)
                print(f"completed: {suite}/{config.stem}")
            if suite in DETAIL_SUITES and not args.skip_details:
                exp = {value: key for key, value in EXP_TO_SUITE.items()}[suite]
                detail = OUTPUT_ROOT / exp / config.stem
                if args.force or not details_current(config, detail):
                    subprocess.run([
                        sys.executable,
                        str(ROOT / "scripts/DATE3/export_date3_details.py"),
                        str(config), str(detail),
                    ], cwd=ROOT, check=True)
                    print(f"details completed: {suite}/{config.stem}")
                else:
                    print(f"resume: valid details exist for {suite}/{config.stem}")
        if summaries and not args.dry_run:
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            path = OUTPUT_ROOT / f"summary_{suite}.csv"
            fields = []
            for row in summaries:
                fields.extend(key for key in row if key not in fields)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(summaries)
            print(f"wrote {path}")
    if not matched_any:
        parser.error("no DATE3 configuration matched the requested selection")
    if args.exp and not args.dry_run:
        from scripts.DATE3.build_experiment_compat import build_experiment
        try:
            build_experiment(args.exp)
        except FileNotFoundError as error:
            print(f"paper aggregation pending: {error}")


if __name__ == "__main__":
    main()
