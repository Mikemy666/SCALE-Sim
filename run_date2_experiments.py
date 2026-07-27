"""Execute and validate DATE2 simulator experiments."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
from scalesim.memory.memdomain_experiment import (
    Baseline, read_matrix, validate_matrix, workload_digest,
)
from scalesim.memory.memdomain_runner import run_matrix_file

ROOT=Path(__file__).resolve().parent
CONFIG_ROOT=ROOT/"configs/MoE/DATE2"; OUTPUT_ROOT=ROOT/"outputs/DATE2"
SUITES=("overall","window_chunk","joint_prefetch","robustness_factorial")
EXP_TO_SUITE={
    "exp4":"overall","exp5":"joint_prefetch","exp6":"robustness_factorial"
}
EXP6_VARIABLES=(
    "expert_count","token_count","top_k","expert_parallel",
    "routing_severity","routing_seed",
)
EXP6_MODELS=("HMoE","Mixtral","MoDSE","Switchtrans")

def current_rows(config: Path, output: Path):
    """Return validated rows only when an existing CSV matches its config."""
    if not output.exists():
        return None
    try:
        rows=validate_matrix(read_matrix(output))
        expected=workload_digest(json.loads(config.read_text(encoding="utf-8")))
        return rows if {row.workload_hash for row in rows}=={expected} else None
    except (OSError, ValueError, KeyError, AssertionError, TypeError, csv.Error):
        return None

def details_current(config: Path, matrix_output: Path, output_dir: Path) -> bool:
    required=(
        "CHUNK_REPORT.csv","EXPERT_REPORT.csv","FFN_STAGE_REPORT.csv",
        "LAYER_DOMINANCE_REPORT.csv","BANK_REPORT.csv","REQUEST_REPORT.csv",
        "EXPERT_INPUT_REPORT.csv","MEASURED_SELECTIONS.csv",
        "COMPILER_BANK_PLAN.csv",
    )
    if not all((output_dir/name).exists() for name in required):
        return False
    meta_path=output_dir/"DETAILS_META.json"
    # Compatibility for reports produced before DETAILS_META existed: reports
    # newer than both the matrix and config necessarily came from the current
    # completed variant run. Any later config/matrix update invalidates them.
    if not meta_path.exists():
        threshold=max(config.stat().st_mtime,matrix_output.stat().st_mtime)
        return min((output_dir/name).stat().st_mtime for name in required)>=threshold
    try:
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        expected=workload_digest(json.loads(config.read_text(encoding="utf-8")))
        return meta.get("workload_hash")==expected
    except (OSError,ValueError,TypeError):
        return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",choices=("all",)+SUITES,default="all")
    parser.add_argument("--exp",choices=("exp1","exp2","exp3","exp4","exp5","exp6"))
    parser.add_argument("--variant")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--force",action="store_true",
                        help="rerun valid matrices instead of resuming")
    parser.add_argument("--skip-details",action="store_true",
                        help="run validated matrices without replaying detailed reports")
    parser.add_argument("--exp6-variable",choices=EXP6_VARIABLES)
    parser.add_argument("--exp6-value")
    parser.add_argument("--exp6-model",choices=EXP6_MODELS)
    args=parser.parse_args()
    exp6_filters=(args.exp6_variable,args.exp6_value,args.exp6_model)
    if any(exp6_filters) and args.exp!="exp6":
        parser.error("--exp6-* filters require --exp exp6")
    if args.exp6_value and not args.exp6_variable:
        parser.error("--exp6-value requires --exp6-variable")
    if args.exp in ("exp1","exp2"):
        command=[sys.executable,str(ROOT/"scripts/DATE2/run_date2_characterization.py"),"--exp",args.exp]
        if args.dry_run: print(" ".join(command))
        else: subprocess.run(command,cwd=ROOT,check=True)
        return
    if args.exp=="exp3":
        matrix_command=[sys.executable,str(ROOT/"run_date2_experiments.py"),
                        "--suite","window_chunk"]
        if args.force: matrix_command.append("--force")
        aggregate_command=[sys.executable,
                           str(ROOT/"scripts/DATE2/run_date2_characterization.py"),
                           "--exp","exp3"]
        if args.dry_run:
            print(" ".join(matrix_command));print(" ".join(aggregate_command))
        else:
            subprocess.run(matrix_command,cwd=ROOT,check=True)
            subprocess.run(aggregate_command,cwd=ROOT,check=True)
        return
    suites=((EXP_TO_SUITE[args.exp],) if args.exp else
            (SUITES if args.suite=="all" else (args.suite,)))
    summary=[]
    matrix_count=0
    for suite in suites:
        configs=sorted((CONFIG_ROOT/suite).glob("*.json"))
        if args.variant: configs=[p for p in configs if p.stem==args.variant]
        if suite=="robustness_factorial" and any(exp6_filters):
            selected=[]
            for path in configs:
                sweep=json.loads(path.read_text(encoding="utf-8")).get("sweep",{})
                if args.exp6_variable and sweep.get("variable")!=args.exp6_variable:
                    continue
                if args.exp6_value and str(sweep.get("value"))!=args.exp6_value:
                    continue
                if args.exp6_model and sweep.get("model")!=args.exp6_model:
                    continue
                selected.append(path)
            configs=selected
        if not configs:
            parser.error(
                f"no configuration matched suite={suite!r}, "
                f"variant={args.variant!r}"
            )
        for config in configs:
            output=OUTPUT_ROOT/suite/f"{config.stem}.csv"
            if args.dry_run:
                exp={v:k for k,v in EXP_TO_SUITE.items()}[suite]
                matrix_state=("force rerun" if args.force else
                              "resume valid" if current_rows(config,output) is not None
                              else "run missing/stale")
                detail_dir=OUTPUT_ROOT/exp/config.stem
                detail_state=("skipped" if args.skip_details else
                              "force rerun" if args.force else
                              "resume valid"
                              if output.exists() and details_current(
                                  config,output,detail_dir)
                              else "run missing/stale")
                print(f"matrix [{matrix_state}]: {config} -> {output}")
                print(f"detail [{detail_state}]: {config} -> {detail_dir}")
                continue
            rows=None if args.force else current_rows(config,output)
            if rows is None:
                output.parent.mkdir(parents=True,exist_ok=True)
                temporary=output.with_suffix(".csv.tmp")
                if temporary.exists():
                    temporary.unlink()
                rows=validate_matrix(run_matrix_file(config,temporary))
                temporary.replace(output)
                print(f"completed: {suite}/{config.stem}")
            else:
                print(f"resume: valid output exists for {suite}/{config.stem}")
            matrix_count+=1
            by={row.baseline:row for row in rows}; static=by[Baseline.STATIC_NOPF.value]
            safe=by[Baseline.MEMDOMAIN_SAFE.value]; oracle=by[Baseline.ORACLE.value]
            assert safe.total_cycles<=static.total_cycles
            assert oracle.total_cycles==min(row.total_cycles for row in rows)
            payload=json.loads(config.read_text(encoding="utf-8"))
            for row in rows:
                # Raw and Oracle remain internal diagnostics.  The paper
                # reports one implementable final scheme, "MemDomain", whose
                # compiler search contains the fixed incumbent.
                if row.baseline in (
                    Baseline.MEMDOMAIN_RAW.value, Baseline.ORACLE.value
                ):
                    continue
                public_baseline = (
                    "MemDomain"
                    if row.baseline == Baseline.MEMDOMAIN_SAFE.value
                    else row.baseline
                )
                summary.append({"suite":suite,"variant":config.stem,"workload":row.workload_name,
                    "baseline":public_baseline,"total_cycles":row.total_cycles,
                    "normalized_cycles":row.total_cycles/static.total_cycles,
                    "speedup":static.total_cycles/row.total_cycles,
                    "memory_stall_cycles":row.total_cycles-row.compute_cycles,
                    "bank_conflict_rate":row.bank_conflict_rate,
                    "prefetch_coverage":row.prefetch_coverage,
                    "timely_prefetch_ratio":row.timely_prefetch_ratio,
                    "communication_stall_cycles":row.communication_stall_cycles,
                    "sweep":json.dumps(payload.get("sweep",{}),sort_keys=True)})
            exp={v:k for k,v in EXP_TO_SUITE.items()}[suite]
            detail_dir=OUTPUT_ROOT/exp/config.stem
            if args.skip_details:
                print(f"details skipped: {suite}/{config.stem}")
            elif args.force or not details_current(config,output,detail_dir):
                subprocess.run([
                    sys.executable,str(ROOT/"scripts/DATE2/export_date2_details.py"),
                    str(config),str(detail_dir)
                ],cwd=ROOT,check=True)
                print(f"details completed: {suite}/{config.stem}")
            else:
                print(f"resume: valid details exist for {suite}/{config.stem}")
    if not args.dry_run and summary:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        suffix=[]
        if args.variant: suffix.append(args.variant)
        if args.exp6_variable: suffix.append(args.exp6_variable)
        if args.exp6_value: suffix.append(args.exp6_value)
        if args.exp6_model: suffix.append(args.exp6_model)
        label="_".join([args.exp or args.suite,*suffix])
        path=OUTPUT_ROOT/("summary_all.csv" if label=="all" else f"summary_{label}.csv")
        with path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=tuple(summary[0])); writer.writeheader(); writer.writerows(summary)
        print(f"Validated {matrix_count} matrices; wrote {path}")

if __name__=="__main__": main()
