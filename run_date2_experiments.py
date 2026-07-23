"""Execute and validate DATE2 simulator experiments."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
from scalesim.memory.memdomain_experiment import Baseline, validate_matrix
from scalesim.memory.memdomain_runner import run_matrix_file

ROOT=Path(__file__).resolve().parent
CONFIG_ROOT=ROOT/"configs/MoE/DATE2"; OUTPUT_ROOT=ROOT/"outputs/DATE2"
SUITES=("overall","ablation","window_chunk","robustness")
EXP_TO_SUITE={"exp4":"overall","exp5":"ablation","exp6":"window_chunk","exp7":"robustness"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",choices=("all",)+SUITES,default="all")
    parser.add_argument("--exp",choices=tuple(f"exp{i}" for i in range(1,8)))
    parser.add_argument("--variant")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if args.exp in ("exp1","exp2"):
        command=[sys.executable,str(ROOT/"scripts/DATE2/run_date2_characterization.py"),"--exp",args.exp]
        if args.dry_run: print(" ".join(command))
        else: subprocess.run(command,cwd=ROOT,check=True)
        return
    if args.exp=="exp3":
        matrix_command=[sys.executable,str(ROOT/"run_date2_experiments.py"),
                        "--suite","window_chunk"]
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
    for suite in suites:
        configs=sorted((CONFIG_ROOT/suite).glob("*.json"))
        if args.variant: configs=[p for p in configs if p.stem==args.variant]
        for config in configs:
            output=OUTPUT_ROOT/suite/f"{config.stem}.csv"
            if args.dry_run:
                exp={v:k for k,v in EXP_TO_SUITE.items()}[suite]
                print(f"matrix: {config} -> {output}")
                print(f"detail: {config} -> {OUTPUT_ROOT/exp/config.stem}")
                continue
            rows=validate_matrix(run_matrix_file(config,output))
            by={row.baseline:row for row in rows}; static=by[Baseline.STATIC_NOPF.value]
            safe=by[Baseline.MEMDOMAIN_SAFE.value]; oracle=by[Baseline.ORACLE.value]
            assert safe.total_cycles<=static.total_cycles
            assert oracle.total_cycles==min(row.total_cycles for row in rows)
            payload=json.loads(config.read_text(encoding="utf-8"))
            for row in rows:
                summary.append({"suite":suite,"variant":config.stem,"workload":row.workload_name,
                    "baseline":row.baseline,"total_cycles":row.total_cycles,
                    "normalized_cycles":row.total_cycles/static.total_cycles,
                    "speedup":static.total_cycles/row.total_cycles,
                    "memory_stall_cycles":row.total_cycles-row.compute_cycles,
                    "bank_conflict_rate":row.bank_conflict_rate,
                    "prefetch_coverage":row.prefetch_coverage,
                    "timely_prefetch_ratio":row.timely_prefetch_ratio,
                    "communication_stall_cycles":row.communication_stall_cycles,
                    "sweep":json.dumps(payload.get("sweep",{}),sort_keys=True)})
            exp={v:k for k,v in EXP_TO_SUITE.items()}[suite]
            subprocess.run([sys.executable,str(ROOT/"scripts/DATE2/export_date2_details.py"),
                            str(config),str(OUTPUT_ROOT/exp/config.stem)],cwd=ROOT,check=True)
    if not args.dry_run and summary:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        label=args.exp or args.suite
        path=OUTPUT_ROOT/("summary_all.csv" if label=="all" else f"summary_{label}.csv")
        with path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=tuple(summary[0])); writer.writeheader(); writer.writerows(summary)
        print(f"Validated {len(summary)//7} matrices; wrote {path}")

if __name__=="__main__": main()
