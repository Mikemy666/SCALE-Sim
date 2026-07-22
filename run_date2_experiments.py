"""Execute and validate DATE2 simulator experiments."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from scalesim.memory.memdomain_experiment import Baseline, validate_matrix
from scalesim.memory.memdomain_runner import run_matrix_file

ROOT=Path(__file__).resolve().parent
CONFIG_ROOT=ROOT/"configs/MoE/DATE2"; OUTPUT_ROOT=ROOT/"outputs/DATE2"
SUITES=("overall","ablation","window_chunk","robustness")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",choices=("all",)+SUITES,default="all")
    parser.add_argument("--variant")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(); suites=SUITES if args.suite=="all" else (args.suite,)
    summary=[]
    for suite in suites:
        configs=sorted((CONFIG_ROOT/suite).glob("*.json"))
        if args.variant: configs=[p for p in configs if p.stem==args.variant]
        for config in configs:
            output=OUTPUT_ROOT/suite/f"{config.stem}.csv"
            if args.dry_run:
                print(f"{config} -> {output}"); continue
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
    if not args.dry_run and summary:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        path=OUTPUT_ROOT/("summary_all.csv" if args.suite=="all" else f"summary_{args.suite}.csv")
        with path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=tuple(summary[0])); writer.writeheader(); writer.writerows(summary)
        print(f"Validated {len(summary)//7} matrices; wrote {path}")

if __name__=="__main__": main()
