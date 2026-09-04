"""Shared DATE3 suite plotting helper."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def plot_suite(suite):
    path=ROOT/f"outputs/DATE3/summary_{suite}.csv"
    if not path.exists(): raise SystemExit(f"{suite} suite has not been run")
    rows=list(csv.DictReader(path.open(newline="",encoding="utf-8")))
    labels=[r.get("workload_name") or r.get("variant") or str(i) for i,r in enumerate(rows)]
    values=[float(r["total_cycles"]) for r in rows]
    fig,ax=plt.subplots(figsize=(max(7,len(rows)*.4),4)); ax.bar(labels,values)
    ax.set(ylabel="Total cycles (lower is better)",title=f"DATE3 {suite}")
    ax.tick_params(axis="x",rotation=30); fig.tight_layout()
    target=ROOT/f"fig/DATE3/{suite}_total_cycles.pdf"; target.parent.mkdir(parents=True,exist_ok=True); fig.savefig(target); print(target)
