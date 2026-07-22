"""Run the paper IV-B overall comparison on four controlled MoE topologies."""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.memdomain_experiment import Baseline
from scalesim.memory.memdomain_runner import load_runner_config, run_matrix_file
from scalesim.memory.moe_workload_catalog import write_runner_payload
from scalesim.memory.topology_workload import generate_topology_runner_payload


MODELS = (
    ("HMoE", "heterogeneous"),
    ("Mixtral", "homogeneous"),
    ("MoDSE", "heterogeneous"),
    ("Switchtrans", "homogeneous"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/MoE_prefetch1/p10")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for model, model_class in MODELS:
        topology = ROOT / f"topologies/MoE/{model}.csv"
        payload = generate_topology_runner_payload(topology, model_class)
        config_path = args.output / f"{model}.json"
        matrix_path = args.output / f"{model}.csv"
        write_runner_payload(config_path, payload)
        rows = run_matrix_file(config_path, matrix_path)
        static = next(row for row in rows if row.baseline == Baseline.STATIC_NOPF.value)
        for row in rows:
            summary.append({
                "model": model, "model_class": model_class,
                "baseline": row.baseline, "total_cycles": row.total_cycles,
                "normalized_cycles": row.total_cycles / static.total_cycles,
                "speedup": static.total_cycles / row.total_cycles,
                "bank_conflict_rate": row.bank_conflict_rate,
                "memory_stall_cycles": row.total_cycles - row.compute_cycles,
            })
    with (args.output / "overall_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
