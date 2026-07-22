"""Run all canonical MemDomain baselines for one workload JSON."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.memdomain_runner import run_matrix_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run_matrix_file(args.config, args.output)
    for row in rows:
        print(f"{row.baseline}: {row.total_cycles} cycles")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
