"""Validate one canonical MoE_prefetch1 experiment matrix CSV."""

import argparse
from pathlib import Path

from scalesim.memory.memdomain_experiment import read_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    rows = read_matrix(args.matrix)
    print(
        f"Valid MemDomain matrix: experiment={rows[0].experiment_id} "
        f"workload={rows[0].workload_name} rows={len(rows)}"
    )


if __name__ == "__main__":
    main()
