"""Generate scaled, provenance-preserving workloads from the model catalog."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.moe_workload_catalog import (
    generate_runner_payload,
    load_catalog,
    write_runner_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", type=Path,
        default=ROOT / "configs/MoE/MoE_prefetch1/workloads/catalog.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "configs/MoE/MoE_prefetch1/workloads/generated",
    )
    parser.add_argument("--dimension-divisor", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    for spec in load_catalog(args.catalog):
        payload = generate_runner_payload(
            spec, dimension_divisor=args.dimension_divisor, tokens=args.tokens
        )
        path = args.output / f"{spec.model_id}.json"
        write_runner_payload(path, payload)
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
