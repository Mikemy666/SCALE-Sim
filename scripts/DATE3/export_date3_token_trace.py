"""Export DATE3 Token traces without rerunning memory-performance simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.date3_ep_model import localize_detailed_npu
from scalesim.memory.memdomain_runner import load_runner_config
from scripts.DATE3.token_trace import export_token_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--token-trace", choices=("summary", "sampled", "full"),
        default="full",
    )
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    detailed = localize_detailed_npu(load_runner_config(args.config))
    summary = export_token_trace(
        args.output, payload, detailed, mode=args.token_trace,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "all_checks_pass": summary["all_checks_pass"],
        "files": summary["files"],
    }, indent=2))


if __name__ == "__main__":
    main()
