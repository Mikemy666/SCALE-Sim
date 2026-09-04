"""Export NPU-local Bank/Chunk reports plus DATE3 EP reports for one variant."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.date3_ep_model import localize_detailed_npu
from scalesim.memory.memdomain_runner import load_runner_config
from scalesim.memory.pivot_ca_runner import implementation_digest
from scripts.DATE2.export_date2_details import export_runner_config
from scripts.DATE3.token_trace import export_token_trace


def _write(path: Path, rows) -> None:
    values = list(rows)
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(values[0]))
        writer.writeheader()
        writer.writerows(values)


def export(config_path: Path, output_dir: Path, *, token_trace: str = "full") -> None:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    original = load_runner_config(config_path)
    detailed = localize_detailed_npu(original)
    export_runner_config(
        detailed.config, output_dir,
        identity_payload=payload, source_config=str(config_path.resolve()),
    )
    metadata_path = output_dir / "DETAILS_META.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["implementation_hash"] = implementation_digest()
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    routes = detailed.routes
    _write(output_dir / "EP_ROUTE_REPORT.csv", (item.to_dict() for item in routes))
    # The policy-independent report records ownership and exact route traffic.
    _write(output_dir / "NPU_WORKLOAD_REPORT.csv", [detailed.summary_row()])
    system = payload["system"]
    remote = detailed.remote_route_replicas
    _write(output_dir / "EP_COMMUNICATION_REPORT.csv", [{
        "remote_route_replicas": remote,
        "dispatch_bytes": remote * int(system["token_payload_bytes"]),
        "return_bytes": remote * int(system["result_payload_bytes"]),
        "top_k": detailed.contract.top_k,
        "num_npus": detailed.contract.num_npus,
        "communication_model": "startup_plus_bandwidth_no_packet_contention",
    }])
    trace = export_token_trace(output_dir, payload, detailed, mode=token_trace)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["token_trace"] = {
        "schema_version": trace.get("schema_version", 1),
        "mode": token_trace,
        "files": trace.get("files", []),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--token-trace", choices=("none", "summary", "sampled", "full"),
        default="full", help="Token trace detail level (default: compressed full trace)",
    )
    args = parser.parse_args()
    export(args.config, args.output, token_trace=args.token_trace)


if __name__ == "__main__":
    main()
