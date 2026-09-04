"""Print one Token's normalized DATE3 EP route and inherited FFN stages."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def _rows(path: Path):
    with gzip.open(path, mode="rt", newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="directory containing TOKEN_TRACE_* files")
    parser.add_argument("--token", type=int, required=True, help="global trace Token ID")
    parser.add_argument("--layer", type=int, help="optional layer filter")
    parser.add_argument("--topk-slot", type=int, help="optional Top-k replica filter")
    args = parser.parse_args()

    summary_path = args.output / "TOKEN_TRACE_SUMMARY.json"
    index_path = args.output / "TOKEN_TRACE_INDEX.csv.gz"
    if not summary_path.exists() or not index_path.exists():
        raise SystemExit(
            "full trace not found; rerun export_date3_details.py with --token-trace full"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matches = []
    for row in _rows(index_path):
        if int(row["token_id"]) != args.token:
            continue
        if args.layer is not None and int(row["layer_id"]) != args.layer:
            continue
        if args.topk_slot is not None and int(row["topk_slot"]) != args.topk_slot:
            continue
        matches.append(row)
    if not matches:
        raise SystemExit(f"no route replica found for Token {args.token}")

    print(f"Token {args.token}: {len(matches)} route replica(s)")
    for row in matches:
        locality = "remote" if int(row["is_remote"]) else "local"
        print(
            f"  L{row['layer_id']} layer-token={row['layer_token_id']} "
            f"slot={row['topk_slot']}: NPU{row['source_npu']} -> "
            f"E{row['global_expert_id']}@NPU{row['owner_npu']} ({locality})"
        )
        print(
            f"    FFN1 [{row['ffn1_start_cycle']}, {row['ffn1_end_cycle']})  "
            f"FFN2 [{row['ffn2_start_cycle']}, {row['ffn2_end_cycle']})"
        )
        print(
            f"    events: {row['dispatch_event_id']} -> {row['ffn1_stage_id']} -> "
            f"{row['ffn2_stage_id']} -> {row['return_event_id']} -> "
            f"{row['combine_event_id']}"
        )
    print(f"Timing semantics: {summary['trace_semantics']['timing']}")
    print(f"All trace checks pass: {summary['all_checks_pass']}")


if __name__ == "__main__":
    main()
