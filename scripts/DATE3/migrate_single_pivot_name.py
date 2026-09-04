"""Normalize completed DATE3 comparisons to one public PIVOT scheme.

This is an aggregation/schema migration only.  It never launches simulation
and never changes baseline_matrix.csv or the internal diagnostic rows.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/DATE3"
SUITES = ("overall", "joint_prefetch", "robustness_factorial", "end_to_end")


def migrate(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    changed = False
    normalized = []
    for row in rows:
        name = row.get("policy_name")
        if name == "MemDomain":
            changed = True
            continue
        if name == "PIVOT-CA":
            row["policy_name"] = "PIVOT"
            changed = True
        normalized.append(row)
    names = [row["policy_name"] for row in normalized]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate policies after PIVOT migration: {path}")
    if changed:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(normalized)
    return changed


def main() -> None:
    changed = []
    for suite in SUITES:
        directory = OUTPUT_ROOT / suite
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*/comparison.csv")):
            if migrate(path):
                changed.append(path)
    print(f"normalized {len(changed)} DATE3 comparison files to single PIVOT")


if __name__ == "__main__":
    main()
