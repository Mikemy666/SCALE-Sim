"""Summarize available DATE3 outputs; never reads DATE2 analysis files."""

from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows=[]
    for path in sorted((ROOT/"outputs/DATE3").glob("*/ */summary.csv".replace(" ",""))):
        with path.open(newline="",encoding="utf-8") as stream:
            row=next(csv.DictReader(stream)); row["source"]=str(path.relative_to(ROOT)); rows.append(row)
    target=ROOT/"outputs/DATE3/analysis.json"; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"wrote {target} with {len(rows)} rows")


if __name__=="__main__": main()

