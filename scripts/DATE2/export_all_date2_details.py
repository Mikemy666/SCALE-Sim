"""Export detailed reports for current DATE2 exp4-exp6 matrices."""
from __future__ import annotations
import argparse
from pathlib import Path
from export_date2_details import export
ROOT=Path(__file__).resolve().parents[2]
GROUPS=(
    ("overall","exp4"),("joint_prefetch","exp5"),
    ("robustness_factorial","exp6"),
)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    # Import from the canonical runner so this utility has exactly the same
    # freshness contract as the public experiment entry point.
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from run_date2_experiments import current_rows, details_current

    count=0
    for suite,exp in GROUPS:
        for config in sorted((ROOT/f"configs/MoE/DATE2/{suite}").glob("*.json")):
            matrix=ROOT/f"outputs/DATE2/{suite}/{config.stem}.csv"
            output=ROOT/f"outputs/DATE2/{exp}/{config.stem}"
            if current_rows(config,matrix) is None:
                print(f"skip stale/missing matrix: {suite}/{config.stem}")
                continue
            if not args.force and details_current(config,matrix,output):
                print(f"resume valid details: {suite}/{config.stem}")
                continue
            export(config,output)
            count+=1
            print(f"details completed: {suite}/{config.stem}")
    print(f"Exported {count} detailed DATE2 variants")
if __name__=="__main__":main()
