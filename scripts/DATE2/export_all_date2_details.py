"""Export detailed reports for DATE2 exp4-exp6 variants."""
from pathlib import Path
from export_date2_details import export
ROOT=Path(__file__).resolve().parents[2]
GROUPS=(("overall","exp4"),("window_chunk","exp5"),("robustness","exp6"))
def main():
    count=0
    for suite,exp in GROUPS:
        for config in sorted((ROOT/f"configs/MoE/DATE2/{suite}").glob("*.json")):
            export(config,ROOT/f"outputs/DATE2/{exp}/{config.stem}");count+=1
    print(f"Exported {count} detailed DATE2 variants")
if __name__=="__main__":main()
