"""Plot minimal DATE3 quality and runtime adaptation evidence."""

from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/"outputs/DATE3/unit_cases/MoDSE_minimal"
    with (source/"quality_epochs.csv").open(newline="",encoding="utf-8") as stream:
        epochs=list(csv.DictReader(stream))
    with (source/"decision_detail.csv").open(newline="",encoding="utf-8") as stream:
        decisions=[row for row in csv.DictReader(stream) if row["selected"].lower()=="true"]
    x=[int(row["epoch_id"]) for row in epochs]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(x,[float(row["coverage"]) if row["coverage"] else float("nan") for row in epochs],"o-",label="Coverage")
    axes[0].plot(x,[float(row["accuracy"]) if row["accuracy"] else float("nan") for row in epochs],"s-",label="Accuracy")
    axes[0].set(xlabel="Runtime epoch",ylabel="Quality ratio",ylim=(0,1.05),title="Measured quality")
    axes[0].legend()
    dx=[int(row["decision_id"]) for row in decisions]
    axes[1].step(dx,[int(row["candidate_chunk"]) for row in decisions],where="mid",label="Chunk (tiles)")
    axes[1].step(dx,[int(row["candidate_window"]) for row in decisions],where="mid",label="Window")
    axes[1].set(xlabel="Runtime decision",ylabel="Selected value",title="Online adaptation")
    axes[1].legend()
    fig.tight_layout()
    target=ROOT/"fig/DATE3/minimal_quality_adaptation.pdf"; target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,bbox_inches="tight"); print(f"wrote {target}")


if __name__=="__main__": main()

