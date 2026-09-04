"""Run DATE3 Exp1/Exp2 with the established P=1 microarchitecture contract.

The implementation reuses the maintained DATE2 characterization functions but
redirects every topology/config/output root to DATE3.  No DATE2 artifact is
written.  P=1 is intentional: these two experiments isolate the local fixed
SP/ACC boundary before the P=2 end-to-end experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.DATE2 import run_date2_characterization as characterization

CFG = ROOT / "configs/MoE/DATE3"
TOP = ROOT / "topologies/MoE/DATE3"
OUT = ROOT / "outputs/DATE3"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(characterization.__file__)):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _current() -> bool:
    implementation = _implementation_hash()
    topology = _hash(TOP / "models/MoDSE.csv")
    required = {
        "exp1": ("layer_characterization.csv", "accumulator_sensitivity.csv",
                 "temporal_bank_demand.csv"),
        "exp2": ("static_bank_sweep.csv", "per_stage_best.csv"),
    }
    for name, files in required.items():
        metadata = OUT / name / "CHARACTERIZATION_META.json"
        if not metadata.exists() or not all((OUT / name / item).exists() for item in files):
            return False
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if value.get("config_sha256") != _hash(CFG / name / f"{name}.json"):
            return False
        if value.get("topology_sha256") != topology:
            return False
        if value.get("implementation_hash") != implementation:
            return False
    return True


def run(exp: str, force: bool = False) -> None:
    if not force and _current():
        print("resume: DATE3 exp1/exp2 characterization is hash-valid")
        return
    characterization.CFG = CFG
    characterization.TOP = TOP
    characterization.OUT = OUT
    characterization.exp1_exp2()
    for name in ("exp1", "exp2"):
        config = CFG / name / f"{name}.json"
        output = OUT / name
        (output / "CHARACTERIZATION_META.json").write_text(json.dumps({
            "schema_version": 1,
            "experiment": name,
            "scope": "P1_local_microarchitecture",
            "config_sha256": _hash(config),
            "topology_sha256": _hash(TOP / "models/MoDSE.csv"),
            "implementation_hash": _implementation_hash(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DATE3 {exp} characterization completed (exp1/exp2 share one trace)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=("exp1", "exp2"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            f"{'resume valid' if _current() and not args.force else 'run'} "
            f"P=1 characterization: {TOP/'models/MoDSE.csv'} -> "
            f"{OUT/'exp1'}, {OUT/'exp2'}"
        )
        return
    run(args.exp, args.force)


if __name__ == "__main__":
    main()
