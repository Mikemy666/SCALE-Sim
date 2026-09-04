"""Public DATE3 Exp1--Exp7 compatibility contract.

Internal diagnostic rows remain available in baseline matrices.  Paper plots
use one stable public name per implementable mechanism and never expose Raw or
Oracle as additional proposed schemes.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/MoE/DATE3"
OUTPUT_ROOT = ROOT / "outputs/DATE3"
FIG_ROOT = ROOT / "fig/DATE3"

MODELS = ("HMoE", "Mixtral", "MoDSE", "Switchtrans")
PUBLIC_BASELINES = (
    "Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF",
    "Static-Opt-FixedPF", "Dynamic-FixedPF", "PIVOT",
)
ROBUSTNESS_SCHEMES = PUBLIC_BASELINES
EXP4_MAPPING_SCHEMES = (
    "Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF", "Ideal-NoPF",
)
REFERENCE_ONLY = ("Ideal-NoPF",)
# PIVOT is the paper name of the MemDomain architecture.  The legacy
# MemDomain-Safe/Raw rows are retained in baseline_matrix.csv only for
# diagnosis; exposing Safe as a second public scheme would double-count the
# proposed architecture.
INTERNAL_TO_PUBLIC = {
    "PIVOT-CA": "PIVOT",
    "Static-NoPF": "Static-Opt-NoPF",
    "Static-NaivePF": "Static-Opt-FixedPF",
    "Dynamic-NaivePF": "Dynamic-FixedPF",
}
ANALYSIS_ONLY = {"MemDomain-Raw", "MemDomain-Safe", "Oracle"}

WINDOWS = (0, 1, 2, 4, 8, 16, 32, 64)
CHUNKS = (1, 2, 4, 8)
EXP_TO_SUITE = {
    "exp4": "overall",
    "exp5": "joint_prefetch",
    "exp6": "robustness_factorial",
    "exp7": "end_to_end",
}

LOCAL_MEMORY_COMPONENTS = (
    "bank_stall_cycles",
    "weight_load_stall_cycles",
    "prefetch_miss_stall_cycles",
    "prefetch_interference_stall_cycles",
    "mapping_overhead_cycles",
)


def public_name(internal: str) -> str:
    return INTERNAL_TO_PUBLIC.get(internal, internal)


def local_memory_stall(row) -> int:
    """Return on-chip/local-memory stall, excluding EP/Combine time."""
    return sum(int(float(row[name])) for name in LOCAL_MEMORY_COMPONENTS)
