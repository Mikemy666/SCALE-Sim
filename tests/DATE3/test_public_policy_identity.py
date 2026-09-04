"""Public DATE3 must not double-count PIVOT and MemDomain."""

from scripts.DATE3.experiment_contract import (
    ANALYSIS_ONLY, PUBLIC_BASELINES, public_name,
)


def test_pivot_is_the_only_public_memdomain_name():
    assert PUBLIC_BASELINES == (
        "Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF",
        "Static-Opt-FixedPF", "Dynamic-FixedPF", "PIVOT",
    )
    assert public_name("PIVOT-CA") == "PIVOT"
    assert {"MemDomain-Safe", "MemDomain-Raw", "Oracle"} <= ANALYSIS_ONLY
    assert "MemDomain" not in PUBLIC_BASELINES
    assert "PIVOT-CA" not in PUBLIC_BASELINES
