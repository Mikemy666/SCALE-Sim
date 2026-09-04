"""Structural tests for the DATE3 paper-control expansion."""

from pathlib import Path

from scalesim.memory.memdomain_runner import (
    compiled_dynamic_config, compiler_bank_service_cycles,
    load_runner_config, profiled_static_allocation, static_allocation_config,
)
from scalesim.memory.date3_ep_model import localize_detailed_npu
from scalesim.memory.buckyball_memdomain import STATIC_ALLOCATION
from scripts.DATE3.experiment_contract import (
    EXP4_MAPPING_SCHEMES, PUBLIC_BASELINES,
)


ROOT = Path(__file__).resolve().parents[2]


def test_public_group_identity_is_unambiguous():
    assert len(PUBLIC_BASELINES) == len(set(PUBLIC_BASELINES)) == 6
    assert EXP4_MAPPING_SCHEMES == (
        "Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF", "Ideal-NoPF",
    )
    assert "PIVOT" in PUBLIC_BASELINES
    assert "MemDomain" not in PUBLIC_BASELINES


def test_profiled_static_plan_is_one_complete_partition():
    config = load_runner_config(
        ROOT / "configs/MoE/DATE3/unit_cases/MoDSE_minimal.json"
    )
    allocation = profiled_static_allocation(config)
    assert allocation.total == config.resources.bank_count == 30
    frozen = static_allocation_config(config, allocation)
    groups = dict(frozen.static_bank_groups)
    assert set(groups) == {"ia", "weight", "oa", "accumulator"}
    assert sorted(bank for value in groups.values() for bank in value) == list(range(30))


def test_static_555_remains_a_legal_explicit_incumbent():
    config = load_runner_config(
        ROOT / "configs/MoE/DATE3/unit_cases/MoDSE_minimal.json"
    )
    frozen = static_allocation_config(config, STATIC_ALLOCATION)
    assert tuple(len(value) for _, value in frozen.static_bank_groups) == (5, 5, 5, 15)


def test_dynamic_executes_stage_plans_and_reduces_model_bank_service():
    for model in ("HMoE", "Mixtral", "MoDSE", "Switchtrans"):
        config = localize_detailed_npu(load_runner_config(
            ROOT / f"configs/MoE/DATE3/overall/{model}.json"
        )).config
        static = static_allocation_config(
            config, profiled_static_allocation(config)
        )
        dynamic = compiled_dynamic_config(config)
        assert dynamic.dynamic_honor_preferred_banks
        assert len(dynamic.dynamic_weight_bank_pools) == len(dynamic.chunks)
        assert compiler_bank_service_cycles(dynamic) < compiler_bank_service_cycles(static)
