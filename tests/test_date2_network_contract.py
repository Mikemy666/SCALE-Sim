from pathlib import Path

from scalesim.memory.date2_network_contract import validate_network_set


def test_four_models_share_one_scale_and_top1_imbalance():
    root = Path(__file__).resolve().parents[1]
    names = ("HMoE", "Mixtral", "MoDSE", "Switchtrans")
    report = validate_network_set([
        root / "topologies" / "MoE" / f"{name}.csv" for name in names
    ])
    assert {item.hidden_size for item in report.values()} == {384}
    assert {item.total_tokens for item in report.values()} == {256}
    assert report["Mixtral"].homogeneous_expert_weights
    assert report["Switchtrans"].homogeneous_expert_weights
    assert not report["HMoE"].homogeneous_expert_weights
    assert not report["MoDSE"].homogeneous_expert_weights
