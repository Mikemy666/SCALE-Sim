"""Build the Exp5 non-stationary, multi-layer evaluation workload.

Four consecutive MoE layers share one PIVOT controller instance.  Fixed
prefetch controls freeze the independently calibrated Window/Chunk pair for
the complete sequence, while PIVOT may adapt at runtime across layer phases.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.date3_ep_model import EPContract, attach_ep_contract
from scalesim.memory.topology_workload import (
    generate_topology_runner_payload, load_moe_topology,
)
from scripts.DATE3.prepare_date3_experiments import POLICY, save
from scripts.prepare_date2_experiments import skewed, skewed_seed, variant_topology

CONFIG_ROOT = ROOT / "configs/MoE/DATE3"
TOPOLOGY_ROOT = ROOT / "topologies/MoE/DATE3"
OUTPUT_ROOT = ROOT / "outputs/DATE3"
WINDOWS = (0, 1, 2, 4, 8, 16, 32, 64)
CHUNKS = (1, 2, 4, 8)
EXPERTS_PER_LAYER = 8
LAYER_GAP_CYCLES = 64
PHASE_ROOT = TOPOLOGY_ROOT / "joint_prefetch/phases"
PHASES = (
    ("balanced_t64", PHASE_ROOT / "L0_balanced_t64.csv"),
    ("light40_t64", PHASE_ROOT / "L1_light40_t64.csv"),
    ("high41_t64", PHASE_ROOT / "L2_high41_t64.csv"),
    ("light42_t128", PHASE_ROOT / "L3_light42_t128.csv"),
)

# The independently frozen fixed-prefetch shadow has very low timely-byte
# coverage on this bandwidth-constrained four-layer trace.  An absolute 95%
# floor is therefore unattainable and permanently locks the controller into
# its reference fallback.  Exp5 uses relative online quality feedback with no
# absolute floor, while retaining late-ratio/timing-error EMAs, admission
# control, and the measured three-way incumbent guard.
EXP5_RELATIVE_QUALITY = {
    "min_coverage": 0.0,
    "min_accuracy": 0.0,
    "epsilon_coverage": 1.0,
    "epsilon_accuracy": 1.0,
}


def _selection() -> tuple[int, int, str]:
    path = OUTPUT_ROOT / "exp5/deployable_selection.csv"
    if not path.exists():
        raise FileNotFoundError(
            "run the independent Exp5 calibration before building multi-layer configs"
        )
    import csv
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["policy_name"]: row for row in csv.DictReader(stream)}
    row = rows["Dynamic-FixedPF"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return int(row["selected_window"]), int(row["selected_chunk_tiles"]), digest


def prepare_phase_topologies() -> None:
    source = TOPOLOGY_ROOT / "models/MoDSE.csv"
    raw_counts = (
        skewed(64, "balanced"),
        skewed_seed(64, "light", 40),
        skewed_seed(64, "high", 41),
        skewed_seed(128, "light", 42),
    )
    counts = []
    for row in raw_counts:
        values = list(row)
        for expert, value in enumerate(values):
            if value > 0:
                continue
            donor = max(range(len(values)), key=values.__getitem__)
            values[donor] -= 1
            values[expert] = 1
        counts.append(tuple(values))
    for (_, target), values in zip(PHASES, counts):
        variant_topology(target, source, values)


def _rename_request(value: str, old_expert: int, new_expert: int) -> str:
    return re.sub(
        rf"(?P<prefix>(?:compute|acc)_e){old_expert}(?=_ff[12](?:_|$))",
        rf"\g<prefix>{new_expert}",
        value,
    )


def combined_payload(chunk_tiles: int):
    phase_payloads = [
        generate_topology_runner_payload(
            topology, "heterogeneous", chunk_size_bytes=chunk_tiles * 256,
            top_k=1, num_gpus=2,
        )
        for _, topology in PHASES
    ]
    value = copy.deepcopy(phase_payloads[0])
    value["chunks"] = []
    value["compute_requests"] = []
    value["compiler_bank_plans"] = []
    value["compute_intervals"] = []
    counts = []
    layer_profiles = []
    cursor = 0
    address_stride = 1 << 24
    for layer_id, ((profile, topology), phase) in enumerate(zip(PHASES, phase_payloads)):
        expert_base = layer_id * EXPERTS_PER_LAYER
        phase_counts = list(phase["topology_provenance"]["token_counts"])
        if len(phase_counts) != EXPERTS_PER_LAYER:
            raise ValueError(f"Exp5 phase {profile} must have eight experts")
        counts.extend(phase_counts)
        layer_profiles.append({
            "layer_id": layer_id,
            "profile": profile,
            "token_counts": phase_counts,
            "source_topology": str(topology),
            "start_cycle": cursor,
        })
        for item in phase["chunks"]:
            copied = dict(item)
            old = int(copied["expert_id"])
            new = expert_base + old
            copied.update({
                "chunk_id": f"L{layer_id}__{copied['chunk_id']}",
                "expert_id": new,
                "use_cycle": int(copied["use_cycle"]) + cursor,
                "logical_address": int(copied["logical_address"])
                    + layer_id * address_stride,
            })
            value["chunks"].append(copied)
        for item in phase["compute_requests"]:
            copied = dict(item)
            request_id = str(copied["request_id"])
            match = re.search(r"(?:compute|acc)_e(\d+)_ff[12](?:_|$)", request_id)
            if match:
                old = int(match.group(1))
                new = expert_base + old
                copied["request_id"] = _rename_request(
                    request_id, old, new
                )
                object_id = re.sub(
                    rf"_e{old}(?=_ff[12](?:_|$))", f"_e{new}",
                    str(copied["object_id"]),
                )
            else:
                copied["request_id"] = f"L{layer_id}__{request_id}"
                object_id = str(copied["object_id"])
            copied["object_id"] = f"L{layer_id}__{object_id}"
            copied["issue_cycle"] = int(copied["issue_cycle"]) + cursor
            copied["address"] = int(copied["address"]) + layer_id * address_stride
            value["compute_requests"].append(copied)
        dimensions = load_moe_topology(topology)["experts"]
        for plan in phase["compiler_bank_plans"]:
            copied = dict(plan)
            match = re.fullmatch(r"MoE-E(\d+)-FF([12])", str(copied["layer"]))
            if match is None:
                raise ValueError(f"invalid compiler plan layer: {copied['layer']}")
            old_expert, part = map(int, match.groups())
            m, n, k = dimensions[(old_expert, part)]
            copied.update({
                "layer": f"L{layer_id}__{copied['layer']}",
                "expert_id": expert_base + old_expert,
                "ffn_part": part,
                "m": m, "n": n, "k": k,
            })
            value["compiler_bank_plans"].append(copied)
        end = cursor + int(phase["compute_cycles"])
        value["compute_intervals"].append([cursor, end])
        cursor = end + LAYER_GAP_CYCLES

    value["compute_cycles"] = cursor - LAYER_GAP_CYCLES
    value["workload_name"] = "MoDSE-4Layer-NonStationary"
    provenance = dict(value["topology_provenance"])
    provenance.update({
        "source_path": "multi_layer_exp5_sequence",
        "token_counts": counts,
        "total_tokens": sum(counts),
        "chunk_size_bytes": chunk_tiles * 256,
        "exp5_multi_layer": True,
        "layer_count": len(PHASES),
        "experts_per_layer": EXPERTS_PER_LAYER,
        "layer_profiles": layer_profiles,
    })
    value["topology_provenance"] = provenance
    value["multi_layer_prefetch"] = {
        "enabled": True,
        "layer_count": len(PHASES),
        "experts_per_layer": EXPERTS_PER_LAYER,
        "controller_state": "persistent_across_layers",
        "fixed_knobs": "one_pair_frozen_across_layers",
        "layer_gap_cycles": LAYER_GAP_CYCLES,
        "profiles": [profile for profile, _ in PHASES],
    }
    return value


def finish(value, suite: str, variant: str, window: int, chunk: int,
           reference_window: int, reference_chunk: int, selection_hash: str,
           overrides=None):
    value = copy.deepcopy(value)
    value["experiment_id"] = f"date3-{suite}-{variant}"
    value["date3_suite"] = suite
    value["date3_variant"] = variant
    value["policy"]["prefetch_window"] = window
    value["policy"]["prefetch_policy"] = "coverage_accuracy_constrained"
    value["coverage_accuracy_policy"] = copy.deepcopy(POLICY)
    value["coverage_accuracy_policy"].update(EXP5_RELATIVE_QUALITY)
    value["coverage_accuracy_policy"].update({
        "reference_window": reference_window,
        "reference_chunk": reference_chunk,
    })
    if overrides:
        value["coverage_accuracy_policy"].update(overrides)
    value["sweep"] = {
        "variable": "multi_layer_window_chunk",
        "model": "MoDSE",
        "window": window,
        "chunk_tiles": chunk,
        "layer_count": len(PHASES),
    }
    value["paper_control_contract"] = {
        "selection_protocol": "independent_calibration_then_freeze",
        "test_trace_visible_during_selection": False,
        "dynamic_fixed_reference": {
            "window": reference_window,
            "chunk_tiles": reference_chunk,
            "selection_hash": selection_hash,
        },
        "test_grid_role": "analysis_only_oracle",
    }
    value = attach_ep_contract(value)
    # EP ownership repeats 4+4 within every physical MoE layer.  A contiguous
    # 32+32 split would incorrectly assign whole layers rather than experts.
    ep = dict(value["ep"])
    ep["expert_owner_map"] = [
        0 if expert % EXPERTS_PER_LAYER < EXPERTS_PER_LAYER // 2 else 1
        for expert in range(len(ep["expert_owner_map"]))
    ]
    value["ep"] = ep
    EPContract.from_payload(value).routes()
    return value


def main() -> None:
    prepare_phase_topologies()
    ref_window, ref_chunk, selection_hash = _selection()
    bases = {chunk: combined_payload(chunk) for chunk in CHUNKS}
    for window in WINDOWS:
        for chunk in CHUNKS:
            variant = f"w{window}_c{chunk}"
            value = finish(
                bases[chunk], "joint_prefetch", variant, window, chunk,
                ref_window, ref_chunk, selection_hash,
            )
            save(CONFIG_ROOT / "joint_prefetch" / f"{variant}.json", value)

    ablations = {
        # Disable feedback adaptation, not merely the now-zero absolute floor.
        "without_quality": {
            "eta_coverage": 0.0, "eta_accuracy": 0.0,
            "timing_margin_scale": 0.0, "severe_late_ratio": 1.0,
        },
        "coverage_only": {"min_coverage": 0.05},
        "accuracy_only": {"min_accuracy": 0.05},
        "both_constraints": {"min_coverage": 0.05, "min_accuracy": 0.05},
        "without_bank_pressure": {"pressure_threshold": 1.0,
                                  "score_weights": {**POLICY["score_weights"],
                                                    "pressure": 0.0}},
        "without_dynamic_chunk": {"candidate_chunks": [ref_chunk]},
        "without_dynamic_window": {"candidate_windows": [ref_window]},
        "full": {},
    }
    for name, overrides in ablations.items():
        variant = f"MoDSE__{name}"
        value = finish(
            bases[ref_chunk], "ablation", variant, ref_window, ref_chunk,
            ref_window, ref_chunk, selection_hash, overrides,
        )
        save(CONFIG_ROOT / "ablation" / f"{variant}.json", value)

    manifest_path = CONFIG_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exp5_multi_layer"] = {
        "layer_count": len(PHASES),
        "profiles": [name for name, _ in PHASES],
        "controller_state": "persistent_across_layers",
        "fixed_knobs": "frozen_across_complete_sequence",
        "dynamic_fixed_reference": {
            "window": ref_window, "chunk_tiles": ref_chunk,
            "selection_hash": selection_hash,
        },
    }
    save(manifest_path, manifest)
    print(f"generated multi-layer Exp5 with frozen reference W={ref_window}, C={ref_chunk}")


if __name__ == "__main__":
    main()
