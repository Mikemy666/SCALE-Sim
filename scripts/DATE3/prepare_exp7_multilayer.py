"""Build the four-block DATE3 end-to-end evaluation workloads.

Each model executes four complete Router-delimited Transformer blocks.  The
non-MoE projections remain in every block, while four different routing/token
phases make a single deployable fixed-prefetch pair non-universal.  PIVOT may
adapt only after the current block's Router becomes visible; no request can
prefetch across a Router boundary.
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
from scripts.DATE3.prepare_exp5_multilayer import EXP5_RELATIVE_QUALITY
from scripts.prepare_date2_experiments import skewed, skewed_seed, variant_topology


CONFIG_ROOT = ROOT / "configs/MoE/DATE3"
TOPOLOGY_ROOT = ROOT / "topologies/MoE/DATE3"
OUTPUT_ROOT = ROOT / "outputs/DATE3"
MODELS = ("HMoE", "Mixtral", "MoDSE", "Switchtrans")
EXPERTS_PER_LAYER = 8
BLOCK_COUNT = 4
LAYER_GAP_CYCLES = 64
PROFILE_NAMES = ("balanced_t64", "light40_t64", "high41_t64", "light42_t128")


def frozen_reference() -> tuple[int, int, str]:
    path = OUTPUT_ROOT / "exp5/deployable_selection.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Exp7 requires the independent Exp5 deployable selection"
        )
    import csv
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["policy_name"]: row for row in csv.DictReader(stream)}
    row = rows["Dynamic-FixedPF"]
    return (
        int(row["selected_window"]),
        int(row["selected_chunk_tiles"]),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def positive_phase_counts():
    raw = (
        skewed(64, "balanced"),
        skewed_seed(64, "light", 40),
        skewed_seed(64, "high", 41),
        skewed_seed(128, "light", 42),
    )
    result = []
    for row in raw:
        values = list(row)
        for expert, value in enumerate(values):
            if value > 0:
                continue
            donor = max(range(len(values)), key=values.__getitem__)
            values[donor] -= 1
            values[expert] = 1
        result.append(tuple(values))
    return tuple(result)


def phase_topologies(model: str):
    source = TOPOLOGY_ROOT / "models" / f"{model}.csv"
    root = TOPOLOGY_ROOT / "end_to_end/phases" / model
    paths = tuple(root / f"L{i}_{name}.csv" for i, name in enumerate(PROFILE_NAMES))
    for path, counts in zip(paths, positive_phase_counts()):
        path.parent.mkdir(parents=True, exist_ok=True)
        variant_topology(path, source, counts)
    return paths


def rename_request(value: str, old_expert: int, new_expert: int) -> str:
    return re.sub(
        rf"(?P<prefix>(?:compute|acc)_e){old_expert}(?=_ff[12](?:_|$))",
        rf"\g<prefix>{new_expert}", value,
    )


def combined_payload(model: str, chunk_tiles: int, paths):
    phases = [
        generate_topology_runner_payload(
            path, "heterogeneous", chunk_size_bytes=chunk_tiles * 256,
            top_k=1, num_gpus=2,
        )
        for path in paths
    ]
    value = copy.deepcopy(phases[0])
    for key in ("chunks", "compute_requests", "compiler_bank_plans", "compute_intervals"):
        value[key] = []
    counts, profiles = [], []
    cursor, address_stride = 0, 1 << 24
    for layer_id, (name, path, phase) in enumerate(zip(PROFILE_NAMES, paths, phases)):
        expert_base = layer_id * EXPERTS_PER_LAYER
        layer_counts = list(phase["topology_provenance"]["token_counts"])
        if len(layer_counts) != EXPERTS_PER_LAYER:
            raise ValueError(f"{model}/{name} must contain eight experts")
        counts.extend(layer_counts)
        profiles.append({
            "layer_id": layer_id, "profile": name,
            "token_counts": layer_counts, "source_topology": str(path),
            "start_cycle": cursor,
        })
        for item in phase["chunks"]:
            copied = dict(item)
            old, new = int(copied["expert_id"]), expert_base + int(copied["expert_id"])
            copied.update({
                "chunk_id": f"L{layer_id}__{copied['chunk_id']}",
                "expert_id": new,
                "use_cycle": int(copied["use_cycle"]) + cursor,
                "logical_address": int(copied["logical_address"]) + layer_id * address_stride,
            })
            value["chunks"].append(copied)
        for item in phase["compute_requests"]:
            copied = dict(item)
            request_id = str(copied["request_id"])
            match = re.search(r"(?:compute|acc)_e(\d+)_ff[12](?:_|$)", request_id)
            if match:
                old, new = int(match.group(1)), expert_base + int(match.group(1))
                copied["request_id"] = rename_request(request_id, old, new)
                object_id = re.sub(
                    rf"_e{old}(?=_ff[12](?:_|$))", f"_e{new}",
                    str(copied["object_id"]),
                )
            else:
                copied["request_id"] = f"L{layer_id}__{request_id}"
                object_id = str(copied["object_id"])
            copied.update({
                "object_id": f"L{layer_id}__{object_id}",
                "issue_cycle": int(copied["issue_cycle"]) + cursor,
                "address": int(copied["address"]) + layer_id * address_stride,
            })
            value["compute_requests"].append(copied)
        dimensions = load_moe_topology(path)["experts"]
        for plan in phase["compiler_bank_plans"]:
            copied = dict(plan)
            match = re.fullmatch(r"MoE-E(\d+)-FF([12])", str(copied["layer"]))
            if match is None:
                raise ValueError(f"invalid compiler plan layer: {copied['layer']}")
            old, part = map(int, match.groups())
            m, n, k = dimensions[(old, part)]
            copied.update({
                "layer": f"L{layer_id}__{copied['layer']}",
                "expert_id": expert_base + old, "ffn_part": part,
                "m": m, "n": n, "k": k,
            })
            value["compiler_bank_plans"].append(copied)
        end = cursor + int(phase["compute_cycles"])
        value["compute_intervals"].append([cursor, end])
        cursor = end + LAYER_GAP_CYCLES

    value["compute_cycles"] = cursor - LAYER_GAP_CYCLES
    value["workload_name"] = f"{model}-4Block-EndToEnd-NonStationary"
    provenance = dict(value["topology_provenance"])
    provenance.update({
        "source_path": "multi_layer_exp7_sequence",
        "token_counts": counts, "total_tokens": sum(counts),
        "chunk_size_bytes": chunk_tiles * 256,
        "exp7_multi_layer": True, "layer_count": BLOCK_COUNT,
        "experts_per_layer": EXPERTS_PER_LAYER, "layer_profiles": profiles,
    })
    value["topology_provenance"] = provenance
    value["multi_layer_prefetch"] = {
        "enabled": True, "layer_count": BLOCK_COUNT,
        "experts_per_layer": EXPERTS_PER_LAYER,
        "controller_state": "persistent_across_layers",
        "fixed_knobs": "one_pair_frozen_across_layers",
        "router_boundary_prefetch": "forbidden",
        "layer_gap_cycles": LAYER_GAP_CYCLES,
        "profiles": list(PROFILE_NAMES),
    }
    return value


def finish(model: str, value, window: int, chunk: int, selection_hash: str):
    value = copy.deepcopy(value)
    value.update({
        "experiment_id": f"date3-end_to_end-{model}",
        "date3_suite": "end_to_end", "date3_variant": model,
    })
    value["policy"]["prefetch_window"] = window
    value["policy"]["prefetch_policy"] = "coverage_accuracy_constrained"
    value["coverage_accuracy_policy"] = copy.deepcopy(POLICY)
    value["coverage_accuracy_policy"].update(EXP5_RELATIVE_QUALITY)
    value["coverage_accuracy_policy"].update({
        "reference_window": window, "reference_chunk": chunk,
    })
    value["paper_control_contract"] = {
        "selection_protocol": "independent_calibration_then_freeze",
        "test_trace_visible_during_selection": False,
        "dynamic_fixed_reference": {
            "window": window, "chunk_tiles": chunk,
            "selection_hash": selection_hash,
        },
        "router_boundary_prefetch": "forbidden",
    }
    value["end_to_end_approximation"] = {
        "scope": "four_complete_moe_transformer_blocks",
        "block_count": BLOCK_COUNT,
        "non_moe_layers": [
            "Attn_Q_proj", "Attn_K_proj", "Attn_V_proj", "QKT_head",
            "QKTV_head", "Attn_O_proj", "Router_logits",
        ],
        "non_moe_memory_policy": "Static-5:5:5-SP-plus-15-ACC",
        "moe_system_model": "DATE3_detailed_NPU_plus_peer_EP",
        "composition": "replace_embedded_non_moe_compute_with_full_cycles",
        "ignored_operations": [
            "embedding", "normalization", "softmax", "residual", "sampling",
        ],
    }
    value = attach_ep_contract(value)
    ep = dict(value["ep"])
    ep["expert_owner_map"] = [
        0 if expert % EXPERTS_PER_LAYER < EXPERTS_PER_LAYER // 2 else 1
        for expert in range(len(ep["expert_owner_map"]))
    ]
    value["ep"] = ep
    EPContract.from_payload(value).routes()
    return value


def main() -> None:
    window, chunk, selection_hash = frozen_reference()
    for model in MODELS:
        paths = phase_topologies(model)
        value = finish(
            model, combined_payload(model, chunk, paths),
            window, chunk, selection_hash,
        )
        save(CONFIG_ROOT / "end_to_end" / f"{model}.json", value)
    print(
        f"generated four-block Exp7 configs with frozen reference W={window}, C={chunk}"
    )


if __name__ == "__main__":
    main()
