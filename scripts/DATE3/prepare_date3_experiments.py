"""Generate DATE3 configs/topologies without modifying DATE2 artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.date3_ep_model import attach_ep_contract
DATE2_CONFIG = ROOT / "configs/MoE/DATE2"
DATE3_CONFIG = ROOT / "configs/MoE/DATE3"
DATE2_TOPOLOGY = ROOT / "topologies/MoE/DATE2"
DATE3_TOPOLOGY = ROOT / "topologies/MoE/DATE3"
MODELS = ("HMoE", "Mixtral", "MoDSE", "Switchtrans")

POLICY = {
    "enabled": True,
    "reference_mode": "shadow_fixed",
    "reference_chunk": 4,
    "reference_window": 8,
    "eta_coverage": 0.25,
    "eta_accuracy": 0.25,
    "min_coverage": 0.95,
    "min_accuracy": 0.95,
    "epsilon_coverage": 0.05,
    "epsilon_accuracy": 0.05,
    "ema_warmup_epochs": 2,
    "candidate_chunks": [1, 2, 4, 8],
    "candidate_windows": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "bank_candidate_count": 4,
    "max_residency_ratio": 0.85,
    "pressure_threshold": 0.90,
    "adaptation_epoch": 1,
    "adaptation_cooldown": 1,
    "score_hysteresis": 0.02,
    "max_chunk_step": 1,
    "max_window_step": 4,
    "base_safety_margin": 2.0,
    "timing_margin_scale": 0.25,
    "minimum_positive_score": -1.0,
    "severe_late_ratio": 0.10,
    "online_incumbent_guard": True,
    "pressure_mode": "mean_max",
    "pressure_weights": {
        "queue": 0.30, "busy": 0.25,
        "conflict": 0.25, "residency": 0.20,
    },
    "score_weights": {
        "latency": 1.0, "occupancy": 0.20, "pressure": 0.20,
        "conflict": 0.15, "mapping": 0.05,
    },
}


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def date3_payload(source: Path, suite: str, variant: str, overrides=None):
    value = json.loads(source.read_text(encoding="utf-8"))
    value["experiment_id"] = f"date3-{suite}-{variant}"
    value["date3_suite"] = suite
    value["date3_variant"] = variant
    value["policy"]["prefetch_policy"] = "coverage_accuracy_constrained"
    value["coverage_accuracy_policy"] = copy.deepcopy(POLICY)
    provenance = value.get("topology_provenance", {})
    if "source_path" in provenance:
        provenance["source_path"] = str(provenance["source_path"]).replace(
            "/topologies/MoE/DATE2/", "/topologies/MoE/DATE3/"
        )
    value["paper_control_contract"] = {
        "implementable": [
            "Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF",
            "Static-Opt-FixedPF", "Dynamic-FixedPF", "PIVOT",
        ],
        "reference_only": ["Ideal-NoPF"],
        "fixed_prefetch_reference": {
            "window": POLICY["reference_window"],
            "chunk_tiles": POLICY["reference_chunk"],
            "same_workload_for_static_and_dynamic": True,
        },
    }
    if overrides:
        value["coverage_accuracy_policy"].update(overrides)
    return attach_ep_contract(value)


def main() -> None:
    for model in MODELS:
        target = DATE3_TOPOLOGY / "models" / f"{model}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATE2_TOPOLOGY / "models" / f"{model}.csv", target)
        payload = date3_payload(
            DATE2_CONFIG / "overall" / f"{model}.json", "overall", model
        )
        save(DATE3_CONFIG / "overall" / f"{model}.json", payload)
        # Overall topology copies are provenance anchors; configs continue to
        # embed the generated request stream exactly as DATE2 does.
        overall_topology = DATE3_TOPOLOGY / "overall" / f"{model}.csv"
        overall_topology.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, overall_topology)

        # Exp7 is one complete MoE Transformer block per model: Attention,
        # Router, and routed expert FF1/FF2.  Non-MoE memory is conservatively
        # held at the fixed 5:5:5 baseline; DATE3 policies optimize MoE+EP.
        end_to_end = date3_payload(
            DATE2_CONFIG / "overall" / f"{model}.json",
            "end_to_end", model,
        )
        end_to_end["end_to_end_approximation"] = {
            "scope": "one_complete_moe_transformer_block",
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
        save(DATE3_CONFIG / "end_to_end" / f"{model}.json", end_to_end)
        end_to_end_topology = DATE3_TOPOLOGY / "end_to_end" / f"{model}.csv"
        end_to_end_topology.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, end_to_end_topology)

    ablations = {
        "without_quality": {"min_coverage": 0.0, "min_accuracy": 0.0,
                            "epsilon_coverage": 1.0, "epsilon_accuracy": 1.0},
        "coverage_only": {"min_accuracy": 0.0, "epsilon_accuracy": 1.0},
        "accuracy_only": {"min_coverage": 0.0, "epsilon_coverage": 1.0},
        "both_constraints": {},
        "without_bank_pressure": {"pressure_threshold": 1.0,
                                  "score_weights": {**POLICY["score_weights"], "pressure": 0.0}},
        "without_dynamic_chunk": {"candidate_chunks": [POLICY["reference_chunk"]]},
        "without_dynamic_window": {"candidate_windows": [POLICY["reference_window"]]},
        "full": {},
    }
    source = DATE2_CONFIG / "overall" / "MoDSE.json"
    for name, overrides in ablations.items():
        save(DATE3_CONFIG / "ablation" / f"MoDSE__{name}.json",
             date3_payload(source, "ablation", f"MoDSE__{name}", overrides))
    ab_topology = DATE3_TOPOLOGY / "ablation" / "MoDSE.csv"
    ab_topology.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATE3_TOPOLOGY / "models/MoDSE.csv", ab_topology)

    # Exp1/Exp2 are P=1 microarchitecture characterization experiments.  This
    # deliberately preserves DATE2's per-layer figure/data meaning while the
    # P=2 end-to-end experiments below use the full EP envelope.
    micro_hardware = {
        "bank_count": 30, "bank_width_bits": 128, "bank_entries": 128,
        "ports_per_bank": 1, "bandwidth_bytes_per_cycle": 480,
        "request_buffer_depth": 32,
    }
    precision = {
        "model": "FP32", "ia": "INT8", "weight": "INT8",
        "accumulator": "INT32", "oa": "INT8",
    }
    save(DATE3_CONFIG / "exp1/exp1.json", {
        "experiment_id": "date3-exp1-characterization",
        "purpose": "fixed_SP_ACC_boundary_and_layer_bottleneck",
        "ep_degree": 1,
        "topology": "topologies/MoE/DATE3/models/MoDSE.csv",
        "static_sp_banks": [5, 5, 5], "static_acc_banks": 15,
        "hardware": micro_hardware, "precision": precision,
    })
    save(DATE3_CONFIG / "exp2/exp2.json", {
        "experiment_id": "date3-exp2-static_ownership_sweep",
        "purpose": "exhaustive_static_SP_ownership",
        "ep_degree": 1,
        "topology": "topologies/MoE/DATE3/models/MoDSE.csv",
        "sp_bank_total": 15, "acc_bank_total": 15,
        "positive_partitions": 91,
        "hardware": micro_hardware, "precision": precision,
    })
    save(DATE3_CONFIG / "exp3/exp3.json", {
        "experiment_id": "date3-exp3-naive_prefetch_pathology",
        "purpose": "fixed_prefetch_timeliness_interference",
        "source_suite": "configs/MoE/DATE3/window_chunk",
        "windows": list((0, 1, 2, 4, 8, 16, 32, 64)),
        "chunk_tiles": list((1, 2, 4, 8)),
    })

    for source_config in sorted((DATE2_CONFIG / "window_chunk").glob("*.json")):
        value = json.loads(source_config.read_text(encoding="utf-8"))
        value["experiment_id"] = f"date3-window-chunk-{source_config.stem}"
        value["date3_suite"] = "window_chunk"
        value["policy"]["prefetch_policy"] = "naive_fixed"
        value = attach_ep_contract(value)
        save(DATE3_CONFIG / "window_chunk" / source_config.name, value)

    # Exp5 preserves the same 8x4 configured Window/Chunk grid, but each point
        # is now a complete PIVOT run whose fixed reference is seeded by that
    # grid point.  Candidate sets remain global so runtime adaptation is real.
    for source_config in sorted((DATE2_CONFIG / "joint_prefetch").glob("*.json")):
        match = re.fullmatch(r"w(\d+)_c(\d+)", source_config.stem)
        if match is None:
            raise ValueError(f"invalid DATE2 joint-prefetch name: {source_config.name}")
        window, chunk = map(int, match.groups())
        payload = date3_payload(
            source_config, "joint_prefetch", source_config.stem,
            {"reference_window": window, "reference_chunk": chunk},
        )
        payload["sweep"] = {
            "variable": "window_chunk_seed", "window": window,
            "chunk_tiles": chunk, "model": "MoDSE",
        }
        save(DATE3_CONFIG / "joint_prefetch" / source_config.name, payload)
    joint_topology = DATE3_TOPOLOGY / "joint_prefetch/MoDSE.csv"
    joint_topology.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATE3_TOPOLOGY / "models/MoDSE.csv", joint_topology)

    sensitivity = {
        "epsilon_coverage": (0.0, 0.05, 0.10),
        "epsilon_accuracy": (0.0, 0.05, 0.10),
        "eta_coverage": (0.10, 0.25, 0.50),
        "eta_accuracy": (0.10, 0.25, 0.50),
        "max_residency_ratio": (0.60, 0.75, 0.90),
        "bank_candidate_count": (1, 2, 4, 8),
    }
    for variable, values in sensitivity.items():
        for value in values:
            name = f"{variable}__{value}"
            payload = date3_payload(source, "quality_sensitivity", name,
                                    {variable: value})
            payload["sweep"] = {"variable": variable, "value": value}
            save(DATE3_CONFIG / "quality_sensitivity" / f"{name}.json", payload)
    quality_topology = DATE3_TOPOLOGY / "quality_sensitivity" / "MoDSE.csv"
    quality_topology.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATE3_TOPOLOGY / "models/MoDSE.csv", quality_topology)

    for source_config in sorted((DATE2_CONFIG / "robustness_factorial").glob("*.json")):
        payload = date3_payload(source_config, "robustness_factorial",
                                source_config.stem)
        save(DATE3_CONFIG / "robustness_factorial" / source_config.name, payload)
    for source_topology in (DATE2_TOPOLOGY / "robustness_factorial").rglob("*.csv"):
        target = DATE3_TOPOLOGY / "robustness_factorial" / source_topology.relative_to(
            DATE2_TOPOLOGY / "robustness_factorial")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_topology, target)

    # Deterministic low-cost integration case: it is deliberately separate
    # from all paper suites and includes one speculative and one evicted Tile.
    unit = date3_payload(source, "unit_cases", "MoDSE_minimal")
    # Retain a small but complete global expert set.  Truncating the flat list
    # would leave Peer owners without weight Tiles and would make an EP unit
    # case look valid while silently assigning zero owner-local HBM traffic.
    kept_per_stage = {}
    minimal_chunks = []
    for chunk in unit["chunks"]:
        identity = (int(chunk["expert_id"]), int(chunk["ffn_part"]))
        # E0 is the detailed-policy adaptation fixture; it needs more than the
        # fixed reference Chunk=4 so a legal Chunk=8 decision is observable.
        stage_limit = 12 if identity[0] == 0 else 4
        if kept_per_stage.get(identity, 0) >= stage_limit:
            continue
        minimal_chunks.append(chunk)
        kept_per_stage[identity] = kept_per_stage.get(identity, 0) + 1
    unit["chunks"] = minimal_chunks
    speculative = copy.deepcopy(next(
        item for item in unit["chunks"] if int(item["expert_id"]) == 0
    ))
    speculative["chunk_id"] = "date3_speculative_unused"
    speculative["tile_id"] = 99999
    speculative["logical_address"] += 16 * 1024 * 1024
    speculative["use_cycle"] += 256
    unit["chunks"].append(speculative)
    unit["coverage_accuracy_policy"].update({
        "ema_warmup_epochs": 1,
        "adaptation_cooldown": 0,
        "score_hysteresis": 0.0,
        "speculative_chunk_ids": [speculative["chunk_id"]],
        "evict_before_use_chunk_ids": [unit["chunks"][18]["chunk_id"]],
        "forced_late_chunk_ids": [unit["chunks"][17]["chunk_id"]],
    })
    save(DATE3_CONFIG / "unit_cases/MoDSE_minimal.json", unit)
    unit_topology = DATE3_TOPOLOGY / "unit_cases/MoDSE_minimal.csv"
    unit_topology.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATE3_TOPOLOGY / "models/MoDSE.csv", unit_topology)

    architecture = {
        "name": "PIVOT", "architecture": "MemDomain",
        "inherits": "DATE2 unified Bank domain",
        "policy": POLICY,
        "expert_parallel": {
            "default_num_npus": 2,
            "detailed_npu_model": "cycle_level_unified_bank",
            "peer_npu_model": "analytical_owner_local",
            "route_granularity": "token_topk_replica",
            "system_completion": "max_detailed_peer_then_combine",
        },
        "cycle_contract": [
            "compute_cycles", "bank_stall_cycles", "weight_load_stall_cycles",
            "prefetch_miss_stall_cycles", "prefetch_interference_stall_cycles",
            "mapping_overhead_cycles", "communication_stall_cycles", "other_stall_cycles",
        ],
        "paper_control_contract": {
            "mapping_only": [
                "Static-555-NoPF", "Static-Opt-NoPF",
                "Dynamic-NoPF", "Ideal-NoPF",
            ],
            "prefetch_enabled": [
                "Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF",
                "Static-Opt-FixedPF", "Dynamic-FixedPF", "PIVOT",
            ],
            "ideal_is_reference_only": True,
        },
    }
    save(DATE3_CONFIG / "architecture.json", architecture)
    suites = {
        name: len(list((DATE3_CONFIG / name).glob("*.json")))
        for name in ("overall", "end_to_end", "ablation", "window_chunk", "joint_prefetch",
                     "quality_sensitivity", "robustness_factorial", "unit_cases")
    }
    manifest = {
        "name": "PIVOT Coverage/Accuracy-Constrained Prefetch",
        "suites": suites,
        "policy_name": "PIVOT",
        "ep_schema_version": 1,
        "online_incumbent_guard": True,
        "date2_modified": False,
        "paper_experiments": {
            "exp1": "layer_characterization",
            "exp2": "static_bank_sweep",
            "exp3": "naive_prefetch_interference",
            "exp4": "dynamic_mapping_no_prefetch",
            "exp5": "joint_prefetch",
            "exp6": "robustness_factorial",
            "exp7": "end_to_end",
        },
    }
    save(DATE3_CONFIG / "manifest.json", manifest)
    print(f"Generated DATE3 suites: {suites}")


if __name__ == "__main__":
    main()
