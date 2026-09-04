#!/usr/bin/env python3
"""Read-only audit helpers for the parameterized MoE Expert-Parallel contract.

This module deliberately does not patch or wrap the simulator.  The emitted
traces are small deterministic *contract-reference* traces used to compare the
two existing execution paths (``scalesim.simulator`` and DATE3/PIVOT-CA).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "topologies/MoE/DATE3/models"
CONFIG_ROOT = ROOT / "configs/MoE/DATE3/overall"
OUTPUT_ROOT = ROOT / "outputs/DATE3/validation"
EXPERT_ROW = re.compile(r"MoE-E(\d+)-FF([12])$")
MODELS = ("HMoE", "Mixtral", "MoDSE", "Switchtrans")


def balanced_owner_map(num_experts: int, num_npus: int) -> dict[int, int]:
    """Reference contiguous mapping with counts differing by at most one."""
    if num_experts <= 0 or num_npus <= 0:
        raise ValueError("expert and NPU counts must be positive")
    quotient, remainder = divmod(num_experts, num_npus)
    owners: dict[int, int] = {}
    expert = 0
    for npu in range(num_npus):
        for _ in range(quotient + int(npu < remainder)):
            owners[expert] = npu
            expert += 1
    return owners


def scale_sim_uniform_owner_map(num_experts: int, num_npus: int) -> dict[int, int]:
    """Ownership representable by current NumGPUs x ExpertsPerGPU config."""
    if num_experts % num_npus:
        raise ValueError(
            "current scale_config derives NumExperts=NumGPUs*ExpertsPerGPU; "
            "a non-divisible global expert count has no representation"
        )
    per_npu = num_experts // num_npus
    return {expert: expert // per_npu for expert in range(num_experts)}


def validate_ownership(owners: Mapping[int, int], num_experts: int,
                       num_npus: int) -> None:
    if set(owners) != set(range(num_experts)):
        raise AssertionError("expert union is incomplete or contains duplicates")
    if any(owner < 0 or owner >= num_npus for owner in owners.values()):
        raise AssertionError("owner is outside the configured NPU range")


def load_topology(path: Path) -> dict[str, object]:
    experts: dict[int, dict[int, tuple[int, int, int]]] = {}
    router_n = None
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or not row[0] or row[0] == "Layer":
                continue
            name = row[0].strip()
            m, n, k = map(int, row[1:4])
            if name == "Router_logits":
                router_n = n
            match = EXPERT_ROW.fullmatch(name)
            if match:
                expert, part = map(int, match.groups())
                if part in experts.setdefault(expert, {}):
                    raise AssertionError(f"duplicate {name}")
                experts[expert][part] = (m, n, k)
    ids = sorted(experts)
    if ids != list(range(len(ids))):
        raise AssertionError(f"{path}: expert IDs are not contiguous")
    for expert in ids:
        if set(experts[expert]) != {1, 2}:
            raise AssertionError(f"{path}: expert {expert} lacks FF1 or FF2")
    return {"experts": experts, "router_n": router_n}


def expert_metrics(stages: Mapping[int, tuple[int, int, int]],
                   weight_bytes_per_element: int = 1) -> dict[str, int]:
    ff1, ff2 = stages[1], stages[2]
    param_elements = ff1[1] * ff1[2] + ff2[1] * ff2[2]
    macs = ff1[0] * ff1[1] * ff1[2] + ff2[0] * ff2[1] * ff2[2]
    return {
        "parameter_elements": param_elements,
        "parameter_bytes": param_elements * weight_bytes_per_element,
        "runtime_macs": macs,
    }


def validate_routes(routes: Sequence[Sequence[int]], num_experts: int,
                    top_k: int) -> None:
    if not 1 <= top_k <= num_experts:
        raise AssertionError("invalid Top-k")
    for selected in routes:
        if len(selected) != top_k or len(set(selected)) != top_k:
            raise AssertionError("each token must have Top-k distinct replicas")
        if any(expert not in range(num_experts) for expert in selected):
            raise AssertionError("route is not in the global expert set")


def communication_bytes(routes: Sequence[Sequence[int]], owners: Mapping[int, int],
                        source_npu: int, hidden_dim: int,
                        bytes_per_element: int = 1) -> int:
    remote_replicas = sum(
        owners[expert] != source_npu for selected in routes for expert in selected
    )
    return remote_replicas * hidden_dim * bytes_per_element


TRACE_FIELDS = (
    "cycle", "token_id", "source_npu", "selected_expert", "owner_npu",
    "is_remote", "routing_weight", "dispatch_bytes", "receive_npu",
    "expert_token_count", "ffn_stage", "gemm_m", "gemm_k", "gemm_n",
    "weight_owner_npu", "weight_chunk_id", "weight_bytes",
    "target_bank_group", "prefetch_issue_cycle", "prefetch_complete_cycle",
    "first_use_cycle", "release_cycle", "result_return_npu",
    "combine_complete",
)


def build_reference_trace(routes: Sequence[Sequence[int]], *, num_experts: int,
                          num_npus: int, source_npu: int = 0,
                          hidden_dim: int = 96,
                          expert_hidden_dim: int = 384,
                          activation_bytes: int = 1,
                          weight_bytes_per_element: int = 1) -> list[dict[str, object]]:
    """Build an auditable event trace, not a replacement simulator result."""
    top_k = len(routes[0]) if routes else 1
    validate_routes(routes, num_experts, top_k)
    owners = scale_sim_uniform_owner_map(num_experts, num_npus)
    counts = Counter(expert for selected in routes for expert in selected)
    rows: list[dict[str, object]] = []
    for token_id, selected in enumerate(routes):
        for slot, expert in enumerate(selected):
            owner = owners[expert]
            remote = owner != source_npu
            dispatch = hidden_dim * activation_bytes if remote else 0
            arrival = 20 + math.ceil(dispatch / 128) if remote else 0
            prior_finish = arrival
            for stage, (gemm_k, gemm_n) in enumerate(
                ((hidden_dim, expert_hidden_dim),
                 (expert_hidden_dim, hidden_dim)), start=1
            ):
                weight_bytes = gemm_k * gemm_n * weight_bytes_per_element
                chunk_bytes = min(2048, weight_bytes)
                issue = prior_finish
                complete = issue + 20 + math.ceil(chunk_bytes / 128)
                first_use = complete
                compute = max(1, math.ceil(counts[expert] * gemm_k * gemm_n / 256))
                release = first_use + compute
                return_cycles = 20 + math.ceil(hidden_dim * activation_bytes / 128) if remote else 0
                combine = release + return_cycles if stage == 2 else ""
                rows.append({
                    "cycle": issue,
                    "token_id": token_id,
                    "source_npu": source_npu,
                    "selected_expert": expert,
                    "owner_npu": owner,
                    "is_remote": int(remote),
                    "routing_weight": round(1.0 / top_k, 6),
                    "dispatch_bytes": dispatch,
                    "receive_npu": owner,
                    "expert_token_count": counts[expert],
                    "ffn_stage": f"FF{stage}",
                    "gemm_m": counts[expert],
                    "gemm_k": gemm_k,
                    "gemm_n": gemm_n,
                    "weight_owner_npu": owner,
                    "weight_chunk_id": f"npu{owner}_e{expert}_ff{stage}_c0",
                    "weight_bytes": weight_bytes,
                    "target_bank_group": f"NPU{owner}:local-bank-group-0",
                    "prefetch_issue_cycle": issue,
                    "prefetch_complete_cycle": complete,
                    "first_use_cycle": first_use,
                    "release_cycle": release,
                    "result_return_npu": source_npu if stage == 2 else "",
                    "combine_complete": combine,
                })
                prior_finish = release
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def repository_snapshot() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    return {
        "repository": str(ROOT),
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "working_tree_clean": not bool(git("status", "--short")),
    }


def audit_models() -> dict[str, object]:
    result: dict[str, object] = {}
    for model in MODELS:
        topology = load_topology(MODEL_ROOT / f"{model}.csv")
        experts = topology["experts"]
        config = json.loads((CONFIG_ROOT / f"{model}.json").read_text(encoding="utf-8"))
        owners = scale_sim_uniform_owner_map(len(experts), 2)
        per_expert = []
        parameter_load = [0, 0]
        compute_load = [0, 0]
        for expert, stages in experts.items():
            metrics = expert_metrics(stages)
            owner = owners[expert]
            parameter_load[owner] += metrics["parameter_bytes"]
            compute_load[owner] += metrics["runtime_macs"]
            per_expert.append({
                "expert_id": expert, "owner_npu": owner,
                "ff1": stages[1], "ff2": stages[2], **metrics,
            })
        result[model] = {
            "global_experts": len(experts),
            "router_output_dimension": topology["router_n"],
            "formal_num_gpus": config["system"]["num_gpus"],
            "formal_top_k": config["topology_provenance"]["top_k"],
            "formal_total_tokens": config["topology_provenance"]["total_tokens"],
            "chunk_size_bytes": config["topology_provenance"]["chunk_size_bytes"],
            "parameter_load_by_reference_owner": parameter_load,
            "compute_load_by_reference_owner": compute_load,
            "experts": per_expert,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    top1 = [[0], [4], [1], [5], [2], [6], [3], [7]]
    top2 = [[0, 4], [1, 5], [2, 6], [3, 7]]
    write_csv(args.output_dir / "ep_model_audit_trace.csv",
              build_reference_trace(top1, num_experts=8, num_npus=2))
    write_csv(args.output_dir / "ep_model_audit_top2_trace.csv",
              build_reference_trace(top2, num_experts=8, num_npus=2))
    summary = {
        "trace_kind": "deterministic_contract_reference_not_simulator_output",
        "repository": repository_snapshot(),
        "models": audit_models(),
        "parameterized_contract": {
            "E4_P1": balanced_owner_map(4, 1),
            "E8_P2": balanced_owner_map(8, 2),
            "E16_P2": balanced_owner_map(16, 2),
            "E10_P3_extension": balanced_owner_map(10, 3),
        },
        "known_date3_pivot_gaps": [
            "Peer NPUs use an analytical owner-local model rather than a second cycle-accurate Bank simulator",
            "communication uses startup/bandwidth transactions without packet-level NoC contention",
            "Top-k Combine preserves routes and weights but does not simulate numeric tensor values",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ep_model_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "audit artifacts generated",
        "top1_rows": len(build_reference_trace(top1, num_experts=8, num_npus=2)),
        "top2_rows": len(build_reference_trace(top2, num_experts=8, num_npus=2)),
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
