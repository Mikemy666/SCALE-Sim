"""Compact, lossless Token-route traces for the DATE3 EP model.

The simulator schedules work at expert/FFN-stage granularity.  Consequently,
the trace records exact Token routing and exact modeled stage intervals, while
explicitly marking stage timing as inherited rather than inventing per-Token
pipeline timestamps.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from scalesim.memory.date3_ep_model import DetailedNPUWorkload


TRACE_SCHEMA_VERSION = 1
TIMING_SEMANTICS = "expert_stage_interval_inherited_by_routed_tokens"


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], *, gzip_: bool) -> bool:
    values = list(rows)
    if not values:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_:
        stream = gzip.open(path, mode="wt", newline="", encoding="utf-8")
    else:
        stream = path.open(mode="w", newline="", encoding="utf-8")
    with stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(values[0]))
        writer.writeheader()
        writer.writerows(values)
    return True


def _experts_per_layer(payload: Mapping[str, object], num_experts: int) -> int:
    for section in ("multi_layer_prefetch", "end_to_end_approximation",
                    "topology_provenance"):
        value = payload.get(section, {})
        if isinstance(value, Mapping) and value.get("experts_per_layer"):
            result = int(value["experts_per_layer"])
            if result > 0 and num_experts % result == 0:
                return result
    return num_experts


def build_token_trace(payload: Mapping[str, object], detailed: DetailedNPUWorkload):
    """Build normalized trace tables and machine-checkable invariants."""
    contract = detailed.contract
    routes = tuple(detailed.routes)
    per_layer = _experts_per_layer(payload, contract.num_experts)
    num_layers = contract.num_experts // per_layer

    token_layers: Dict[int, set] = defaultdict(set)
    for route in routes:
        token_layers[route.token_id].add(route.global_expert_id // per_layer)
    layer_tokens: Dict[int, list] = defaultdict(list)
    for token, layers in token_layers.items():
        if len(layers) == 1:
            layer_tokens[next(iter(layers))].append(token)
    layer_token_id = {
        (layer, token): index
        for layer, tokens in layer_tokens.items()
        for index, token in enumerate(sorted(tokens))
    }

    route_rows = []
    for route in routes:
        expert = route.global_expert_id
        layer = expert // per_layer
        local_token = layer_token_id.get((layer, route.token_id), "")
        route_id = f"L{layer}:T{local_token}:K{route.topk_slot}"
        route_rows.append({
            "route_id": route_id,
            "token_id": route.token_id,
            "layer_id": layer,
            "layer_token_id": local_token,
            "topk_slot": route.topk_slot,
            "source_npu": route.source_npu,
            "global_expert_id": expert,
            "layer_expert_id": expert % per_layer,
            "owner_npu": route.owner_npu,
            "destination_offset": route.destination_offset,
            "routing_weight": route.routing_weight,
            "is_remote": int(route.is_remote),
        })

    stage_rows = []
    stages = {}
    for stage in sorted(contract.stages, key=lambda item: (
            item.original_start_cycle, item.expert_id, item.ffn_part)):
        expert = stage.expert_id
        layer = expert // per_layer
        stage_id = f"E{expert}:FF{stage.ffn_part}"
        end = stage.original_start_cycle + stage.compute_cycles
        stages[(expert, stage.ffn_part)] = (stage_id, stage.original_start_cycle, end)
        stage_rows.append({
            "stage_id": stage_id,
            "layer_id": layer,
            "global_expert_id": expert,
            "layer_expert_id": expert % per_layer,
            "owner_npu": contract.owner_by_expert[expert],
            "ffn_part": stage.ffn_part,
            "stage_start_cycle": stage.original_start_cycle,
            "stage_end_cycle": end,
            "compute_cycles": stage.compute_cycles,
            "route_replicas": contract.token_counts[expert],
            "timing_semantics": TIMING_SEMANTICS,
        })

    index_rows = []
    for row in route_rows:
        expert = int(row["global_expert_id"])
        ffn1 = stages.get((expert, 1), ("", "", ""))
        ffn2 = stages.get((expert, 2), ("", "", ""))
        token_key = f"L{row['layer_id']}:T{row['layer_token_id']}"
        index_rows.append({
            **row,
            "dispatch_event_id": f"dispatch:{row['route_id']}",
            "ffn1_stage_id": ffn1[0],
            "ffn1_start_cycle": ffn1[1],
            "ffn1_end_cycle": ffn1[2],
            "ffn2_stage_id": ffn2[0],
            "ffn2_start_cycle": ffn2[1],
            "ffn2_end_cycle": ffn2[2],
            "return_event_id": f"return:{row['route_id']}",
            "combine_event_id": f"combine:{token_key}",
            "timing_semantics": TIMING_SEMANTICS,
        })

    reasons: Dict[int, set] = defaultdict(set)
    if index_rows:
        reasons[int(index_rows[0]["token_id"])].add("first_token")
        reasons[int(index_rows[-1]["token_id"])].add("last_token")
    seen_experts = set()
    seen_remote = set()
    for row in index_rows:
        token = int(row["token_id"])
        expert = int(row["global_expert_id"])
        if expert not in seen_experts:
            reasons[token].add(f"first_for_expert_{expert}")
            seen_experts.add(expert)
        locality = "remote" if int(row["is_remote"]) else "local"
        if locality not in seen_remote:
            reasons[token].add(f"first_{locality}_route")
            seen_remote.add(locality)
    sample_rows = [
        {**row, "sample_reason": "|".join(sorted(reasons[int(row["token_id"])]))}
        for row in index_rows if int(row["token_id"]) in reasons
    ]

    route_counts = Counter(int(row["global_expert_id"]) for row in route_rows)
    token_slots: Dict[int, list] = defaultdict(list)
    token_experts: Dict[int, list] = defaultdict(list)
    owner_offsets: Dict[int, list] = defaultdict(list)
    for row in route_rows:
        token_slots[int(row["token_id"])].append(int(row["topk_slot"]))
        token_experts[int(row["token_id"])].append(int(row["global_expert_id"]))
        owner_offsets[int(row["owner_npu"])].append(int(row["destination_offset"]))
    stage_map = {(item.expert_id, item.ffn_part): item for item in contract.stages}
    checks = {
        "route_replica_count_matches_contract": len(routes) == sum(contract.token_counts),
        "unique_token_topk_keys": len({(r.token_id, r.topk_slot) for r in routes}) == len(routes),
        "every_token_has_exact_topk_slots": all(
            sorted(values) == list(range(contract.top_k)) for values in token_slots.values()),
        "topk_experts_are_distinct": all(
            len(values) == len(set(values)) for values in token_experts.values()),
        "expert_counts_match_contract": all(
            route_counts[expert] == count
            for expert, count in enumerate(contract.token_counts)),
        "expert_owner_matches_contract": all(
            route.owner_npu == contract.owner_by_expert[route.global_expert_id]
            for route in routes),
        "destination_offsets_contiguous_per_owner": all(
            sorted(values) == list(range(len(values))) for values in owner_offsets.values()),
        "all_routed_experts_have_ffn1_and_ffn2": all(
            (expert, 1) in stage_map and (expert, 2) in stage_map for expert in route_counts),
        "ffn_stage_intervals_are_positive": all(
            stage.compute_cycles > 0 for stage in contract.stages),
        "ffn1_precedes_ffn2_per_expert": all(
            (expert, 1) not in stage_map or (expert, 2) not in stage_map
            or stage_map[(expert, 1)].original_start_cycle
            <= stage_map[(expert, 2)].original_start_cycle
            for expert in route_counts),
        "topk_replicas_stay_in_one_layer": all(
            len(layers) == 1 for layers in token_layers.values()),
    }
    summary = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_semantics": {
            "routing": "exact deterministic EP route replicas used by the simulator",
            "timing": TIMING_SEMANTICS,
            "caveat": (
                "stage intervals are exact model stages shared by routed tokens; "
                "they are not fabricated per-token pipeline timestamps"
            ),
        },
        "dimensions": {
            "layers": num_layers,
            "experts_per_layer": per_layer,
            "global_experts": contract.num_experts,
            "npus": contract.num_npus,
            "top_k": contract.top_k,
        },
        "counts": {
            "tokens": len(token_slots),
            "route_replicas": len(route_rows),
            "remote_route_replicas": sum(int(row["is_remote"]) for row in route_rows),
            "expert_stages": len(stage_rows),
            "sample_tokens": len(reasons),
            "sample_route_replicas": len(sample_rows),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    return route_rows, stage_rows, index_rows, sample_rows, summary


def export_token_trace(output_dir: Path, payload: Mapping[str, object],
                       detailed: DetailedNPUWorkload, mode: str = "full") -> Mapping[str, object]:
    """Export summary, deterministic sample, and optionally full gzip tables."""
    if mode not in {"none", "summary", "sampled", "full"}:
        raise ValueError(f"unsupported Token trace mode: {mode}")
    if mode == "none":
        return {"mode": mode, "files": []}
    output_dir = Path(output_dir)
    routes, stages, index, sample, summary = build_token_trace(payload, detailed)
    files = []
    if mode in {"sampled", "full"}:
        if _write_csv(output_dir / "TOKEN_STAGE_TRACE.csv.gz", stages, gzip_=True):
            files.append("TOKEN_STAGE_TRACE.csv.gz")
        if _write_csv(output_dir / "TOKEN_TRACE_SAMPLE.csv", sample, gzip_=False):
            files.append("TOKEN_TRACE_SAMPLE.csv")
    if mode == "full":
        if _write_csv(output_dir / "TOKEN_ROUTE_TRACE.csv.gz", routes, gzip_=True):
            files.append("TOKEN_ROUTE_TRACE.csv.gz")
        if _write_csv(output_dir / "TOKEN_TRACE_INDEX.csv.gz", index, gzip_=True):
            files.append("TOKEN_TRACE_INDEX.csv.gz")
    summary["mode"] = mode
    summary["files"] = files + ["TOKEN_TRACE_SUMMARY.json"]
    (output_dir / "TOKEN_TRACE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
