"""DATE3 Expert-Parallel contract, deterministic routing, and NPU localization.

The module separates the global MoE data flow from the PIVOT-CA local memory
controller.  It intentionally models only the detailed NPU here; analytical
Peer timing and Return/Combine are added by the later system-timeline stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence, Tuple

from scalesim.memory.memdomain_runner import RunnerConfig


REQUEST_STAGE = re.compile(r"(?:compute|acc)_e(\d+)_ff([12])(?:_|$)")


def balanced_owner_map(num_experts: int, num_npus: int) -> Tuple[int, ...]:
    """Contiguous owner map with no omission and at most one-count skew."""
    if num_experts <= 0 or num_npus <= 0:
        raise ValueError("EP expert and NPU counts must be positive")
    quotient, remainder = divmod(num_experts, num_npus)
    owners = []
    for npu in range(num_npus):
        owners.extend([npu] * (quotient + int(npu < remainder)))
    if len(owners) != num_experts:
        raise AssertionError("internal EP owner construction error")
    return tuple(owners)


def deterministic_routes_from_counts(
    counts: Sequence[int], top_k: int
) -> Tuple[Tuple[int, ...], ...]:
    """Construct exact global Top-k assignments from controlled expert counts."""
    if not counts or any(int(value) < 0 for value in counts):
        raise ValueError("expert route counts must be non-negative")
    if not 1 <= int(top_k) <= len(counts):
        raise ValueError("Top-k must be within the global expert set")
    remaining = {expert: int(count) for expert, count in enumerate(counts)}
    replicas = sum(remaining.values())
    if replicas % top_k:
        raise ValueError("route replicas must be divisible by Top-k")
    assignments = []
    for _ in range(replicas // top_k):
        available = sorted(
            (expert for expert, count in remaining.items() if count > 0),
            key=lambda expert: (-remaining[expert], expert),
        )
        if len(available) < top_k:
            raise ValueError("counts cannot form distinct global Top-k routes")
        selected = tuple(available[:top_k])
        for expert in selected:
            remaining[expert] -= 1
        assignments.append(selected)
    if any(remaining.values()):
        raise AssertionError("route count construction did not close")
    return tuple(assignments)


@dataclass(frozen=True)
class RouteReplica:
    token_id: int
    source_npu: int
    global_expert_id: int
    owner_npu: int
    routing_weight: float
    topk_slot: int
    destination_offset: int
    is_remote: bool

    def to_dict(self) -> Mapping[str, object]:
        return {
            "token_id": self.token_id,
            "source_npu": self.source_npu,
            "global_expert_id": self.global_expert_id,
            "owner_npu": self.owner_npu,
            "routing_weight": self.routing_weight,
            "topk_slot": self.topk_slot,
            "destination_offset": self.destination_offset,
            "is_remote": self.is_remote,
        }


@dataclass(frozen=True)
class ExpertStage:
    expert_id: int
    ffn_part: int
    original_start_cycle: int
    compute_cycles: int


@dataclass(frozen=True)
class EPContract:
    num_experts: int
    num_npus: int
    detailed_npu_id: int
    top_k: int
    owner_by_expert: Tuple[int, ...]
    token_counts: Tuple[int, ...]
    stages: Tuple[ExpertStage, ...]
    source_distribution: str = "all_on_detailed"

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EPContract":
        ep = payload.get("ep")
        if not isinstance(ep, Mapping):
            raise ValueError("DATE3 configuration requires an ep contract")
        num_experts = int(ep["num_experts"])
        num_npus = int(ep["num_npus"])
        detailed = int(ep.get("detailed_npu_id", 0))
        top_k = int(ep["top_k"])
        owners = tuple(int(value) for value in ep["expert_owner_map"])
        counts = tuple(int(value) for value in ep["token_counts"])
        stages = tuple(ExpertStage(
            int(item["expert_id"]), int(item["ffn_part"]),
            int(item["original_start_cycle"]), int(item["compute_cycles"]),
        ) for item in ep.get("expert_stages", ()))
        value = cls(
            num_experts, num_npus, detailed, top_k, owners, counts, stages,
            str(ep.get("source_token_distribution", "all_on_detailed")),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.num_experts <= 0 or self.num_npus <= 0:
            raise ValueError("invalid DATE3 EP dimensions")
        if not 0 <= self.detailed_npu_id < self.num_npus:
            raise ValueError("detailed NPU is outside the EP degree")
        if len(self.owner_by_expert) != self.num_experts:
            raise ValueError("every global expert must have exactly one owner")
        if any(not 0 <= owner < self.num_npus for owner in self.owner_by_expert):
            raise ValueError("expert owner is outside the EP degree")
        if len(self.token_counts) != self.num_experts:
            raise ValueError("token counts must cover every global expert")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("invalid global Top-k")
        if sum(self.token_counts) % self.top_k:
            raise ValueError("route replicas are not divisible by Top-k")
        identities = [(stage.expert_id, stage.ffn_part) for stage in self.stages]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate expert FFN stage metadata")
        if any(expert not in range(self.num_experts) or part not in (1, 2)
               for expert, part in identities):
            raise ValueError("invalid expert FFN stage identity")
        if self.source_distribution not in {"all_on_detailed", "round_robin"}:
            raise ValueError("unsupported Token source distribution")

    def source_for_token(self, token_id: int) -> int:
        if self.source_distribution == "round_robin":
            return token_id % self.num_npus
        return self.detailed_npu_id

    def routes(self) -> Tuple[RouteReplica, ...]:
        assignments = deterministic_routes_from_counts(self.token_counts, self.top_k)
        destination_counts = [0] * self.num_npus
        rows = []
        for token_id, selected in enumerate(assignments):
            source = self.source_for_token(token_id)
            for slot, expert in enumerate(selected):
                owner = self.owner_by_expert[expert]
                offset = destination_counts[owner]
                destination_counts[owner] += 1
                rows.append(RouteReplica(
                    token_id, source, expert, owner, 1.0 / self.top_k,
                    slot, offset, owner != source,
                ))
        return tuple(rows)


@dataclass(frozen=True)
class DetailedNPUWorkload:
    config: RunnerConfig
    contract: EPContract
    routes: Tuple[RouteReplica, ...]
    local_experts: Tuple[int, ...]
    active_local_experts: Tuple[int, ...]
    local_route_replicas: int
    remote_route_replicas: int

    def summary_row(self) -> Mapping[str, object]:
        return {
            "npu_id": self.contract.detailed_npu_id,
            "num_npus": self.contract.num_npus,
            "global_expert_count": self.contract.num_experts,
            "local_experts": "|".join(map(str, self.local_experts)),
            "active_local_experts": "|".join(map(str, self.active_local_experts)),
            "local_expert_count": len(self.local_experts),
            "active_local_expert_count": len(self.active_local_experts),
            "local_route_replicas": self.local_route_replicas,
            "remote_route_replicas": self.remote_route_replicas,
            "local_chunk_count": len(self.config.chunks),
            "local_compute_request_count": len(self.config.compute_requests),
            "local_compute_cycles": self.config.compute_cycles,
        }


def _request_stage(request) -> Tuple[int, int] | None:
    match = REQUEST_STAGE.search(str(request.request_id))
    return tuple(map(int, match.groups())) if match else None


def localize_detailed_npu(config: RunnerConfig) -> DetailedNPUWorkload:
    """Filter and compact DATE3 work onto the detailed NPU's local timeline."""
    contract = EPContract.from_payload(config.payload)
    routes = contract.routes()
    detailed = contract.detailed_npu_id
    local_experts = tuple(
        expert for expert, owner in enumerate(contract.owner_by_expert)
        if owner == detailed
    )
    routed_counts = [0] * contract.num_experts
    for route in routes:
        routed_counts[route.global_expert_id] += 1
    active = tuple(expert for expert in local_experts if routed_counts[expert] > 0)

    stages = {(item.expert_id, item.ffn_part): item for item in contract.stages}
    present = sorted({
        (chunk.expert_id, chunk.ffn_part) for chunk in config.chunks
        if chunk.expert_id in active
    })
    missing = [identity for identity in present if identity not in stages]
    if missing:
        raise ValueError(f"DATE3 EP stage metadata missing {missing}")

    first_stage = min(
        (stage.original_start_cycle for stage in contract.stages),
        default=config.compute_cycles,
    )
    cursor = first_stage
    rebased = {}
    for identity in present:
        stage = stages[identity]
        rebased[identity] = cursor
        cursor += stage.compute_cycles
    local_compute_cycles = cursor + 32

    local_chunks = []
    for chunk in config.chunks:
        identity = (chunk.expert_id, chunk.ffn_part)
        if identity not in rebased:
            continue
        stage = stages[identity]
        local_chunks.append(replace(
            chunk,
            use_cycle=rebased[identity]
            + max(0, chunk.use_cycle - stage.original_start_cycle),
        ))

    local_requests = []
    for request in config.compute_requests:
        identity = _request_stage(request)
        if identity is None:
            local_requests.append(request)
            continue
        if identity not in rebased:
            continue
        stage = stages[identity]
        local_requests.append(replace(
            request,
            issue_cycle=rebased[identity]
            + max(0, request.issue_cycle - stage.original_start_cycle),
        ))

    payload = dict(config.payload)
    system = dict(payload.get("system", {}))
    remote = sum(route.is_remote for route in routes)
    system.update({
        "num_gpus": contract.num_npus,
        "remote_route_replicas": remote,
        "total_route_replicas": len(routes),
        "remote_token_fraction": remote / len(routes) if routes else 0.0,
    })
    payload["system"] = system
    payload["ep_runtime"] = {
        "detailed_npu_id": detailed,
        "local_experts": list(local_experts),
        "active_local_experts": list(active),
        "local_route_replicas": sum(route.owner_npu == detailed for route in routes),
        "remote_route_replicas": remote,
    }
    local_config = replace(
        config,
        compute_cycles=local_compute_cycles,
        compute_intervals=((0, local_compute_cycles),),
        chunks=tuple(local_chunks),
        compute_requests=tuple(local_requests),
        payload=payload,
    )
    return DetailedNPUWorkload(
        local_config, contract, routes, local_experts, active,
        sum(route.owner_npu == detailed for route in routes), remote,
    )


def attach_ep_contract(payload: Mapping[str, object], *, default_num_npus: int = 2,
                       detailed_npu_id: int = 0) -> Mapping[str, object]:
    """Add deterministic DATE3 EP metadata to a generated workload payload."""
    value = dict(payload)
    provenance = dict(value.get("topology_provenance", {}))
    counts = tuple(int(item) for item in provenance.get("token_counts", ()))
    experts = sorted({int(item["expert_id"]) for item in value.get("chunks", ())})
    if not experts or experts != list(range(len(experts))) or len(counts) != len(experts):
        raise ValueError("DATE3 EP attachment requires contiguous experts and token counts")
    sweep = value.get("sweep", {})
    num_npus = (
        int(sweep.get("value"))
        if sweep.get("variable") == "expert_parallel"
        else int(default_num_npus)
    )
    top_k = int(provenance.get("top_k", 1))

    bases = {}
    for item in value.get("compute_requests", ()):
        match = REQUEST_STAGE.search(str(item.get("request_id", "")))
        if match:
            identity = tuple(map(int, match.groups()))
            bases[identity] = min(bases.get(identity, int(item["issue_cycle"])),
                                  int(item["issue_cycle"]))
    ordered = sorted(bases, key=lambda item: bases[item])
    stages = []
    for index, identity in enumerate(ordered):
        start = bases[identity]
        finish = (
            bases[ordered[index + 1]] if index + 1 < len(ordered)
            else max(start + 1, int(value["compute_cycles"]) - 32)
        )
        stages.append({
            "expert_id": identity[0], "ffn_part": identity[1],
            "original_start_cycle": start,
            "compute_cycles": max(1, finish - start),
        })
    owners = balanced_owner_map(len(experts), num_npus)
    value["ep"] = {
        "schema_version": 1,
        "num_experts": len(experts),
        "num_npus": num_npus,
        "detailed_npu_id": detailed_npu_id,
        "expert_owner_map": list(owners),
        "top_k": top_k,
        "routing_mode": "deterministic_topology_counts",
        "token_counts": list(counts),
        "source_token_distribution": "all_on_detailed",
        "expert_stages": stages,
    }
    system = dict(value.get("system", {}))
    system["num_gpus"] = num_npus
    # Dispatch moves the input activation replica; Return moves one expert
    # output replica.  The default keeps both tensors at the topology-derived
    # token payload width while leaving the values explicit and auditable.
    system.setdefault("result_payload_bytes", int(system.get("token_payload_bytes", 0)))
    system.setdefault("combine_cycles_per_token", 1)
    value["system"] = system
    source = str(provenance.get("source_path", ""))
    if "/DATE2/" in source:
        provenance["source_path"] = source.replace("/DATE2/", "/DATE3/")
    value["topology_provenance"] = provenance
    # Validate all derived invariants before a generated config reaches disk.
    EPContract.from_payload(value).routes()
    return value
