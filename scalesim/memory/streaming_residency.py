"""Fixed-capacity event-driven Chunk streaming over the unified Bank session."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

from scalesim.memory.chunk_residency import WeightChunk
from scalesim.memory.unified_bank_domain import (
    RequestService,
    UnifiedBankDomain,
    UnifiedDomainReport,
    UnifiedMemoryRequest,
)
from scalesim.memory.virtual_bank_mapping import (
    BankPressure,
    VirtualBankMappingTable,
    VirtualMemoryObject,
)


@dataclass(frozen=True)
class StreamingLoadPlan:
    chunk: WeightChunk
    issue_cycle: int
    load_kind: str
    preferred_banks: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.issue_cycle < 0 or self.load_kind not in {"demand", "prefetch"}:
            raise ValueError("invalid streaming load plan")


@dataclass(frozen=True)
class StreamingChunkResult:
    chunk_id: str
    planned_kind: str
    effective_kind: str
    planned_issue_cycle: int
    actual_issue_cycle: int
    completion_cycle: int
    use_cycle: int
    consume_cycle: int
    release_cycle: int
    allocation_wait_cycles: int
    miss_stall_cycles: int
    classification: str
    physical_banks: Tuple[int, ...]


@dataclass(frozen=True)
class StreamingResidencyReport:
    chunks: Tuple[StreamingChunkResult, ...]
    memory_report: UnifiedDomainReport
    allocation_wait_cycles: int
    miss_stall_cycles: int
    timely_prefetches: int
    late_prefetches: int
    demand_misses: int
    peak_occupied_bytes: int
    occupancy_byte_cycles: int


class StreamingResidencyEngine:
    def __init__(
        self,
        domain: UnifiedBankDomain,
        mapping: VirtualBankMappingTable,
    ):
        if domain.resources != mapping.resources:
            raise ValueError("domain and mapping resources differ")
        self.domain = domain
        self.mapping = mapping

    def run(
        self,
        plans: Iterable[StreamingLoadPlan],
        compute_requests: Iterable[UnifiedMemoryRequest] = (),
        pressure: Optional[Mapping[int, BankPressure]] = None,
    ) -> StreamingResidencyReport:
        plans = tuple(plans)
        identifiers = [plan.chunk.chunk_id for plan in plans]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("streaming Chunk IDs must be unique")
        session = self.domain.new_session()
        events = []
        sequence = 0
        for request in compute_requests:
            key = session._order_key(request)
            heapq.heappush(events, (key, sequence, "compute", request, None))
            sequence += 1
        for plan in plans:
            request_kind = "prefetch" if plan.load_kind == "prefetch" else "read"
            key = (plan.issue_cycle, 1 if request_kind == "prefetch" else 0, f"load:{plan.chunk.chunk_id}")
            heapq.heappush(events, (key, sequence, "load", plan, plan.issue_cycle))
            sequence += 1

        releases = []
        results = []
        occupancy_byte_cycles = 0

        def release_through(cycle: int) -> None:
            while releases and releases[0][0] <= cycle:
                release_cycle, object_id = heapq.heappop(releases)
                self.mapping.release(object_id, release_cycle)

        while events:
            key, _, event_type, payload, original_issue = heapq.heappop(events)
            cycle = key[0]
            release_through(cycle)
            if event_type == "compute":
                session.submit(payload)
                continue

            plan = payload
            chunk = plan.chunk
            effective_kind = plan.load_kind
            if cycle >= chunk.use_cycle:
                effective_kind = "demand"
            obj = VirtualMemoryObject(
                object_id=f"weight:{chunk.chunk_id}",
                tensor_type="weight",
                size_bytes=chunk.size_bytes,
                bank_group_size=chunk.bank_group_size,
                expert_id=chunk.expert_id,
                ffn_part=chunk.ffn_part,
                tile_id=chunk.tile_id,
                chunk_id=chunk.tile_id,
            )
            try:
                record = self.mapping.allocate(
                    obj, cycle, pressure, plan.preferred_banks or None
                )
            except MemoryError:
                if not releases:
                    raise MemoryError(
                        f"streaming allocation for {chunk.chunk_id} cannot make progress"
                    )
                retry = max(cycle + 1, releases[0][0])
                retry_kind = "read" if retry >= chunk.use_cycle else (
                    "prefetch" if plan.load_kind == "prefetch" else "read"
                )
                retry_key = (retry, 1 if retry_kind == "prefetch" else 0, f"load:{chunk.chunk_id}")
                heapq.heappush(
                    events, (retry_key, sequence, "load", plan, original_issue)
                )
                sequence += 1
                continue

            request = self.mapping.make_request(
                request_id=f"load:{chunk.chunk_id}",
                object_id=obj.object_id,
                cycle=cycle,
                address=chunk.logical_address,
                size_bytes=chunk.size_bytes,
                kind="prefetch" if effective_kind == "prefetch" else "read",
            )
            service: RequestService = session.submit(request)
            consume = max(chunk.use_cycle, service.completion_cycle)
            release = consume
            heapq.heappush(releases, (release, obj.object_id))
            stall = max(0, service.completion_cycle - chunk.use_cycle)
            if effective_kind == "demand":
                classification = "demand_miss"
            else:
                classification = "timely" if service.completion_cycle <= chunk.use_cycle else "late"
            results.append(StreamingChunkResult(
                chunk_id=chunk.chunk_id,
                planned_kind=plan.load_kind,
                effective_kind=effective_kind,
                planned_issue_cycle=int(original_issue),
                actual_issue_cycle=cycle,
                completion_cycle=service.completion_cycle,
                use_cycle=chunk.use_cycle,
                consume_cycle=consume,
                release_cycle=release,
                allocation_wait_cycles=cycle - int(original_issue),
                miss_stall_cycles=stall,
                classification=classification,
                physical_banks=record.physical_banks,
            ))
            occupancy_byte_cycles += chunk.size_bytes * (release - cycle)

        release_through(max((item.release_cycle for item in results), default=0))
        if self.mapping.statistics().occupied_bytes != 0:
            raise AssertionError("streaming execution leaked resident capacity")
        ordered_results = tuple(sorted(results, key=lambda item: item.chunk_id))
        return StreamingResidencyReport(
            chunks=ordered_results,
            memory_report=session.report(),
            allocation_wait_cycles=sum(item.allocation_wait_cycles for item in results),
            miss_stall_cycles=sum(item.miss_stall_cycles for item in results),
            timely_prefetches=sum(item.classification == "timely" for item in results),
            late_prefetches=sum(item.classification == "late" for item in results),
            demand_misses=sum(item.classification == "demand_miss" for item in results),
            peak_occupied_bytes=self.mapping.statistics().peak_occupied_bytes,
            occupancy_byte_cycles=occupancy_byte_cycles,
        )
