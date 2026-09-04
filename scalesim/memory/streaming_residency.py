"""Fixed-capacity event-driven Chunk streaming over the unified Bank session."""

from __future__ import annotations

import heapq
import re
from collections import Counter
from dataclasses import dataclass, replace
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
    mapping_latency_cycles: int = 0
    # DATE3 opt-in lifetime controls.  DATE2 leaves both at their defaults and
    # therefore retains its original consume-at-first-use behavior.
    will_use: bool = True
    eviction_cycle: Optional[int] = None
    unused_release_cycle: Optional[int] = None
    # Time from HBM request issue until the bytes reach the on-chip Bank
    # ingress.  Kept separate from virtual-mapping latency so reports do not
    # misclassify transfer time as address-translation overhead.
    offchip_latency_cycles: int = 0

    def __post_init__(self) -> None:
        if (
            self.issue_cycle < 0
            or self.load_kind not in {"demand", "prefetch"}
            or self.mapping_latency_cycles < 0
            or self.offchip_latency_cycles < 0
        ):
            raise ValueError("invalid streaming load plan")
        if not self.will_use and self.load_kind != "prefetch":
            raise ValueError("only a prefetch may be marked unused")
        if self.eviction_cycle is not None and self.eviction_cycle < self.issue_cycle:
            raise ValueError("eviction cannot precede issue")
        if (self.unused_release_cycle is not None
                and self.unused_release_cycle < self.issue_cycle):
            raise ValueError("unused release cannot precede issue")


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
    mapping_latency_cycles: int = 0
    mapping_ready_cycle: int = 0
    first_use_cycle: Optional[int] = None
    eviction_cycle: Optional[int] = None
    offchip_latency_cycles: int = 0
    hbm_issue_cycle: int = 0
    hbm_complete_cycle: int = 0
    hbm_queue_wait_cycles: int = 0


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
    hbm_queue_wait_cycles: int
    hbm_service_cycles: int
    hbm_busy_cycles: int
    hbm_max_queue_depth: int


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
        dynamic_compute_mapping: bool = False,
        bind_prefetched_weight_reads: bool = False,
    ) -> StreamingResidencyReport:
        plans = tuple(plans)
        compute_requests = tuple(compute_requests)
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
        # One configured off-chip bandwidth represents one shared HBM command
        # and data channel.  Requests may be generated concurrently, but their
        # startup+serialization service cannot overlap on that channel.
        hbm_available_cycle = 0
        hbm_pending_finishes = []
        hbm_info = {}
        hbm_service_cycles = 0
        hbm_max_queue_depth = 0
        compute_remaining = Counter(
            request.object_id for request in compute_requests
        )
        compute_mappings = {}
        resident_weight_banks = {}
        planned_weight_banks = {}
        if bind_prefetched_weight_reads:
            for plan in plans:
                identity = (plan.chunk.expert_id, plan.chunk.ffn_part)
                candidates = (
                    plan.preferred_banks
                    or tuple(range(self.mapping.resources.bank_count))
                )
                planned_weight_banks.setdefault(identity, []).append(
                    (plan.chunk.use_cycle, tuple(candidates))
                )

        def stage_identity(value: str):
            match = re.search(r"(?:^|_)e(\d+)_ff([12])(?:_|$)", value)
            return ((int(match.group(1)), int(match.group(2)))
                    if match else None)

        def release_through(cycle: int) -> None:
            while releases and releases[0][0] <= cycle:
                release_cycle, object_id = heapq.heappop(releases)
                self.mapping.release(object_id, release_cycle)

        def capacity_retry_cycle(cycle: int) -> Optional[int]:
            """Return the next cycle at which allocation can make progress.

            A live compute vBank is released only after its final request has
            been submitted.  Consequently ``releases`` can legitimately be
            empty while a later compute event will create the release.  The
            old code treated that transient state as an unrecoverable capacity
            failure.  Run the next pending event first, then retry one cycle
            later; if a concrete release is already known, wait for it
            directly.  Returning ``None`` is the only true deadlock case.
            """
            if releases:
                return max(cycle + 1, releases[0][0])
            if events:
                next_event = min(item[0][0] for item in events)
                return max(cycle + 1, next_event + 1)
            return None

        while events:
            key, _, event_type, payload, original_issue = heapq.heappop(events)
            cycle = key[0]
            release_through(cycle)
            if event_type == "compute":
                request = payload
                identity = (
                    stage_identity(request.object_id)
                    or stage_identity(request.request_id)
                )
                if (
                    bind_prefetched_weight_reads
                    and request.tensor_type == "weight"
                    and identity in planned_weight_banks
                ):
                    choices = resident_weight_banks.get(
                        identity, planned_weight_banks[identity]
                    )
                    banks = min(
                        choices,
                        key=lambda item: (
                            abs(item[0] - request.issue_cycle), item[0], item[1]
                        ),
                    )[1]
                    group_size = request.bank_group_size or len(
                        request.preferred_banks
                    )
                    if group_size:
                        banks = banks[:min(group_size, len(banks))]
                    session.submit(replace(request, preferred_banks=banks))
                    continue
                mapped_id = f"{request.tensor_type}:{request.object_id}"
                if mapped_id not in compute_mappings:
                    group_size = request.bank_group_size or max(
                        1, len(request.preferred_banks)
                    )
                    # Buckyball maps one live virtual Bank group exclusively
                    # to its physical Banks. Claiming the complete group
                    # capacity prevents unrelated vBanks from byte-packing
                    # into the same pBank.
                    per_bank = (
                        self.mapping.resources.capacity_bytes
                        // self.mapping.resources.bank_count
                    )
                    obj = VirtualMemoryObject(
                        object_id=mapped_id,
                        tensor_type=request.tensor_type,
                        size_bytes=group_size * per_bank,
                        bank_group_size=group_size,
                    )
                    candidates = (
                        None if dynamic_compute_mapping
                        else request.preferred_banks or None
                    )
                    try:
                        compute_mappings[mapped_id] = self.mapping.allocate(
                            obj, cycle, pressure, candidates
                        )
                    except MemoryError:
                        # Capacity pressure is a scheduled wait, not a fatal
                        # mapping failure reported to the paper.  A release
                        # may not be queued yet when the currently resident
                        # object's final access is itself a future event.
                        self.mapping.allocation_failures -= 1
                        retry = capacity_retry_cycle(cycle)
                        if retry is None:
                            raise MemoryError(
                                f"compute allocation for {mapped_id} cannot make progress"
                            )
                        retried = UnifiedMemoryRequest(
                            request.request_id, retry, request.tensor_type,
                            request.object_id, request.address,
                            request.size_bytes, request.kind,
                            request.preferred_banks, request.wmode,
                            request.bank_group_size,
                        )
                        heapq.heappush(
                            events,
                            (session._order_key(retried), sequence,
                             "compute", retried, None),
                        )
                        sequence += 1
                        continue
                mapped = self.mapping.make_request(
                    request.request_id, mapped_id, cycle, request.address,
                    request.size_bytes, request.kind, request.wmode,
                )
                service = session.submit(mapped)
                compute_remaining[request.object_id] -= 1
                if compute_remaining[request.object_id] == 0:
                    heapq.heappush(
                        releases, (service.completion_cycle, mapped_id)
                    )
                continue

            plan = payload
            chunk = plan.chunk
            if event_type == "load" and (
                plan.mapping_latency_cycles or plan.offchip_latency_cycles
            ):
                while hbm_pending_finishes and hbm_pending_finishes[0] <= cycle:
                    heapq.heappop(hbm_pending_finishes)
                hbm_issue = max(cycle, hbm_available_cycle)
                hbm_complete = hbm_issue + plan.offchip_latency_cycles
                if plan.offchip_latency_cycles:
                    hbm_available_cycle = hbm_complete
                    heapq.heappush(hbm_pending_finishes, hbm_complete)
                    hbm_service_cycles += plan.offchip_latency_cycles
                    hbm_max_queue_depth = max(
                        hbm_max_queue_depth, len(hbm_pending_finishes)
                    )
                hbm_info[plan.chunk.chunk_id] = (
                    hbm_issue, hbm_complete, hbm_issue - cycle
                )
                ready = hbm_complete + plan.mapping_latency_cycles
                ready_kind = "read" if ready >= chunk.use_cycle else (
                    "prefetch" if plan.load_kind == "prefetch" else "read"
                )
                ready_key = (
                    ready, 1 if ready_kind == "prefetch" else 0,
                    f"load:{chunk.chunk_id}",
                )
                heapq.heappush(
                    events, (ready_key, sequence, "mapped_load", plan, original_issue)
                )
                sequence += 1
                continue
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
                self.mapping.allocation_failures -= 1
                retry = capacity_retry_cycle(cycle)
                if retry is None:
                    raise MemoryError(
                        f"streaming allocation for {chunk.chunk_id} cannot make progress"
                    )
                retry_kind = "read" if retry >= chunk.use_cycle else (
                    "prefetch" if plan.load_kind == "prefetch" else "read"
                )
                retry_key = (retry, 1 if retry_kind == "prefetch" else 0, f"load:{chunk.chunk_id}")
                heapq.heappush(
                    events, (retry_key, sequence, "mapped_load", plan, original_issue)
                )
                sequence += 1
                continue

            identity = (chunk.expert_id, chunk.ffn_part)
            if (
                bind_prefetched_weight_reads
                and plan.load_kind == "prefetch"
                and cycle < chunk.use_cycle
            ):
                resident_weight_banks.setdefault(identity, []).append(
                    (chunk.use_cycle, record.physical_banks)
                )

            request = self.mapping.make_request(
                request_id=f"load:{chunk.chunk_id}",
                object_id=obj.object_id,
                cycle=cycle,
                address=chunk.logical_address,
                size_bytes=chunk.size_bytes,
                kind="prefetch" if effective_kind == "prefetch" else "read",
            )
            service: RequestService = session.submit(request)
            evicted_before_use = (
                plan.eviction_cycle is not None
                and plan.eviction_cycle < chunk.use_cycle
            )
            if not plan.will_use:
                consume = service.completion_cycle
                release = max(
                    service.completion_cycle,
                    plan.unused_release_cycle
                    if plan.unused_release_cycle is not None
                    else chunk.use_cycle,
                )
            elif evicted_before_use:
                consume = max(service.completion_cycle, int(plan.eviction_cycle))
                release = consume
            else:
                consume = max(chunk.use_cycle, service.completion_cycle)
                release = consume
            heapq.heappush(releases, (release, obj.object_id))
            stall = (
                max(0, service.completion_cycle - chunk.use_cycle)
                if plan.will_use and not evicted_before_use else 0
            )
            if not plan.will_use:
                classification = "unused"
            elif evicted_before_use:
                classification = "evicted_before_use"
            elif effective_kind == "demand":
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
                mapping_latency_cycles=plan.mapping_latency_cycles,
                mapping_ready_cycle=(
                    int(original_issue) + plan.mapping_latency_cycles
                ),
                first_use_cycle=chunk.use_cycle if plan.will_use else None,
                eviction_cycle=plan.eviction_cycle,
                offchip_latency_cycles=plan.offchip_latency_cycles,
                hbm_issue_cycle=hbm_info.get(
                    chunk.chunk_id, (int(original_issue), int(original_issue), 0)
                )[0],
                hbm_complete_cycle=hbm_info.get(
                    chunk.chunk_id, (int(original_issue), int(original_issue), 0)
                )[1],
                hbm_queue_wait_cycles=hbm_info.get(
                    chunk.chunk_id, (int(original_issue), int(original_issue), 0)
                )[2],
            ))
            occupancy_byte_cycles += chunk.size_bytes * (release - cycle)

        while releases:
            release_through(releases[0][0])
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
            hbm_queue_wait_cycles=sum(
                item.hbm_queue_wait_cycles for item in results
            ),
            hbm_service_cycles=hbm_service_cycles,
            hbm_busy_cycles=hbm_service_cycles,
            hbm_max_queue_depth=hbm_max_queue_depth,
        )
