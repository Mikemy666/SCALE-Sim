"""Chunk load, prefetch, residency, consumption, and release model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Tuple

from scalesim.memory.unified_bank_domain import UnifiedBankDomain, UnifiedMemoryRequest
from scalesim.memory.virtual_bank_mapping import (
    BankPressure,
    VirtualBankMappingTable,
    VirtualMemoryObject,
)


class ChunkState(str, Enum):
    REGISTERED = "registered"
    LOADING = "loading"
    RESIDENT = "resident"
    CONSUMED = "consumed"
    RELEASED = "released"
    CANCELED = "canceled"


@dataclass(frozen=True)
class WeightChunk:
    chunk_id: str
    expert_id: int
    ffn_part: int
    tile_id: int
    size_bytes: int
    use_cycle: int
    logical_address: int
    bank_group_size: int = 1

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if self.expert_id < 0 or self.ffn_part not in (1, 2) or self.tile_id < 0:
            raise ValueError("invalid expert/FFN/Tile identity")
        if self.size_bytes <= 0 or self.use_cycle < 0 or self.logical_address < 0:
            raise ValueError("invalid Chunk size/use cycle/address")
        if self.bank_group_size <= 0:
            raise ValueError("bank_group_size must be positive")


@dataclass
class ChunkRuntime:
    chunk: WeightChunk
    state: ChunkState = ChunkState.REGISTERED
    load_kind: Optional[str] = None
    issue_cycle: Optional[int] = None
    completion_cycle: Optional[int] = None
    consume_cycle: Optional[int] = None
    release_cycle: Optional[int] = None
    stall_cycles: int = 0
    classification: Optional[str] = None
    request_id: Optional[str] = None


@dataclass(frozen=True)
class ChunkResidencyReport:
    registered_chunks: int
    prefetch_requests: int
    demand_requests: int
    canceled_prefetches: int
    consumed_chunks: int
    timely_prefetches: int
    late_prefetches: int
    unused_prefetches: int
    demand_misses: int
    total_prefetch_bytes: int
    total_demand_bytes: int
    total_miss_stall_cycles: int
    prefetch_coverage: float
    prefetch_accuracy: float
    timely_prefetch_ratio: float
    late_prefetch_ratio: float
    unused_prefetch_ratio: float
    prefetch_occupancy_byte_cycles: int
    compute_transfer_overlap_cycles: int


def _merge_intervals(intervals: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _intersection_cycles(
    left: Iterable[Tuple[int, int]], right: Iterable[Tuple[int, int]]
) -> int:
    left_merged = _merge_intervals(left)
    right_merged = _merge_intervals(right)
    total = 0
    i = j = 0
    while i < len(left_merged) and j < len(right_merged):
        start = max(left_merged[i][0], right_merged[j][0])
        end = min(left_merged[i][1], right_merged[j][1])
        total += max(0, end - start)
        if left_merged[i][1] <= right_merged[j][1]:
            i += 1
        else:
            j += 1
    return total


class ChunkResidencyManager:
    def __init__(self, mapping: VirtualBankMappingTable):
        self.mapping = mapping
        self.chunks: Dict[str, ChunkRuntime] = {}
        self.requests: Dict[str, UnifiedMemoryRequest] = {}
        self.compute_intervals: Tuple[Tuple[int, int], ...] = ()
        self.finalized = False
        self.canceled_prefetches = 0

    def register(self, chunk: WeightChunk) -> ChunkRuntime:
        if chunk.chunk_id in self.chunks:
            raise ValueError(f"duplicate Chunk ID: {chunk.chunk_id}")
        runtime = ChunkRuntime(chunk)
        self.chunks[chunk.chunk_id] = runtime
        return runtime

    def set_compute_intervals(self, intervals: Iterable[Tuple[int, int]]) -> None:
        if self.finalized:
            raise ValueError("compute intervals must be set before transfer finalization")
        for start, end in intervals:
            if start < 0 or end < start:
                raise ValueError("invalid compute interval")
        self.compute_intervals = _merge_intervals(intervals)

    def _schedule(
        self,
        chunk_id: str,
        issue_cycle: int,
        load_kind: str,
        pressure: Optional[Mapping[int, BankPressure]],
    ) -> UnifiedMemoryRequest:
        if self.finalized:
            raise ValueError("cannot schedule after transfer finalization")
        runtime = self.chunks[chunk_id]
        if runtime.state != ChunkState.REGISTERED:
            raise ValueError(f"Chunk {chunk_id} is already scheduled")
        if issue_cycle < 0:
            raise ValueError("load issue cycle must be non-negative")
        chunk = runtime.chunk
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
        self.mapping.allocate(obj, issue_cycle, pressure)
        request_id = f"load:{chunk.chunk_id}"
        request = self.mapping.make_request(
            request_id=request_id,
            object_id=obj.object_id,
            cycle=issue_cycle,
            address=chunk.logical_address,
            size_bytes=chunk.size_bytes,
            kind="prefetch" if load_kind == "prefetch" else "read",
        )
        runtime.state = ChunkState.LOADING
        runtime.load_kind = load_kind
        runtime.issue_cycle = issue_cycle
        runtime.request_id = request_id
        self.requests[request_id] = request
        return request

    def prefetch(
        self, chunk_id: str, issue_cycle: int,
        pressure: Optional[Mapping[int, BankPressure]] = None,
    ) -> UnifiedMemoryRequest:
        return self._schedule(chunk_id, issue_cycle, "prefetch", pressure)

    def demand_load(
        self, chunk_id: str, issue_cycle: Optional[int] = None,
        pressure: Optional[Mapping[int, BankPressure]] = None,
    ) -> UnifiedMemoryRequest:
        runtime = self.chunks[chunk_id]
        issue = runtime.chunk.use_cycle if issue_cycle is None else issue_cycle
        return self._schedule(chunk_id, issue, "demand", pressure)

    def cancel_prefetch(self, chunk_id: str, cycle: int) -> None:
        if self.finalized:
            raise ValueError("cannot cancel after transfer finalization")
        runtime = self.chunks[chunk_id]
        if runtime.state != ChunkState.LOADING or runtime.load_kind != "prefetch":
            raise ValueError("only a scheduled prefetch can be canceled")
        if cycle < int(runtime.issue_cycle):
            raise ValueError("cancel cycle precedes prefetch issue")
        self.requests.pop(str(runtime.request_id))
        self.mapping.release(f"weight:{chunk_id}", cycle)
        runtime.state = ChunkState.CANCELED
        runtime.release_cycle = cycle
        runtime.classification = "canceled"
        self.canceled_prefetches += 1

    def finalize_transfers(self, domain: UnifiedBankDomain) -> None:
        if self.finalized:
            raise ValueError("transfers already finalized")
        report = domain.simulate(self.requests.values())
        by_id = {service.request_id: service for service in report.services}
        for runtime in self.chunks.values():
            if runtime.state == ChunkState.LOADING:
                runtime.completion_cycle = by_id[str(runtime.request_id)].completion_cycle
        self.finalized = True

    def advance(self, cycle: int) -> None:
        if not self.finalized:
            raise ValueError("transfers must be finalized before state advance")
        if cycle < 0:
            raise ValueError("advance cycle must be non-negative")
        for runtime in self.chunks.values():
            if (runtime.state == ChunkState.LOADING
                    and runtime.completion_cycle is not None
                    and runtime.completion_cycle <= cycle):
                runtime.state = ChunkState.RESIDENT

    def consume(self, chunk_id: str, cycle: Optional[int] = None) -> int:
        if not self.finalized:
            raise ValueError("transfers must be finalized before consumption")
        runtime = self.chunks[chunk_id]
        if runtime.state not in (ChunkState.LOADING, ChunkState.RESIDENT):
            raise ValueError(f"Chunk {chunk_id} is not loadable for consumption")
        use_cycle = runtime.chunk.use_cycle if cycle is None else cycle
        if use_cycle < runtime.chunk.use_cycle:
            raise ValueError("Chunk cannot be consumed before its declared use cycle")
        self.advance(use_cycle)
        if runtime.state not in (ChunkState.LOADING, ChunkState.RESIDENT):
            raise ValueError(f"Chunk {chunk_id} is not loadable for consumption")
        completion = int(runtime.completion_cycle)
        actual = max(use_cycle, completion)
        runtime.consume_cycle = actual
        runtime.stall_cycles = max(0, completion - use_cycle)
        if runtime.load_kind == "prefetch":
            runtime.classification = "timely" if completion <= use_cycle else "late"
        else:
            runtime.classification = "demand_miss"
        runtime.state = ChunkState.CONSUMED
        return runtime.stall_cycles

    def release(self, chunk_id: str, cycle: int) -> None:
        runtime = self.chunks[chunk_id]
        if runtime.state not in (ChunkState.LOADING, ChunkState.RESIDENT, ChunkState.CONSUMED):
            raise ValueError("only loaded or consumed Chunks can be released")
        if not self.finalized or runtime.completion_cycle is None:
            raise ValueError("cannot release before transfer finalization")
        minimum = int(runtime.completion_cycle)
        if runtime.consume_cycle is not None:
            minimum = max(minimum, runtime.consume_cycle)
        if cycle < minimum:
            raise ValueError("release cycle precedes load completion/consumption")
        self.advance(cycle)
        if runtime.state == ChunkState.RESIDENT:
            if runtime.load_kind != "prefetch":
                raise ValueError("unconsumed demand loads cannot be released")
            runtime.classification = "unused"
        self.mapping.release(f"weight:{chunk_id}", cycle)
        runtime.release_cycle = cycle
        runtime.state = ChunkState.RELEASED

    def report(self) -> ChunkResidencyReport:
        runtimes = tuple(self.chunks.values())
        prefetches = [
            item for item in runtimes
            if item.load_kind == "prefetch" and item.state != ChunkState.CANCELED
        ]
        demands = [item for item in runtimes if item.load_kind == "demand"]
        consumed = [item for item in runtimes if item.consume_cycle is not None]
        timely = [item for item in runtimes if item.classification == "timely"]
        late = [item for item in runtimes if item.classification == "late"]
        unused = [item for item in runtimes if item.classification == "unused"]
        demand_misses = [item for item in runtimes if item.classification == "demand_miss"]
        used_prefetches = len(timely) + len(late)
        completed_prefetches = len(prefetches)

        occupancy = 0
        transfer_intervals = []
        for item in prefetches:
            if item.issue_cycle is not None and item.completion_cycle is not None:
                transfer_intervals.append((item.issue_cycle, item.completion_cycle))
            if item.issue_cycle is not None:
                end = item.release_cycle
                if end is None:
                    end = item.consume_cycle or item.completion_cycle or item.issue_cycle
                occupancy += item.chunk.size_bytes * max(0, end - item.issue_cycle)

        def ratio(numerator: int, denominator: int) -> float:
            return float(numerator) / float(denominator) if denominator else 0.0

        return ChunkResidencyReport(
            registered_chunks=len(runtimes),
            prefetch_requests=completed_prefetches,
            demand_requests=len(demands),
            canceled_prefetches=self.canceled_prefetches,
            consumed_chunks=len(consumed),
            timely_prefetches=len(timely),
            late_prefetches=len(late),
            unused_prefetches=len(unused),
            demand_misses=len(demand_misses),
            total_prefetch_bytes=sum(item.chunk.size_bytes for item in prefetches),
            total_demand_bytes=sum(item.chunk.size_bytes for item in demands),
            total_miss_stall_cycles=sum(item.stall_cycles for item in consumed),
            prefetch_coverage=ratio(used_prefetches, len(consumed)),
            prefetch_accuracy=ratio(used_prefetches, completed_prefetches),
            timely_prefetch_ratio=ratio(len(timely), completed_prefetches),
            late_prefetch_ratio=ratio(len(late), completed_prefetches),
            unused_prefetch_ratio=ratio(len(unused), completed_prefetches),
            prefetch_occupancy_byte_cycles=occupancy,
            compute_transfer_overlap_cycles=_intersection_cycles(
                self.compute_intervals, transfer_intervals
            ),
        )
