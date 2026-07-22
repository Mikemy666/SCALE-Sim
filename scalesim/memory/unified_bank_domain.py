"""Deterministic unified physical-Bank service model for MemDomain.

IA, Weight, OA/Accumulator, and prefetch traffic share one physical Bank pool.
This P2 model covers the common request/service path only. Residency, virtual
mapping lifetimes, and pressure-aware placement are added in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Dict, Iterable, Mapping, Optional, Tuple

from scalesim.memory.memdomain_policy import ResourceBudget


TENSOR_TYPES = {"ia", "weight", "oa", "accumulator"}
REQUEST_KINDS = {"read", "write", "prefetch"}


@dataclass(frozen=True)
class UnifiedMemoryRequest:
    request_id: str
    issue_cycle: int
    tensor_type: str
    object_id: str
    address: int
    size_bytes: int
    kind: str = "read"
    preferred_banks: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.request_id or not self.object_id:
            raise ValueError("request_id and object_id must not be empty")
        if self.issue_cycle < 0 or self.address < 0:
            raise ValueError("issue_cycle and address must be non-negative")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.tensor_type not in TENSOR_TYPES:
            raise ValueError(f"unsupported tensor type: {self.tensor_type}")
        if self.kind not in REQUEST_KINDS:
            raise ValueError(f"unsupported request kind: {self.kind}")


@dataclass(frozen=True)
class RequestService:
    request_id: str
    issue_cycle: int
    start_cycle: int
    completion_cycle: int
    queue_wait_cycles: int
    banks: Tuple[int, ...]
    beat_count: int


@dataclass(frozen=True)
class UnifiedDomainReport:
    services: Tuple[RequestService, ...]
    per_bank_accesses: Mapping[int, int]
    per_bank_busy_cycles: Mapping[int, int]
    per_bank_conflicts: Mapping[int, int]
    per_bank_queue_wait: Mapping[int, int]
    per_bank_max_queue_depth: Mapping[int, int]
    per_tensor_requests: Mapping[str, int]
    total_bytes: int
    total_beats: int
    total_queue_wait_cycles: int
    finish_cycle: int


class UnifiedBankDomain:
    """One shared Bank pool with deterministic address interleaving."""

    def __init__(self, resources: ResourceBudget, interleave_bytes: int = 64):
        if interleave_bytes <= 0:
            raise ValueError("interleave_bytes must be positive")
        self.resources = resources
        self.interleave_bytes = int(interleave_bytes)
        self.per_bank_bandwidth = (
            float(resources.bandwidth_bytes_per_cycle) / float(resources.bank_count)
        )
        if self.per_bank_bandwidth <= 0:
            raise ValueError("effective per-Bank bandwidth must be positive")

    def _validate_preferred_banks(self, request: UnifiedMemoryRequest) -> Tuple[int, ...]:
        banks = request.preferred_banks or tuple(range(self.resources.bank_count))
        if len(set(banks)) != len(banks):
            raise ValueError(f"duplicate preferred Bank in request {request.request_id}")
        if any(bank < 0 or bank >= self.resources.bank_count for bank in banks):
            raise ValueError(f"preferred Bank out of range in request {request.request_id}")
        if not banks:
            raise ValueError(f"empty preferred Bank set in request {request.request_id}")
        return banks

    def _beats(self, request: UnifiedMemoryRequest) -> Tuple[Tuple[int, int], ...]:
        allowed = self._validate_preferred_banks(request)
        beats = []
        remaining = int(request.size_bytes)
        cursor = int(request.address)
        while remaining:
            line_offset = cursor % self.interleave_bytes
            beat_bytes = min(remaining, self.interleave_bytes - line_offset)
            logical_line = cursor // self.interleave_bytes
            bank = allowed[logical_line % len(allowed)]
            beats.append((bank, beat_bytes))
            cursor += beat_bytes
            remaining -= beat_bytes
        return tuple(beats)

    def simulate(self, requests: Iterable[UnifiedMemoryRequest]) -> UnifiedDomainReport:
        priority = {"read": 0, "write": 0, "prefetch": 1}
        ordered = sorted(
            tuple(requests),
            key=lambda item: (item.issue_cycle, priority[item.kind], item.request_id),
        )
        identifiers = [request.request_id for request in ordered]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("request_id values must be unique")

        bank_ports = {
            bank: [0] * self.resources.ports_per_bank
            for bank in range(self.resources.bank_count)
        }
        accesses = {bank: 0 for bank in bank_ports}
        busy = {bank: 0 for bank in bank_ports}
        conflicts = {bank: 0 for bank in bank_ports}
        queue_wait = {bank: 0 for bank in bank_ports}
        outstanding = {bank: [] for bank in bank_ports}
        max_queue_depth = {bank: 0 for bank in bank_ports}
        per_tensor = {tensor: 0 for tensor in sorted(TENSOR_TYPES)}
        services = []
        total_bytes = 0
        total_beats = 0

        for request in ordered:
            beats = self._beats(request)
            starts = []
            completions = []
            used_banks = []
            for bank, beat_bytes in beats:
                arrival = request.issue_cycle
                outstanding[bank] = [
                    cycle for cycle in outstanding[bank] if cycle > arrival
                ]
                while len(outstanding[bank]) >= self.resources.request_buffer_depth:
                    arrival = min(outstanding[bank])
                    outstanding[bank] = [
                        cycle for cycle in outstanding[bank] if cycle > arrival
                    ]
                port_index = min(
                    range(len(bank_ports[bank])),
                    key=lambda index: (bank_ports[bank][index], index),
                )
                ready = bank_ports[bank][port_index]
                start = max(arrival, ready)
                duration = max(1, int(ceil(float(beat_bytes) / self.per_bank_bandwidth)))
                completion = start + duration
                bank_ports[bank][port_index] = completion
                outstanding[bank].append(completion)
                max_queue_depth[bank] = max(
                    max_queue_depth[bank], len(outstanding[bank])
                )
                wait = start - request.issue_cycle

                accesses[bank] += 1
                busy[bank] += duration
                queue_wait[bank] += wait
                if wait > 0:
                    conflicts[bank] += 1
                starts.append(start)
                completions.append(completion)
                used_banks.append(bank)

            service = RequestService(
                request_id=request.request_id,
                issue_cycle=request.issue_cycle,
                start_cycle=min(starts),
                completion_cycle=max(completions),
                queue_wait_cycles=max(completions) - request.issue_cycle
                - max(1, int(ceil(float(request.size_bytes) / self.resources.bandwidth_bytes_per_cycle))),
                banks=tuple(sorted(set(used_banks))),
                beat_count=len(beats),
            )
            services.append(service)
            per_tensor[request.tensor_type] += 1
            total_bytes += request.size_bytes
            total_beats += len(beats)

        total_wait = sum(queue_wait.values())
        finish = max((service.completion_cycle for service in services), default=0)
        return UnifiedDomainReport(
            services=tuple(services),
            per_bank_accesses=accesses,
            per_bank_busy_cycles=busy,
            per_bank_conflicts=conflicts,
            per_bank_queue_wait=queue_wait,
            per_bank_max_queue_depth=max_queue_depth,
            per_tensor_requests=per_tensor,
            total_bytes=total_bytes,
            total_beats=total_beats,
            total_queue_wait_cycles=total_wait,
            finish_cycle=finish,
        )
