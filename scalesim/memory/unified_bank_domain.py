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
    # Buckyball target semantics: wmode=0 is a normal overwrite and wmode=1
    # is an atomic AccPipe read-add-write transaction.
    wmode: int = 0
    bank_group_size: int = 0

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
        if self.wmode not in (0, 1):
            raise ValueError("wmode must be 0 (overwrite) or 1 (accumulate)")
        if self.wmode == 1 and not (
            self.tensor_type == "accumulator" and self.kind == "write"
        ):
            raise ValueError("wmode=1 is valid only for accumulator writes")
        if self.bank_group_size < 0:
            raise ValueError("bank_group_size must be non-negative")


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
        if (
            self.resources.ports_per_bank == 1
            and request.address % self.interleave_bytes == 0
            and request.size_bytes % self.interleave_bytes == 0
        ):
            lines = request.size_bytes // self.interleave_bytes
            first = (request.address // self.interleave_bytes) % len(allowed)
            quotient, remainder = divmod(lines, len(allowed))
            grouped = {}
            for offset, bank in enumerate(allowed):
                extra = int((offset - first) % len(allowed) < remainder)
                count = quotient + extra
                if count:
                    grouped[bank] = count * self.interleave_bytes
            return tuple(sorted(grouped.items()))
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
        if self.resources.ports_per_bank == 1:
            # With one port per Bank, all beats from one request are serialized
            # exactly as one run. Compressing the run preserves completion and
            # port occupancy while avoiding millions of Python event objects.
            grouped: Dict[int, int] = {}
            for bank, beat_bytes in beats:
                grouped[bank] = grouped.get(bank, 0) + beat_bytes
            return tuple(sorted(grouped.items()))
        return tuple(beats)

    def _beat_duration(self, request: UnifiedMemoryRequest, beat_bytes: int) -> int:
        transfer = max(
            1, int(ceil(float(beat_bytes) / self.per_bank_bandwidth))
        )
        # Confirmed architecture contract: synchronous SRAM read (1), INT32
        # add (1), and SRAM writeback (1). The single Bank port remains locked
        # for the complete atomic RMW.
        return 3 * transfer if request.wmode == 1 else transfer

    def _logical_beats(self, request: UnifiedMemoryRequest, beat_bytes: int) -> int:
        return int(ceil(float(beat_bytes) / self.interleave_bytes))

    def _ideal_request_duration(self, request: UnifiedMemoryRequest) -> int:
        beats = self._beats(request)
        per_bank = {}
        for bank, beat_bytes in beats:
            per_bank[bank] = per_bank.get(bank, 0) + self._beat_duration(
                request, beat_bytes
            )
        return max(per_bank.values(), default=0)

    def _atomic_runs(
        self, request: UnifiedMemoryRequest
    ) -> Optional[Mapping[int, int]]:
        """Return beat counts for the common 4-Bank ACC tile fast path."""
        if (
            request.wmode != 1
            or self.resources.ports_per_bank != 1
            or request.address % self.interleave_bytes
            or request.size_bytes % self.interleave_bytes
        ):
            return None
        counts: Dict[int, int] = {}
        for bank, _ in self._beats(request):
            counts[bank] = counts.get(bank, 0) + 1
        return counts

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
                duration = self._beat_duration(request, beat_bytes)
                completion = start + duration
                bank_ports[bank][port_index] = completion
                outstanding[bank].append(completion)
                max_queue_depth[bank] = max(
                    max_queue_depth[bank], len(outstanding[bank])
                )
                wait = start - request.issue_cycle

                accesses[bank] += self._logical_beats(request, beat_bytes)
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
                - self._ideal_request_duration(request),
                banks=tuple(sorted(set(used_banks))),
                beat_count=sum(
                    self._logical_beats(request, size) for _, size in beats
                ),
            )
            services.append(service)
            per_tensor[request.tensor_type] += 1
            total_bytes += request.size_bytes
            total_beats += sum(
                self._logical_beats(request, size) for _, size in beats
            )

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

    def new_session(self) -> "UnifiedBankSession":
        return UnifiedBankSession(self)


class UnifiedBankSession:
    """Persistent chronological request service for event-driven execution."""

    def __init__(self, domain: UnifiedBankDomain):
        self.domain = domain
        resources = domain.resources
        self.bank_ports = {
            bank: [0] * resources.ports_per_bank
            for bank in range(resources.bank_count)
        }
        self.outstanding = {bank: [] for bank in self.bank_ports}
        self.accesses = {bank: 0 for bank in self.bank_ports}
        self.busy = {bank: 0 for bank in self.bank_ports}
        self.conflicts = {bank: 0 for bank in self.bank_ports}
        self.queue_wait = {bank: 0 for bank in self.bank_ports}
        self.max_queue_depth = {bank: 0 for bank in self.bank_ports}
        self.per_tensor = {tensor: 0 for tensor in sorted(TENSOR_TYPES)}
        self.services = []
        self.request_ids = set()
        self.total_bytes = 0
        self.total_beats = 0
        self.last_order_key = None

    @staticmethod
    def _order_key(request: UnifiedMemoryRequest) -> Tuple[int, int, str]:
        return (request.issue_cycle, 1 if request.kind == "prefetch" else 0, request.request_id)

    def submit(self, request: UnifiedMemoryRequest) -> RequestService:
        key = self._order_key(request)
        if self.last_order_key is not None and key < self.last_order_key:
            raise ValueError("session requests must be submitted in chronological priority order")
        if request.request_id in self.request_ids:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self.last_order_key = key
        self.request_ids.add(request.request_id)

        starts = []
        completions = []
        used_banks = []
        beats = self.domain._beats(request)
        for bank, beat_bytes in beats:
            arrival = request.issue_cycle
            self.outstanding[bank] = [
                cycle for cycle in self.outstanding[bank] if cycle > arrival
            ]
            while len(self.outstanding[bank]) >= self.domain.resources.request_buffer_depth:
                arrival = min(self.outstanding[bank])
                self.outstanding[bank] = [
                    cycle for cycle in self.outstanding[bank] if cycle > arrival
                ]
            port_index = min(
                range(len(self.bank_ports[bank])),
                key=lambda index: (self.bank_ports[bank][index], index),
            )
            ready = self.bank_ports[bank][port_index]
            start = max(arrival, ready)
            duration = self.domain._beat_duration(request, beat_bytes)
            completion = start + duration
            self.bank_ports[bank][port_index] = completion
            self.outstanding[bank].append(completion)
            self.max_queue_depth[bank] = max(
                self.max_queue_depth[bank], len(self.outstanding[bank])
            )
            wait = start - request.issue_cycle
            self.accesses[bank] += self.domain._logical_beats(
                request, beat_bytes
            )
            self.busy[bank] += duration
            self.queue_wait[bank] += wait
            if wait > 0:
                self.conflicts[bank] += 1
            starts.append(start)
            completions.append(completion)
            used_banks.append(bank)

        service = RequestService(
            request_id=request.request_id,
            issue_cycle=request.issue_cycle,
            start_cycle=min(starts),
            completion_cycle=max(completions),
            queue_wait_cycles=max(completions) - request.issue_cycle
            - self.domain._ideal_request_duration(request),
            banks=tuple(sorted(set(used_banks))),
            beat_count=sum(
                self.domain._logical_beats(request, size)
                for _, size in beats
            ),
        )
        self.services.append(service)
        self.per_tensor[request.tensor_type] += 1
        self.total_bytes += request.size_bytes
        self.total_beats += sum(
            self.domain._logical_beats(request, size) for _, size in beats
        )
        return service

    def pressure(self) -> Mapping[int, Mapping[str, int]]:
        return {
            bank: {
                "queue_depth": len(self.outstanding[bank]),
                "busy_cycles": self.busy[bank],
                "conflicts": self.conflicts[bank],
            }
            for bank in self.bank_ports
        }

    def report(self) -> UnifiedDomainReport:
        return UnifiedDomainReport(
            services=tuple(self.services),
            per_bank_accesses=dict(self.accesses),
            per_bank_busy_cycles=dict(self.busy),
            per_bank_conflicts=dict(self.conflicts),
            per_bank_queue_wait=dict(self.queue_wait),
            per_bank_max_queue_depth=dict(self.max_queue_depth),
            per_tensor_requests=dict(self.per_tensor),
            total_bytes=self.total_bytes,
            total_beats=self.total_beats,
            total_queue_wait_cycles=sum(self.queue_wait.values()),
            finish_cycle=max((item.completion_cycle for item in self.services), default=0),
        )
