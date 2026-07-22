"""Virtual-object to physical-Bank-group lifetime mapping for MemDomain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.unified_bank_domain import TENSOR_TYPES, UnifiedMemoryRequest


PLACEMENT_POLICIES = {
    "round_robin",
    "least_occupied",
    "least_queue_pressure",
    "conflict_aware",
}


@dataclass(frozen=True)
class VirtualMemoryObject:
    object_id: str
    tensor_type: str
    size_bytes: int
    bank_group_size: int = 1
    expert_id: int = -1
    ffn_part: int = 0
    tile_id: int = -1
    chunk_id: int = -1

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must not be empty")
        if self.tensor_type not in TENSOR_TYPES:
            raise ValueError(f"unsupported tensor type: {self.tensor_type}")
        if self.size_bytes <= 0 or self.bank_group_size <= 0:
            raise ValueError("size_bytes and bank_group_size must be positive")
        if self.ffn_part not in (0, 1, 2):
            raise ValueError("ffn_part must be 0, 1, or 2")


@dataclass(frozen=True)
class BankPressure:
    queue_depth: int = 0
    busy_cycles: int = 0
    conflicts: int = 0

    def __post_init__(self) -> None:
        if min(self.queue_depth, self.busy_cycles, self.conflicts) < 0:
            raise ValueError("Bank pressure values must be non-negative")

    @property
    def score(self) -> int:
        return self.queue_depth * 4 + self.conflicts * 2 + self.busy_cycles


@dataclass
class MappingRecord:
    object: VirtualMemoryObject
    physical_banks: Tuple[int, ...]
    bytes_per_bank: Mapping[int, int]
    allocation_cycle: int
    release_cycle: Optional[int] = None
    resolve_count: int = 0

    @property
    def active(self) -> bool:
        return self.release_cycle is None


@dataclass(frozen=True)
class MappingStatistics:
    mapping_count: int
    release_count: int
    allocation_failures: int
    resolve_count: int
    active_mappings: int
    occupied_bytes: int
    peak_occupied_bytes: int
    per_bank_occupied_bytes: Mapping[int, int]
    per_bank_peak_occupied_bytes: Mapping[int, int]


class VirtualBankMappingTable:
    """Capacity-safe mappings that remain stable for an object's lifetime."""

    def __init__(self, resources: ResourceBudget, policy: str = "least_occupied"):
        if policy not in PLACEMENT_POLICIES:
            raise ValueError(f"unsupported placement policy: {policy}")
        self.resources = resources
        self.policy = policy
        base, remainder = divmod(resources.capacity_bytes, resources.bank_count)
        self.bank_capacity = {
            bank: base + (1 if bank < remainder else 0)
            for bank in range(resources.bank_count)
        }
        self.bank_occupied = {bank: 0 for bank in self.bank_capacity}
        self.bank_peak = {bank: 0 for bank in self.bank_capacity}
        self.records: Dict[str, MappingRecord] = {}
        self.mapping_count = 0
        self.release_count = 0
        self.allocation_failures = 0
        self.resolve_count = 0
        self.peak_occupied_bytes = 0
        self._round_robin_cursor = 0

    def _free_bytes(self, bank: int) -> int:
        return self.bank_capacity[bank] - self.bank_occupied[bank]

    def _ordered_banks(self, pressure: Mapping[int, BankPressure]) -> Tuple[int, ...]:
        banks = tuple(range(self.resources.bank_count))
        if self.policy == "round_robin":
            start = self._round_robin_cursor % self.resources.bank_count
            return banks[start:] + banks[:start]
        if self.policy == "least_occupied":
            return tuple(sorted(banks, key=lambda bank: (self.bank_occupied[bank], bank)))
        if self.policy == "least_queue_pressure":
            return tuple(sorted(
                banks,
                key=lambda bank: (pressure.get(bank, BankPressure()).queue_depth,
                                  self.bank_occupied[bank], bank),
            ))
        return tuple(sorted(
            banks,
            key=lambda bank: (pressure.get(bank, BankPressure()).score,
                              self.bank_occupied[bank], bank),
        ))

    def _plan_bytes(
        self, banks: Tuple[int, ...], size_bytes: int
    ) -> Optional[Mapping[int, int]]:
        if size_bytes < len(banks):
            return None
        free = {bank: self._free_bytes(bank) for bank in banks}
        if any(value <= 0 for value in free.values()) or sum(free.values()) < size_bytes:
            return None
        planned = {bank: 1 for bank in banks}
        remaining = size_bytes - len(banks)
        while remaining:
            bank = max(
                banks,
                key=lambda item: (free[item] - planned[item], -self.bank_occupied[item], -item),
            )
            available = free[bank] - planned[bank]
            if available <= 0:
                return None
            grant = min(remaining, available)
            planned[bank] += grant
            remaining -= grant
        return planned

    def allocate(
        self,
        obj: VirtualMemoryObject,
        cycle: int,
        pressure: Optional[Mapping[int, BankPressure]] = None,
    ) -> MappingRecord:
        if cycle < 0:
            raise ValueError("allocation cycle must be non-negative")
        if obj.object_id in self.records:
            raise ValueError(f"object_id already used: {obj.object_id}")
        if obj.bank_group_size > self.resources.bank_count:
            self.allocation_failures += 1
            raise MemoryError("bank_group_size exceeds physical Bank count")

        ordered = self._ordered_banks(pressure or {})
        # Search combinations in policy order without changing state. A small
        # Bank count makes this deterministic bounded search appropriate here.
        from itertools import combinations

        selected = None
        planned = None
        for indices in combinations(range(len(ordered)), obj.bank_group_size):
            banks = tuple(ordered[index] for index in indices)
            candidate = self._plan_bytes(banks, obj.size_bytes)
            if candidate is not None:
                selected, planned = banks, candidate
                break
        if selected is None or planned is None:
            self.allocation_failures += 1
            raise MemoryError(f"insufficient unified Bank capacity for {obj.object_id}")

        for bank, amount in planned.items():
            self.bank_occupied[bank] += amount
            self.bank_peak[bank] = max(self.bank_peak[bank], self.bank_occupied[bank])
        record = MappingRecord(obj, tuple(selected), dict(planned), cycle)
        self.records[obj.object_id] = record
        self.mapping_count += 1
        self.peak_occupied_bytes = max(
            self.peak_occupied_bytes, sum(self.bank_occupied.values())
        )
        if self.policy == "round_robin":
            self._round_robin_cursor = (selected[-1] + 1) % self.resources.bank_count
        return record

    def resolve(self, object_id: str, cycle: int) -> MappingRecord:
        if object_id not in self.records:
            raise KeyError(f"unknown virtual object: {object_id}")
        record = self.records[object_id]
        if cycle < record.allocation_cycle:
            raise ValueError("object accessed before allocation")
        if record.release_cycle is not None and cycle >= record.release_cycle:
            raise ValueError("object accessed after release")
        record.resolve_count += 1
        self.resolve_count += 1
        return record

    def release(self, object_id: str, cycle: int) -> MappingRecord:
        record = self.resolve(object_id, cycle)
        for bank, amount in record.bytes_per_bank.items():
            self.bank_occupied[bank] -= amount
            if self.bank_occupied[bank] < 0:
                raise AssertionError("physical Bank occupancy underflow")
        record.release_cycle = cycle
        self.release_count += 1
        return record

    def make_request(
        self,
        request_id: str,
        object_id: str,
        cycle: int,
        address: int,
        size_bytes: int,
        kind: str = "read",
    ) -> UnifiedMemoryRequest:
        record = self.resolve(object_id, cycle)
        return UnifiedMemoryRequest(
            request_id=request_id,
            issue_cycle=cycle,
            tensor_type=record.object.tensor_type,
            object_id=object_id,
            address=address,
            size_bytes=size_bytes,
            kind=kind,
            preferred_banks=record.physical_banks,
        )

    def statistics(self) -> MappingStatistics:
        return MappingStatistics(
            mapping_count=self.mapping_count,
            release_count=self.release_count,
            allocation_failures=self.allocation_failures,
            resolve_count=self.resolve_count,
            active_mappings=sum(1 for record in self.records.values() if record.active),
            occupied_bytes=sum(self.bank_occupied.values()),
            peak_occupied_bytes=self.peak_occupied_bytes,
            per_bank_occupied_bytes=dict(self.bank_occupied),
            per_bank_peak_occupied_bytes=dict(self.bank_peak),
        )
