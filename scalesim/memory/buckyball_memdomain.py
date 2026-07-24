"""Architectural contract for the Buckyball MemDomain DATE experiments.

The baseline statically owns 15 SP Banks (5/5/5 IA/Weight/OA) and 15 ACC
Banks.  MemDomain exposes the same thirty homogeneous physical Banks through
one allocation and address-translation domain.  This module intentionally
contains no paper-result constants; it provides legality and deterministic
compiler-cost ordering shared by every experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil
from typing import Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class BuckyballMemoryContract:
    bank_count: int = 30
    bank_width_bits: int = 128
    bank_entries: int = 128
    ports_per_bank: int = 1
    static_ia_banks: int = 5
    static_weight_banks: int = 5
    static_oa_banks: int = 5
    static_acc_banks: int = 15
    tile_size: int = 16
    input_bits: int = 8
    accumulator_bits: int = 32
    offchip_bandwidth_bits_per_cycle: int = 128
    offchip_startup_cycles: int = 20
    rmw_cycles: int = 3
    requant_pipeline_cycles: int = 3

    def __post_init__(self) -> None:
        if self.bank_count != 30:
            raise ValueError("DATE2 Buckyball contract requires 30 Banks")
        if self.bank_width_bits != 128 or self.bank_entries != 128:
            raise ValueError("a Buckyball Bank is 128 entries x 128 bits")
        if (
            self.static_ia_banks
            + self.static_weight_banks
            + self.static_oa_banks
            + self.static_acc_banks
            != self.bank_count
        ):
            raise ValueError("static ownership must conserve all physical Banks")
        if self.accumulator_bits % self.input_bits:
            raise ValueError("ACC/input width ratio must be integral")

    @property
    def bank_bytes(self) -> int:
        return self.bank_entries * self.bank_width_bits // 8

    @property
    def capacity_bytes(self) -> int:
        return self.bank_count * self.bank_bytes

    @property
    def per_bank_bandwidth_bytes_per_cycle(self) -> int:
        return self.bank_width_bits // 8

    @property
    def aggregate_bank_bandwidth_bytes_per_cycle(self) -> int:
        return self.bank_count * self.per_bank_bandwidth_bytes_per_cycle

    @property
    def acc_stripe_banks(self) -> int:
        return self.accumulator_bits // self.input_bits

    @property
    def static_usable_acc_banks(self) -> int:
        return (
            self.static_acc_banks // self.acc_stripe_banks
        ) * self.acc_stripe_banks

    @property
    def static_acc_fragmentation_banks(self) -> int:
        return self.static_acc_banks - self.static_usable_acc_banks

    @property
    def int8_tile_bytes(self) -> int:
        return self.tile_size * self.tile_size * self.input_bits // 8

    @property
    def accumulator_tile_bytes(self) -> int:
        return self.tile_size * self.tile_size * self.accumulator_bits // 8

    @property
    def accumulator_rmw_tile_cycles(self) -> int:
        return self.tile_size * self.rmw_cycles

    @property
    def requant_tile_cycles(self) -> int:
        return self.tile_size + self.requant_pipeline_cycles - 1

    def offchip_cycles(self, size_bytes: int) -> int:
        if size_bytes <= 0:
            raise ValueError("off-chip request must contain data")
        width = self.offchip_bandwidth_bits_per_cycle // 8
        return self.offchip_startup_cycles + ceil(size_bytes / width)


CONTRACT = BuckyballMemoryContract()


@dataclass(frozen=True)
class BankAllocation:
    ia: int
    weight: int
    oa: int
    accumulator: int

    @property
    def total(self) -> int:
        return self.ia + self.weight + self.oa + self.accumulator

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.ia, self.weight, self.oa, self.accumulator


STATIC_ALLOCATION = BankAllocation(5, 5, 5, 15)


def legal_allocations(
    contract: BuckyballMemoryContract = CONTRACT,
) -> Iterable[BankAllocation]:
    """Compiler-visible legal col combinations, including the static incumbent."""
    for ia, weight, oa in product(
        range(1, contract.bank_count + 1), repeat=3
    ):
        remaining = contract.bank_count - ia - weight - oa
        for acc in range(
            contract.acc_stripe_banks,
            remaining + 1,
            contract.acc_stripe_banks,
        ):
            yield BankAllocation(ia, weight, oa, acc)
    # The fixed baseline has a three-Bank unusable ACC tail.  It remains an
    # explicit incumbent even though 15 is not a legal unified ACC stripe.
    yield STATIC_ALLOCATION


@dataclass(frozen=True, order=True)
class CompilerObjective:
    """Confirmed lexicographic objective; performance terms dominate."""
    total_cycles: int
    exposed_memory_stall_cycles: int
    bank_conflicts: int
    allocated_banks: int
    stable_order: Tuple[int, int, int, int]


def select_compiler_allocation(
    candidates: Iterable[Tuple[BankAllocation, CompilerObjective]],
) -> Tuple[BankAllocation, CompilerObjective]:
    values = tuple(candidates)
    if not values:
        raise ValueError("compiler allocation search requires candidates")
    if not any(item[0] == STATIC_ALLOCATION for item in values):
        raise ValueError("static incumbent must be contained in compiler search")
    return min(values, key=lambda item: item[1])


class PhysicalBankAllocator:
    """Exclusive vBank-to-pBank mappings matching Buckyball's mapping table.

    Unlike a byte-packed cache, one physical Bank has at most one live virtual
    owner. Data tiles may reuse addresses inside that owner until explicit
    release. Static mode enforces the legacy IA/W/OA/ACC ownership ranges;
    unified mode permits every tensor to use every free Bank.
    """

    _STATIC_POOLS = {
        "ia": tuple(range(0, 5)),
        "weight": tuple(range(5, 10)),
        "oa": tuple(range(10, 15)),
        "accumulator": tuple(range(15, 30)),
    }

    def __init__(self, unified: bool, contract: BuckyballMemoryContract = CONTRACT):
        self.unified = bool(unified)
        self.contract = contract
        self.owner_by_bank: Dict[int, Optional[str]] = {
            bank: None for bank in range(contract.bank_count)
        }
        self.tensor_by_object: Dict[str, str] = {}
        self.banks_by_object: Dict[str, Tuple[int, ...]] = {}

    def allocate(
        self,
        object_id: str,
        tensor_type: str,
        col: int,
        pressure: Optional[Mapping[int, int]] = None,
    ) -> Tuple[int, ...]:
        if object_id in self.banks_by_object:
            raise ValueError(f"virtual Bank already allocated: {object_id}")
        if tensor_type not in self._STATIC_POOLS:
            raise ValueError(f"unsupported tensor type: {tensor_type}")
        if col <= 0:
            raise ValueError("col must be positive")
        if tensor_type == "accumulator" and self.unified:
            if col % self.contract.acc_stripe_banks:
                raise ValueError("unified ACC col must be a multiple of four")
        candidates = (
            tuple(range(self.contract.bank_count))
            if self.unified else self._STATIC_POOLS[tensor_type]
        )
        free = [bank for bank in candidates if self.owner_by_bank[bank] is None]
        ranked = sorted(free, key=lambda bank: ((pressure or {}).get(bank, 0), bank))
        if len(ranked) < col:
            raise MemoryError(f"insufficient {tensor_type} Banks for {object_id}")
        selected = tuple(ranked[:col])
        for bank in selected:
            self.owner_by_bank[bank] = object_id
        self.tensor_by_object[object_id] = tensor_type
        self.banks_by_object[object_id] = selected
        return selected

    def release(self, object_id: str) -> Tuple[int, ...]:
        if object_id not in self.banks_by_object:
            raise KeyError(f"unknown virtual Bank: {object_id}")
        banks = self.banks_by_object.pop(object_id)
        self.tensor_by_object.pop(object_id)
        for bank in banks:
            if self.owner_by_bank[bank] != object_id:
                raise AssertionError("physical Bank ownership corrupted")
            self.owner_by_bank[bank] = None
        return banks

    def resolve(self, object_id: str) -> Tuple[int, ...]:
        if object_id not in self.banks_by_object:
            raise KeyError(f"unallocated virtual Bank: {object_id}")
        return self.banks_by_object[object_id]

    @property
    def free_banks(self) -> Tuple[int, ...]:
        return tuple(bank for bank, owner in self.owner_by_bank.items() if owner is None)
