from scalesim.memory.buckyball_memdomain import (
    CONTRACT,
    STATIC_ALLOCATION,
    BankAllocation,
    CompilerObjective,
    PhysicalBankAllocator,
    select_compiler_allocation,
)
from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.unified_bank_domain import UnifiedBankDomain, UnifiedMemoryRequest


def test_confirmed_physical_contract():
    assert CONTRACT.bank_bytes == 2048
    assert CONTRACT.capacity_bytes == 60 * 1024
    assert CONTRACT.aggregate_bank_bandwidth_bytes_per_cycle == 480
    assert CONTRACT.static_acc_fragmentation_banks == 3
    assert CONTRACT.int8_tile_bytes == 256
    assert CONTRACT.accumulator_tile_bytes == 1024
    assert CONTRACT.accumulator_rmw_tile_cycles == 48
    assert CONTRACT.requant_tile_cycles == 18


def test_four_bank_accumulator_tile_rmw_takes_48_cycles():
    resources = ResourceBudget(30, 30 * 2048, 480, 1, 32)
    domain = UnifiedBankDomain(resources, interleave_bytes=16)
    request = UnifiedMemoryRequest(
        "acc", 0, "accumulator", "acc-tile", 0, 1024, "write",
        (0, 1, 2, 3), 1,
    )
    service = domain.simulate((request,)).services[0]
    assert service.start_cycle == 0
    assert service.completion_cycle == 48
    assert service.queue_wait_cycles == 0


def test_compiler_objective_contains_and_can_fallback_to_static():
    faster_dynamic = BankAllocation(2, 8, 2, 16)
    allocation, _ = select_compiler_allocation((
        (faster_dynamic, CompilerObjective(100, 20, 4, 28, faster_dynamic.as_tuple())),
        (STATIC_ALLOCATION, CompilerObjective(110, 30, 5, 30, STATIC_ALLOCATION.as_tuple())),
    ))
    assert allocation == faster_dynamic

    allocation, _ = select_compiler_allocation((
        (faster_dynamic, CompilerObjective(120, 40, 1, 28, faster_dynamic.as_tuple())),
        (STATIC_ALLOCATION, CompilerObjective(110, 30, 5, 30, STATIC_ALLOCATION.as_tuple())),
    ))
    assert allocation == STATIC_ALLOCATION


def test_static_ownership_and_unified_low_pressure_mapping():
    static = PhysicalBankAllocator(unified=False)
    assert static.allocate("ia", "ia", 5) == (0, 1, 2, 3, 4)
    try:
        static.allocate("ia2", "ia", 1)
        assert False, "static IA must not borrow another pool"
    except MemoryError:
        pass

    unified = PhysicalBankAllocator(unified=True)
    selected = unified.allocate(
        "acc", "accumulator", 4, {0: 9, 1: 8, 2: 7, 3: 6}
    )
    assert selected == (4, 5, 6, 7)
    assert unified.resolve("acc") == selected
    unified.release("acc")
    assert len(unified.free_banks) == 30
