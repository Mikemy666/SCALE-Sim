"""The DATE3 off-chip link is shared, not one private link per Chunk."""

from scalesim.memory.chunk_residency import WeightChunk
from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.streaming_residency import (
    StreamingLoadPlan,
    StreamingResidencyEngine,
)
from scalesim.memory.unified_bank_domain import UnifiedBankDomain
from scalesim.memory.virtual_bank_mapping import VirtualBankMappingTable


def test_same_cycle_hbm_requests_are_serialized():
    resources = ResourceBudget(4, 4096, 128.0, 1, 8)
    engine = StreamingResidencyEngine(
        UnifiedBankDomain(resources, 16),
        VirtualBankMappingTable(resources, "round_robin"),
    )
    chunks = (
        WeightChunk("a", 0, 1, 0, 64, 100, 0),
        WeightChunk("b", 0, 1, 1, 64, 100, 64),
    )
    report = engine.run(tuple(
        StreamingLoadPlan(
            chunk, 0, "prefetch", offchip_latency_cycles=10
        ) for chunk in chunks
    ))
    by_id = {item.chunk_id: item for item in report.chunks}
    assert by_id["a"].hbm_issue_cycle == 0
    assert by_id["a"].hbm_complete_cycle == 10
    assert by_id["b"].hbm_issue_cycle == 10
    assert by_id["b"].hbm_complete_cycle == 20
    assert report.hbm_queue_wait_cycles == 10
    assert report.hbm_service_cycles == 20
    assert report.hbm_max_queue_depth == 2
