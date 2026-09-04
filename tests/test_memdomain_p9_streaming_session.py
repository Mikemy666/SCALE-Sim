import unittest

from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.chunk_residency import WeightChunk
from scalesim.memory.streaming_residency import StreamingLoadPlan, StreamingResidencyEngine
from scalesim.memory.unified_bank_domain import UnifiedBankDomain, UnifiedMemoryRequest
from scalesim.memory.virtual_bank_mapping import VirtualBankMappingTable


class UnifiedBankSessionTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceBudget(4, 4096, 64, 1, 8)
        self.domain = UnifiedBankDomain(self.resources, 16)
        self.requests = [
            UnifiedMemoryRequest("a", 0, "ia", "a", 0, 32),
            UnifiedMemoryRequest("b", 1, "weight", "b", 16, 64, "prefetch"),
            UnifiedMemoryRequest("c", 4, "accumulator", "c", 0, 32, "write"),
        ]

    def test_session_matches_batch_service_report(self):
        batch = self.domain.simulate(self.requests)
        session = self.domain.new_session()
        for request in self.requests:
            session.submit(request)
        self.assertEqual(session.report(), batch)

    def test_state_persists_across_submissions(self):
        session = self.domain.new_session()
        first = UnifiedMemoryRequest("a", 0, "weight", "a", 0, 64)
        second = UnifiedMemoryRequest("b", 0, "weight", "b", 0, 64)
        session.submit(first)
        service = session.submit(second)
        self.assertGreater(service.queue_wait_cycles, 0)

    def test_out_of_order_submission_is_rejected(self):
        session = self.domain.new_session()
        session.submit(UnifiedMemoryRequest("later", 10, "ia", "x", 0, 16))
        with self.assertRaisesRegex(ValueError, "chronological"):
            session.submit(UnifiedMemoryRequest("earlier", 9, "ia", "y", 0, 16))

    def test_same_cycle_prefetch_must_follow_compute(self):
        session = self.domain.new_session()
        session.submit(UnifiedMemoryRequest("pf", 0, "weight", "p", 0, 16, "prefetch"))
        with self.assertRaisesRegex(ValueError, "priority"):
            session.submit(UnifiedMemoryRequest("compute", 0, "ia", "c", 0, 16))

    def test_fixed_capacity_stream_releases_before_next_allocation(self):
        resources = ResourceBudget(1, 64, 16, 1, 4)
        domain = UnifiedBankDomain(resources, 16)
        mapping = VirtualBankMappingTable(resources)
        plans = [
            StreamingLoadPlan(WeightChunk("a", 0, 1, 0, 64, 10, 0), 0, "prefetch"),
            StreamingLoadPlan(WeightChunk("b", 0, 1, 1, 64, 30, 64), 1, "prefetch"),
        ]
        report = StreamingResidencyEngine(domain, mapping).run(plans)
        by_id = {item.chunk_id: item for item in report.chunks}
        self.assertEqual(report.peak_occupied_bytes, 64)
        self.assertGreater(by_id["b"].actual_issue_cycle, 1)
        self.assertEqual(mapping.statistics().occupied_bytes, 0)

    def test_capacity_wait_crossing_deadline_becomes_demand(self):
        resources = ResourceBudget(1, 64, 8, 1, 4)
        engine = StreamingResidencyEngine(
            UnifiedBankDomain(resources, 8), VirtualBankMappingTable(resources)
        )
        plans = [
            StreamingLoadPlan(WeightChunk("a", 0, 1, 0, 64, 20, 0), 0, "prefetch"),
            StreamingLoadPlan(WeightChunk("b", 0, 1, 1, 64, 10, 64), 1, "prefetch"),
        ]
        report = engine.run(plans)
        by_id = {item.chunk_id: item for item in report.chunks}
        self.assertEqual(by_id["b"].effective_kind, "demand")
        self.assertEqual(by_id["b"].classification, "demand_miss")

    def test_compute_and_streaming_load_share_session(self):
        resources = ResourceBudget(1, 128, 16, 1, 4)
        engine = StreamingResidencyEngine(
            UnifiedBankDomain(resources, 16), VirtualBankMappingTable(resources)
        )
        plan = StreamingLoadPlan(WeightChunk("a", 0, 1, 0, 64, 20, 0), 0, "prefetch")
        compute = UnifiedMemoryRequest("compute", 0, "ia", "ia", 0, 16)
        report = engine.run([plan], [compute])
        self.assertEqual(len(report.memory_report.services), 2)
        # Exclusive Buckyball vBank ownership delays allocation instead of
        # illegally byte-packing compute and prefetch objects into one pBank.
        self.assertGreater(report.allocation_wait_cycles, 0)

    def test_compute_waits_for_a_release_created_by_a_future_access(self):
        resources = ResourceBudget(1, 128, 16, 1, 4)
        engine = StreamingResidencyEngine(
            UnifiedBankDomain(resources, 16), VirtualBankMappingTable(resources)
        )
        compute = [
            # Object a owns the only exclusive vBank.  Its release cannot be
            # queued until this final access at cycle 10 is processed.
            UnifiedMemoryRequest("a0", 0, "ia", "a", 0, 16),
            UnifiedMemoryRequest("b0", 1, "accumulator", "b", 0, 16,
                                 "write", wmode=1, bank_group_size=1),
            UnifiedMemoryRequest("a1", 10, "ia", "a", 16, 16),
        ]
        report = engine.run([], compute, dynamic_compute_mapping=True)
        services = {item.request_id: item for item in report.memory_report.services}
        self.assertGreater(services["b0"].issue_cycle, 10)
        self.assertEqual(engine.mapping.statistics().occupied_bytes, 0)


if __name__ == "__main__":
    unittest.main()
