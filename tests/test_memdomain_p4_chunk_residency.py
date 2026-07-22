import unittest

from scalesim.memory.chunk_residency import ChunkResidencyManager, ChunkState, WeightChunk
from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.unified_bank_domain import UnifiedBankDomain
from scalesim.memory.virtual_bank_mapping import VirtualBankMappingTable


class ChunkResidencyTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceBudget(4, 4096, 64, 1, 8)
        self.mapping = VirtualBankMappingTable(self.resources)
        self.manager = ChunkResidencyManager(self.mapping)
        self.domain = UnifiedBankDomain(self.resources, interleave_bytes=16)

    @staticmethod
    def chunk(name, use, address=0, size=64):
        return WeightChunk(name, 0, 1, 0, size, use, address, bank_group_size=4)

    def test_timely_prefetch(self):
        self.manager.register(self.chunk("c", use=10))
        self.manager.prefetch("c", 0)
        self.manager.finalize_transfers(self.domain)
        self.manager.advance(10)
        self.assertEqual(self.manager.chunks["c"].state, ChunkState.RESIDENT)
        self.assertEqual(self.manager.consume("c"), 0)
        self.assertEqual(self.manager.chunks["c"].classification, "timely")

    def test_late_prefetch_reports_exact_stall(self):
        resources = ResourceBudget(1, 4096, 1, 1, 8)
        manager = ChunkResidencyManager(VirtualBankMappingTable(resources))
        manager.register(WeightChunk("c", 0, 1, 0, 8, 4, 0))
        manager.prefetch("c", 0)
        manager.finalize_transfers(UnifiedBankDomain(resources, 8))
        self.assertEqual(manager.consume("c"), 4)
        self.assertEqual(manager.chunks["c"].classification, "late")

    def test_demand_load_is_a_miss(self):
        self.manager.register(self.chunk("c", use=10))
        self.manager.demand_load("c")
        self.manager.finalize_transfers(self.domain)
        self.assertGreater(self.manager.consume("c"), 0)
        self.assertEqual(self.manager.chunks["c"].classification, "demand_miss")

    def test_unused_prefetch_requires_release_after_completion(self):
        self.manager.register(self.chunk("c", use=10))
        self.manager.prefetch("c", 0)
        self.manager.finalize_transfers(self.domain)
        completion = self.manager.chunks["c"].completion_cycle
        self.manager.release("c", completion)
        self.assertEqual(self.manager.chunks["c"].classification, "unused")
        self.assertEqual(self.mapping.statistics().occupied_bytes, 0)

    def test_cancel_removes_request_and_recovers_capacity(self):
        self.manager.register(self.chunk("c", use=10))
        self.manager.prefetch("c", 0)
        self.manager.cancel_prefetch("c", 1)
        self.manager.finalize_transfers(self.domain)
        self.assertEqual(self.manager.chunks["c"].state, ChunkState.CANCELED)
        self.assertEqual(self.mapping.statistics().occupied_bytes, 0)
        report = self.manager.report()
        self.assertEqual(report.prefetch_requests, 0)
        self.assertEqual(report.total_prefetch_bytes, 0)
        self.assertEqual(report.canceled_prefetches, 1)

    def test_prefetch_and_demand_metrics_are_not_hit_rate_proxies(self):
        self.manager.register(self.chunk("timely", 10, 0))
        self.manager.register(self.chunk("unused", 10, 64))
        self.manager.register(self.chunk("demand", 10, 128))
        self.manager.prefetch("timely", 0)
        self.manager.prefetch("unused", 0)
        self.manager.demand_load("demand")
        self.manager.finalize_transfers(self.domain)
        self.manager.consume("timely")
        self.manager.consume("demand")
        self.manager.release("unused", self.manager.chunks["unused"].completion_cycle)
        report = self.manager.report()
        self.assertEqual(report.timely_prefetches, 1)
        self.assertEqual(report.unused_prefetches, 1)
        self.assertEqual(report.demand_misses, 1)
        self.assertEqual(report.prefetch_coverage, 0.5)
        self.assertEqual(report.prefetch_accuracy, 0.5)

    def test_compute_transfer_overlap_uses_interval_union(self):
        self.manager.register(self.chunk("a", 20, 0, 128))
        self.manager.register(self.chunk("b", 20, 128, 128))
        self.manager.set_compute_intervals([(0, 10), (5, 15)])
        self.manager.prefetch("a", 0)
        self.manager.prefetch("b", 0)
        self.manager.finalize_transfers(self.domain)
        report = self.manager.report()
        self.assertLessEqual(report.compute_transfer_overlap_cycles, 15)
        self.assertGreater(report.compute_transfer_overlap_cycles, 0)

    def test_duplicate_schedule_and_early_release_are_rejected(self):
        self.manager.register(self.chunk("c", 10))
        self.manager.prefetch("c", 0)
        with self.assertRaisesRegex(ValueError, "already scheduled"):
            self.manager.prefetch("c", 1)
        self.manager.finalize_transfers(self.domain)
        with self.assertRaisesRegex(ValueError, "precedes"):
            self.manager.release("c", 0)

    def test_report_byte_counts_are_conserved(self):
        self.manager.register(self.chunk("p", 10, size=64))
        self.manager.register(self.chunk("d", 10, 64, size=128))
        self.manager.prefetch("p", 0)
        self.manager.demand_load("d")
        self.manager.finalize_transfers(self.domain)
        report = self.manager.report()
        self.assertEqual(report.total_prefetch_bytes, 64)
        self.assertEqual(report.total_demand_bytes, 128)


if __name__ == "__main__":
    unittest.main()
