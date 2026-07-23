import unittest

from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.unified_bank_domain import UnifiedBankDomain
from scalesim.memory.virtual_bank_mapping import (
    BankPressure,
    VirtualBankMappingTable,
    VirtualMemoryObject,
)


class VirtualBankMappingTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceBudget(4, 400, 64, 1, 8)
        self.table = VirtualBankMappingTable(self.resources)

    @staticmethod
    def obj(name, tensor="weight", size=80, group=1):
        return VirtualMemoryObject(name, tensor, size, group, expert_id=0, ffn_part=1, chunk_id=0)

    def test_mapping_is_stable_until_release(self):
        record = self.table.allocate(self.obj("w", group=2), 10)
        self.assertIs(self.table.resolve("w", 11), record)
        self.assertEqual(self.table.resolve("w", 99).physical_banks, record.physical_banks)

    def test_release_recovers_exact_capacity(self):
        self.table.allocate(self.obj("w", size=120, group=2), 0)
        self.assertEqual(self.table.statistics().occupied_bytes, 120)
        self.table.release("w", 5)
        self.assertEqual(self.table.statistics().occupied_bytes, 0)
        self.assertEqual(self.table.statistics().release_count, 1)

    def test_access_before_allocation_or_after_release_is_rejected(self):
        self.table.allocate(self.obj("w"), 10)
        with self.assertRaisesRegex(ValueError, "before allocation"):
            self.table.resolve("w", 9)
        self.table.release("w", 20)
        with self.assertRaisesRegex(ValueError, "after release"):
            self.table.resolve("w", 20)

    def test_failed_allocation_is_atomic(self):
        before = self.table.statistics()
        with self.assertRaises(MemoryError):
            self.table.allocate(self.obj("large", size=401, group=4), 0)
        after = self.table.statistics()
        self.assertEqual(after.occupied_bytes, before.occupied_bytes)
        self.assertEqual(after.mapping_count, before.mapping_count)
        self.assertEqual(after.allocation_failures, 1)

    def test_duplicate_object_id_is_rejected_even_after_release(self):
        self.table.allocate(self.obj("w"), 0)
        self.table.release("w", 1)
        with self.assertRaisesRegex(ValueError, "already used"):
            self.table.allocate(self.obj("w"), 2)

    def test_least_occupied_spreads_objects(self):
        first = self.table.allocate(self.obj("a"), 0)
        second = self.table.allocate(self.obj("b"), 0)
        self.assertNotEqual(first.physical_banks, second.physical_banks)

    def test_pressure_aware_policy_avoids_hot_bank(self):
        table = VirtualBankMappingTable(self.resources, "conflict_aware")
        pressure = {0: BankPressure(queue_depth=10, busy_cycles=20, conflicts=30)}
        record = table.allocate(self.obj("w"), 0, pressure)
        self.assertNotIn(0, record.physical_banks)

    def test_conflict_aware_group_avoids_active_lifetime_overlap(self):
        table = VirtualBankMappingTable(self.resources, "conflict_aware")
        first = table.allocate(self.obj("a", size=40, group=2), 0)
        second = table.allocate(self.obj("b", size=40, group=2), 0)
        self.assertTrue(
            set(first.physical_banks).isdisjoint(second.physical_banks)
        )

    def test_round_robin_is_deterministic(self):
        table = VirtualBankMappingTable(self.resources, "round_robin")
        banks = [table.allocate(self.obj(str(index), size=10), 0).physical_banks[0]
                 for index in range(4)]
        self.assertEqual(banks, [0, 1, 2, 3])

    def test_mapping_drives_unified_request_bank_group(self):
        record = self.table.allocate(self.obj("w", group=2), 0)
        request = self.table.make_request("r", "w", 1, 0, 32)
        self.assertEqual(request.preferred_banks, record.physical_banks)
        report = UnifiedBankDomain(self.resources, 16).simulate([request])
        self.assertTrue(set(report.services[0].banks).issubset(record.physical_banks))

    def test_mapping_statistics_track_peak_and_resolves(self):
        self.table.allocate(self.obj("w", size=100), 0)
        self.table.resolve("w", 1)
        stats = self.table.statistics()
        self.assertEqual(stats.mapping_count, 1)
        self.assertEqual(stats.resolve_count, 1)
        self.assertEqual(stats.peak_occupied_bytes, 100)
        self.assertEqual(stats.active_mappings, 1)

    def test_seeded_capacity_stress_never_overfills_a_bank(self):
        table = VirtualBankMappingTable(self.resources, "least_occupied")
        for index in range(8):
            table.allocate(self.obj(f"o{index}", size=40), index)
            stats = table.statistics()
            for bank, occupied in stats.per_bank_occupied_bytes.items():
                self.assertLessEqual(occupied, table.bank_capacity[bank])
        self.assertEqual(table.statistics().occupied_bytes, 320)


if __name__ == "__main__":
    unittest.main()
