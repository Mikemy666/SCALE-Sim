import unittest

from scalesim.memory.chunk_residency import ChunkResidencyManager, WeightChunk
from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.prefetch_policy import (
    BankAwarePrefetchPolicy,
    BankSnapshot,
    NaivePrefetchPolicy,
    NoPrefetchPolicy,
    PrefetchAction,
    PrefetchOutcome,
    apply_prefetch_decision,
    select_oracle_prefetch,
    select_safe_prefetch,
)
from scalesim.memory.virtual_bank_mapping import BankPressure, VirtualBankMappingTable


class PrefetchPolicyTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceBudget(4, 400, 64, 1, 8)
        self.chunk = WeightChunk("c", 0, 1, 0, 80, 20, 0, bank_group_size=1)

    def snapshot(self, cycle=0, hot_bank=None, free=None):
        pressure = {}
        if hot_bank is not None:
            pressure[hot_bank] = BankPressure(10, 100, 20)
        return BankSnapshot(cycle, pressure, free or {0: 100, 1: 100, 2: 100, 3: 100})

    def test_no_prefetch_demands_at_use_cycle(self):
        decision = NoPrefetchPolicy().decide(self.chunk)
        self.assertEqual(decision.action, PrefetchAction.DEMAND)
        self.assertEqual(decision.issue_cycle, self.chunk.use_cycle)

    def test_naive_fixed_window_ignores_pressure_and_uses_fixed_banks(self):
        chunks = [
            WeightChunk(f"c{i}", 0, 1, i, 16, 10 * (i + 1), 16 * i)
            for i in range(3)
        ]
        decisions = NaivePrefetchPolicy(1, (0,)).plan(chunks)
        self.assertEqual([item.issue_cycle for item in decisions], [0, 10, 20])
        self.assertTrue(all(item.target_banks == (0,) for item in decisions))

    def test_zero_window_is_no_prefetch(self):
        decision = NaivePrefetchPolicy(0, (0,)).plan([self.chunk])[0]
        self.assertEqual(decision.action, PrefetchAction.DEMAND)

    def test_bank_aware_redirects_from_hot_default_bank(self):
        decision = BankAwarePrefetchPolicy().decide(
            self.chunk, self.snapshot(hot_bank=0), 5, default_banks=(0,)
        )
        self.assertEqual(decision.action, PrefetchAction.PREFETCH)
        self.assertNotIn(0, decision.target_banks)
        self.assertTrue(decision.redirected)

    def test_bank_aware_keeps_prefetch_when_all_banks_are_hot(self):
        pressure = {bank: BankPressure(10, 100, 20) for bank in range(4)}
        decision = BankAwarePrefetchPolicy().decide(
            self.chunk, BankSnapshot(0, pressure, {bank: 100 for bank in range(4)}), 5
        )
        self.assertEqual(decision.action, PrefetchAction.PREFETCH)
        self.assertEqual(decision.issue_cycle, 0)

    def test_bank_aware_preserves_late_but_capacity_feasible_prefetch(self):
        pressure = {bank: BankPressure(10, 100, 20) for bank in range(4)}
        decision = BankAwarePrefetchPolicy().decide(
            self.chunk, BankSnapshot(16, pressure, {bank: 100 for bank in range(4)}), 5
        )
        self.assertEqual(decision.action, PrefetchAction.PREFETCH)

    def test_capacity_is_checked_across_selected_group(self):
        chunk = WeightChunk("g", 0, 1, 0, 150, 20, 0, bank_group_size=2)
        decision = BankAwarePrefetchPolicy().decide(
            chunk, self.snapshot(free={0: 100, 1: 60, 2: 0, 3: 0}), 5
        )
        self.assertEqual(decision.action, PrefetchAction.PREFETCH)
        self.assertEqual(set(decision.target_banks), {0, 1})

    def test_group_search_skips_low_capacity_cooler_pair(self):
        chunk = WeightChunk("g", 0, 1, 0, 150, 20, 0, bank_group_size=2)
        pressure = {
            2: BankPressure(queue_depth=1),
            3: BankPressure(queue_depth=1),
        }
        snapshot = BankSnapshot(0, pressure, {0: 20, 1: 20, 2: 100, 3: 60})
        decision = BankAwarePrefetchPolicy().decide(chunk, snapshot, 5)
        self.assertEqual(decision.action, PrefetchAction.PREFETCH)
        self.assertEqual(set(decision.target_banks), {2, 3})

    def test_virtual_mapping_receives_a_ranked_bank_pool(self):
        decision = BankAwarePrefetchPolicy().decide(
            self.chunk, self.snapshot(hot_bank=0), 5, default_banks=(0, 1)
        )
        self.assertEqual(decision.action, PrefetchAction.PREFETCH)
        self.assertEqual(len(decision.target_banks), 2)
        self.assertNotIn(0, decision.target_banks)

    def test_decision_targets_are_enforced_by_mapping(self):
        manager = ChunkResidencyManager(VirtualBankMappingTable(self.resources))
        manager.register(self.chunk)
        decision = BankAwarePrefetchPolicy().decide(
            self.chunk, self.snapshot(hot_bank=0), 5, default_banks=(0,)
        )
        apply_prefetch_decision(manager, decision, self.snapshot(hot_bank=0).pressure)
        record = manager.mapping.records["weight:c"]
        self.assertEqual(record.physical_banks, decision.target_banks)

    def test_safe_prefetch_falls_back_to_no_prefetch(self):
        none = PrefetchOutcome("none", "none", 100)
        aware = PrefetchOutcome("aware", "bank_aware", 120)
        safe = select_safe_prefetch(none, aware)
        self.assertEqual(safe.total_cycles, 100)
        self.assertTrue(safe.metadata["fallback_to_no_prefetch"])

    def test_safe_prefetch_keeps_beneficial_bank_aware(self):
        none = PrefetchOutcome("none", "none", 100)
        aware = PrefetchOutcome("aware", "bank_aware", 80)
        safe = select_safe_prefetch(none, aware)
        self.assertEqual(safe.total_cycles, 80)
        self.assertFalse(safe.metadata["fallback_to_no_prefetch"])

    def test_oracle_candidates_must_include_no_prefetch(self):
        with self.assertRaisesRegex(ValueError, "include no-prefetch"):
            select_oracle_prefetch([PrefetchOutcome("aware", "bank_aware", 80)])

    def test_oracle_cannot_lose_to_no_prefetch(self):
        none = PrefetchOutcome("none", "none", 100)
        naive = PrefetchOutcome("naive", "naive", 130)
        aware = PrefetchOutcome("aware", "bank_aware", 90)
        oracle = select_oracle_prefetch([none, naive, aware])
        self.assertEqual(oracle.total_cycles, 90)


if __name__ == "__main__":
    unittest.main()
