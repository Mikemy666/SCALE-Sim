import unittest
import random

from scalesim.memory.memdomain_policy import (
    CycleBreakdown,
    PolicyResult,
    ResourceBudget,
    assert_theoretical_order,
    best_static,
    select_oracle_dynamic,
    select_safe_dynamic,
)


class MemDomainP1PolicyTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceBudget(24, 3 * 1024 * 1024, 384, 1, 32)

    def result(self, name, kind, total_stall, allocation=(8, 8, 8), **parts):
        cycles = CycleBreakdown(compute=100, bank_stall=total_stall, **parts)
        return PolicyResult(name, kind, self.resources, cycles, allocation)

    def test_end_to_end_objective_includes_every_stall_component(self):
        cycles = CycleBreakdown(100, 2, 3, 5, 7, 11, 13, 17)
        self.assertEqual(cycles.total, 158)

    def test_best_static_uses_total_cycles_not_bank_stall_only(self):
        low_bank_high_other = self.result("a", "static", 1, other_stall=50)
        higher_bank_low_total = self.result("b", "static", 10)
        self.assertEqual(best_static([low_bank_high_other, higher_bank_low_total]).name, "b")

    def test_oracle_candidate_set_contains_and_cannot_lose_to_best_static(self):
        statics = [self.result("s0", "static", 30), self.result("s1", "static", 20)]
        runtime = self.result("runtime", "runtime_dynamic", 40)
        oracle = select_oracle_dynamic(statics, [runtime])
        self.assertEqual(oracle.total_cycles, 120)
        self.assertTrue(oracle.metadata["fallback_to_static"])

    def test_safe_dynamic_uses_runtime_when_end_to_end_better(self):
        statics = [self.result("static", "static", 20)]
        runtime = self.result("runtime", "runtime_dynamic", 10, mapping_overhead=5)
        safe = select_safe_dynamic(statics, runtime)
        self.assertEqual(safe.total_cycles, 115)
        self.assertFalse(safe.metadata["fallback_to_static"])

    def test_safe_dynamic_falls_back_when_mapping_overhead_loses(self):
        statics = [self.result("static", "static", 20)]
        runtime = self.result("runtime", "runtime_dynamic", 5, mapping_overhead=30)
        safe = select_safe_dynamic(statics, runtime)
        self.assertEqual(safe.total_cycles, 120)
        self.assertTrue(safe.metadata["fallback_to_static"])

    def test_resource_mismatch_is_rejected(self):
        static = self.result("static", "static", 20)
        other = ResourceBudget(23, 3 * 1024 * 1024, 384, 1, 32)
        runtime = PolicyResult(
            "runtime", "runtime_dynamic", other, CycleBreakdown(100, 1), (7, 8, 8)
        )
        with self.assertRaisesRegex(ValueError, "different resources"):
            select_safe_dynamic([static], runtime)

    def test_bank_allocation_must_conserve_total(self):
        with self.assertRaisesRegex(ValueError, "conserve"):
            self.result("bad", "static", 1, allocation=(8, 8, 7))

    def test_negative_cost_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CycleBreakdown(100, mapping_overhead=-1)

    def test_explicit_theoretical_order_check(self):
        statics = [self.result("s0", "static", 30), self.result("s1", "static", 20)]
        runtime = self.result("runtime", "runtime_dynamic", 25)
        oracle = select_oracle_dynamic(statics, [runtime])
        safe = select_safe_dynamic(statics, runtime)
        assert_theoretical_order(statics, oracle, safe)

    def test_order_holds_across_seeded_candidate_sweep(self):
        rng = random.Random(20260722)
        allocations = ((8, 8, 8), (4, 14, 6), (12, 8, 4), (2, 11, 11))
        for sample in range(100):
            statics = [
                self.result(f"s{index}", "static", rng.randrange(0, 200), allocation)
                for index, allocation in enumerate(allocations)
            ]
            runtime = self.result(
                f"runtime{sample}", "runtime_dynamic", rng.randrange(0, 220),
                allocations[sample % len(allocations)], mapping_overhead=rng.randrange(0, 30),
            )
            oracle = select_oracle_dynamic(statics, [runtime])
            safe = select_safe_dynamic(statics, runtime)
            assert_theoretical_order(statics, oracle, safe)


if __name__ == "__main__":
    unittest.main()
