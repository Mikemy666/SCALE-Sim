import csv
import json
import tempfile
import unittest
from pathlib import Path

from scalesim.memory.chunk_residency import WeightChunk
from scalesim.memory.memdomain_runner import load_runner_config, run_matrix
from scalesim.memory.pivot_ca_prefetch import (
    CandidateTile, CoverageAccuracyConstrainedPrefetchPolicy,
    CoverageAccuracyPolicyConfig, PrefetchQualityStats, TileLifetime,
    _ema, quality_from_lifetimes,
)
from scalesim.memory.pivot_ca_runner import run_pivot_ca, run_pivot_ca_file
from scalesim.memory.prefetch_policy import BankSnapshot
from scalesim.memory.virtual_bank_mapping import BankPressure

ROOT = Path(__file__).resolve().parents[1]
UNIT_CONFIG = ROOT / "configs/MoE/DATE3/unit_cases/MoDSE_minimal.json"
DATE2_TINY = ROOT / "configs/MoE/MoE_prefetch1/baseline/tiny_workload.json"


class PivotQualityTests(unittest.TestCase):
    def test_coverage_is_useful_over_required_bytes(self):
        stats = quality_from_lifetimes([
            TileLifetime("useful", 80, 10, 0, 5, 10, 10),
            TileLifetime("demand", 20, 10, None, None, 10, 10),
        ])
        self.assertEqual(stats.coverage, 0.8)

    def test_accuracy_is_useful_over_prefetched_bytes(self):
        stats = quality_from_lifetimes([
            TileLifetime("useful", 70, 10, 0, 5, 10, 10),
            TileLifetime("unused", 30, None, 0, 5, None, 10),
        ])
        self.assertEqual(stats.accuracy, 0.7)

    def test_late_is_not_useful(self):
        stats = quality_from_lifetimes([
            TileLifetime("late", 100, 10, 0, 11, 10, 11),
        ])
        self.assertEqual(stats.late_bytes, 100)
        self.assertEqual(stats.useful_timely_bytes, 0)

    def test_unused_lowers_accuracy(self):
        stats = quality_from_lifetimes([
            TileLifetime("unused", 100, None, 0, 5, None, 20),
        ])
        self.assertEqual(stats.unused_bytes, 100)
        self.assertEqual(stats.accuracy, 0.0)

    def test_evicted_before_use_is_not_useful(self):
        stats = quality_from_lifetimes([
            TileLifetime("evicted", 100, 20, 0, 5, 20, 10, 10),
        ])
        self.assertEqual(stats.evicted_before_use_bytes, 100)
        self.assertEqual(stats.useful_timely_bytes, 0)

    def test_identical_duplicate_tile_counts_once(self):
        tile = TileLifetime("same", 100, 10, 0, 5, 10, 10)
        stats = quality_from_lifetimes([tile, tile])
        self.assertEqual(stats.required_bytes, 100)
        self.assertEqual(stats.prefetched_bytes, 100)

    def test_invalid_denominators_are_not_fabricated(self):
        no_required = quality_from_lifetimes([
            TileLifetime("unused", 10, None, 0, 1, None, 2),
        ])
        no_prefetch = quality_from_lifetimes([
            TileLifetime("demand", 10, 2, None, None, 2, 2),
        ])
        self.assertFalse(no_required.coverage_valid)
        self.assertIsNone(no_required.coverage)
        self.assertFalse(no_prefetch.accuracy_valid)
        self.assertIsNone(no_prefetch.accuracy)

    def test_ema_formula(self):
        self.assertAlmostEqual(_ema(0.5, 1.0, 0.2), 0.6)


class PivotPolicyTests(unittest.TestCase):
    def config(self, **changes):
        values = {
            "ema_warmup_epochs": 0,
            "candidate_chunks": (1, 2),
            "candidate_windows": (1, 2, 4),
            "reference_chunk": 1,
            "reference_window": 2,
            "min_coverage": 0.4,
            "min_accuracy": 0.4,
            "minimum_positive_score": -1.0,
            "adaptation_cooldown": 0,
            "score_hysteresis": 0.0,
        }
        values.update(changes)
        return CoverageAccuracyPolicyConfig(**values)

    def snapshot(self, free=400, pressure=None):
        return BankSnapshot(
            0, pressure or {}, {i: free for i in range(4)},
            {i: 400 for i in range(4)},
        )

    def choose(self, policy, snapshot=None):
        return policy.choose(
            cycle=0, layer="x", expert=0, stage=1,
            tiles=(CandidateTile("a", 100, 20), CandidateTile("b", 100, 30)),
            snapshot=snapshot or self.snapshot(),
            lead_cycles={1: 2, 2: 8, 4: 32},
            bandwidth_bytes_per_cycle=100, group_size=1,
        )

    def test_minimum_window_selects_first_sufficient(self):
        value = CoverageAccuracyConstrainedPrefetchPolicy.minimum_window(
            (1, 2, 4), {1: 2, 2: 6, 4: 20}, 5, 1)
        self.assertEqual(value, 2)

    def test_quality_constraint_rejects_high_score_candidate(self):
        policy = CoverageAccuracyConstrainedPrefetchPolicy(
            self.config(min_coverage=0.75))
        _, rows = self.choose(policy)
        rejected = [row.candidate for row in rows
                    if row.candidate.chunk_size == 1]
        self.assertTrue(rejected)
        self.assertTrue(all(not item.feasible for item in rejected))
        self.assertIn("coverage_below_threshold",
                      {item.rejection_reason for item in rejected})

    def test_lower_pressure_group_is_selected(self):
        pressure = {0: BankPressure(20, 200, 20)}
        policy = CoverageAccuracyConstrainedPrefetchPolicy(self.config())
        selected, _ = self.choose(policy, self.snapshot(pressure=pressure))
        self.assertNotIn(0, selected.bank_group)

    def test_capacity_infeasible_candidate_is_removed(self):
        policy = CoverageAccuracyConstrainedPrefetchPolicy(self.config())
        selected, _ = self.choose(policy, self.snapshot(free=50))
        self.assertEqual(selected.chunk_size, 0)
        self.assertEqual(selected.rejection_reason, "noprefetch_fallback")

    def test_fallback_prefers_reference_when_quality_search_fails(self):
        policy = CoverageAccuracyConstrainedPrefetchPolicy(
            self.config(min_coverage=1.0, min_accuracy=1.0,
                        candidate_chunks=(1,)))
        selected, rows = self.choose(policy)
        self.assertEqual(selected.rejection_reason, "reference_fixed_fallback")
        self.assertTrue(any(row.fallback_level == 1 for row in rows))

    def test_hysteresis_holds_small_improvement(self):
        policy = CoverageAccuracyConstrainedPrefetchPolicy(
            self.config(score_hysteresis=100.0))
        first, _ = self.choose(policy)
        second, _ = self.choose(policy)
        self.assertEqual((first.chunk_size, first.window, first.bank_group),
                         (second.chunk_size, second.window, second.bank_group))


class PivotIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution = run_pivot_ca(load_runner_config(UNIT_CONFIG))
        cls.summary = cls.execution.summary

    def test_one_run_adapts_chunk_window_and_bank_group(self):
        self.assertLess(self.summary["selected_chunk_min"],
                        self.summary["selected_chunk_max"])
        self.assertLess(self.summary["selected_window_min"],
                        self.summary["selected_window_max"])
        self.assertGreater(self.summary["selected_bank_group_count"], 1)

    def test_quality_and_failure_classes_are_observable(self):
        self.assertGreater(self.summary["coverage"], 0)
        self.assertLess(self.summary["coverage"], 1)
        self.assertGreater(self.summary["accuracy"], 0)
        self.assertLess(self.summary["accuracy"], 1)
        self.assertGreater(self.summary["late_bytes"], 0)
        self.assertGreater(self.summary["unused_bytes"], 0)
        self.assertGreater(self.summary["evicted_before_use_bytes"], 0)
        self.assertGreater(self.summary["fallback_count"], 0)

    def test_total_cycle_additive_contract(self):
        names = (
            "compute_cycles", "bank_stall_cycles", "weight_load_stall_cycles",
            "prefetch_miss_stall_cycles", "prefetch_interference_stall_cycles",
            "mapping_overhead_cycles", "communication_stall_cycles",
            "other_stall_cycles",
        )
        self.assertEqual(self.summary["total_cycles"],
                         sum(self.summary[name] for name in names))
        self.assertTrue(all(self.summary[name] >= 0 for name in names))

    def test_shadow_reference_has_no_real_requests(self):
        self.assertEqual(self.summary["shadow_real_request_count"], 0)

    def test_reports_are_deterministic_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            first = run_pivot_ca_file(UNIT_CONFIG, Path(directory) / "a")
            second = run_pivot_ca_file(UNIT_CONFIG, Path(directory) / "b")
            self.assertEqual(first.summary, second.summary)
            payload = json.loads(UNIT_CONFIG.read_text(encoding="utf-8"))
            self.assertTrue(first.summary["config_hash"])
            self.assertEqual(first.summary["policy_name"], "PIVOT-CA")

    def test_date2_baseline_remains_runnable(self):
        rows = run_matrix(load_runner_config(DATE2_TINY))
        self.assertEqual(len(rows), 7)


if __name__ == "__main__":
    unittest.main()
