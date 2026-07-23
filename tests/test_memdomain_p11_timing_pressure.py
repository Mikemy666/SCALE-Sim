import tempfile
import unittest
from pathlib import Path

from scalesim.memory.memdomain_experiment import Baseline
from scalesim.memory.memdomain_runner import load_runner_config, run_raw_baseline
from scalesim.memory.moe_workload_catalog import write_runner_payload
from scalesim.memory.topology_workload import generate_topology_runner_payload

ROOT = Path(__file__).resolve().parents[1]


class TimingAndPressureRegressionTests(unittest.TestCase):
    def payload(self, window=2):
        payload = generate_topology_runner_payload(
            ROOT / "topologies/MoE/MoDSE.csv", "heterogeneous"
        )
        payload["policy"]["prefetch_window"] = window
        return payload

    def execute_baseline(self, payload, baseline):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            write_runner_payload(path, payload)
            return run_raw_baseline(load_runner_config(path), baseline)

    def test_stage_timeline_replaces_fixed_eight_cycle_spacing(self):
        payload = self.payload()
        uses = [item["use_cycle"] for item in payload["chunks"]]
        gaps = {right - left for left, right in zip(uses, uses[1:])}
        self.assertGreater(len(payload["compute_requests"]), 2)
        self.assertNotEqual(gaps, {8})
        self.assertGreater(min(uses), 32)

    def test_window_sweep_contains_late_and_timely_regions(self):
        ratios = []
        for window in (1, 2, 4, 8):
            row = self.execute_baseline(self.payload(window), Baseline.STATIC_NAIVEPF)
            ratios.append(row.timely_prefetch_ratio)
        self.assertGreater(ratios[0], 0.0)
        self.assertLess(ratios[0], 1.0)
        self.assertEqual(ratios[-1], 1.0)
        self.assertEqual(ratios, sorted(ratios))

    def test_bank_aware_uses_local_pressure_and_beats_dynamic_naive(self):
        payload = self.payload(2)
        naive = self.execute_baseline(payload, Baseline.DYNAMIC_NAIVEPF)
        aware = self.execute_baseline(payload, Baseline.MEMDOMAIN_RAW)
        self.assertEqual(aware.prefetch_requests, naive.prefetch_requests)
        self.assertEqual(aware.prefetch_bytes, naive.prefetch_bytes)
        self.assertEqual(aware.prefetch_coverage, naive.prefetch_coverage)
        self.assertLess(aware.total_cycles, naive.total_cycles)
        self.assertLess(aware.bank_conflict_count, naive.bank_conflict_count)
        self.assertGreater(aware.timely_prefetch_ratio,
                           naive.timely_prefetch_ratio)

    def test_capacity_bounded_adaptive_window_improves_joint_policy(self):
        fixed_payload = self.payload(2)
        adaptive_payload = self.payload(2)
        adaptive_payload["policy"].update({
            "adaptive_prefetch": True,
            "max_prefetch_window": 8,
            "max_prefetch_capacity_fraction": 0.25,
        })
        fixed = self.execute_baseline(fixed_payload, Baseline.MEMDOMAIN_RAW)
        adaptive = self.execute_baseline(
            adaptive_payload, Baseline.MEMDOMAIN_RAW
        )
        self.assertLess(adaptive.total_cycles, fixed.total_cycles)
        self.assertGreater(adaptive.timely_prefetch_ratio,
                           fixed.timely_prefetch_ratio)
        self.assertIn("adaptive_window=8", adaptive.candidate_source)

    def test_adaptive_window_respects_capacity_guard(self):
        payload = self.payload(2)
        payload["policy"].update({
            "adaptive_prefetch": True,
            "max_prefetch_window": 8,
            "max_prefetch_capacity_fraction": 0.01,
        })
        row = self.execute_baseline(payload, Baseline.MEMDOMAIN_RAW)
        self.assertIn("adaptive_window=2", row.candidate_source)


if __name__ == "__main__":
    unittest.main()
