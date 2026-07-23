import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scalesim.memory.memdomain_experiment import (
    Baseline,
    read_matrix,
)
from scalesim.memory.memdomain_runner import (
    load_runner_config,
    run_matrix,
    run_matrix_file,
    run_raw_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/MoE/MoE_prefetch1/baseline/tiny_workload.json"


class MemDomainRunnerTests(unittest.TestCase):
    def test_runner_produces_valid_seven_row_matrix(self):
        rows = run_matrix(load_runner_config(CONFIG))
        self.assertEqual(len(rows), 7)
        self.assertEqual({row.baseline for row in rows}, {item.value for item in Baseline})

    def test_all_raw_rows_share_workload_and_resources(self):
        rows = run_matrix(load_runner_config(CONFIG))
        self.assertEqual(len({row.workload_hash for row in rows}), 1)
        self.assertEqual(len({row.resources for row in rows}), 1)

    def test_prefetch_and_mapping_metrics_come_from_execution(self):
        rows = {row.baseline: row for row in run_matrix(load_runner_config(CONFIG))}
        self.assertEqual(rows[Baseline.STATIC_NOPF.value].prefetch_requests, 0)
        self.assertGreater(rows[Baseline.STATIC_NAIVEPF.value].prefetch_requests, 0)
        self.assertGreater(rows[Baseline.MEMDOMAIN_RAW.value].mapping_count, 0)
        self.assertGreater(rows[Baseline.STATIC_NAIVEPF.value].compute_transfer_overlap_cycles, 0)

    def test_static_rows_are_selected_from_exhaustive_equal_width_groups(self):
        config = load_runner_config(CONFIG)
        rows = {row.baseline: row for row in run_matrix(config)}
        selected = rows[Baseline.STATIC_NOPF.value]
        fixed = []
        for bank in range(config.resources.bank_count):
            candidate = replace(config, static_weight_banks=(bank,))
            fixed.append(run_raw_baseline(candidate, Baseline.STATIC_NOPF).total_cycles)
        self.assertEqual(selected.total_cycles, min(fixed))
        self.assertTrue(selected.candidate_source.startswith("exhaustive_cyclic_static_weight_groups:"))

    def test_safe_and_oracle_never_exceed_static_no_prefetch(self):
        rows = {row.baseline: row for row in run_matrix(load_runner_config(CONFIG))}
        static = rows[Baseline.STATIC_NOPF.value].total_cycles
        self.assertLessEqual(rows[Baseline.MEMDOMAIN_SAFE.value].total_cycles, static)
        self.assertLessEqual(rows[Baseline.ORACLE.value].total_cycles, static)

    def test_dynamic_search_contains_matched_static_incumbent(self):
        rows = {row.baseline: row for row in run_matrix(load_runner_config(CONFIG))}
        self.assertLessEqual(
            rows[Baseline.DYNAMIC_NOPF.value].total_cycles,
            rows[Baseline.STATIC_NOPF.value].total_cycles,
        )
        self.assertLessEqual(
            rows[Baseline.DYNAMIC_NAIVEPF.value].total_cycles,
            rows[Baseline.STATIC_NAIVEPF.value].total_cycles,
        )

    def test_matched_naive_prefetch_work_is_identical(self):
        rows = {row.baseline: row for row in run_matrix(load_runner_config(CONFIG))}
        static = rows[Baseline.STATIC_NAIVEPF.value]
        dynamic = rows[Baseline.DYNAMIC_NAIVEPF.value]
        self.assertEqual(static.prefetch_requests, dynamic.prefetch_requests)
        self.assertEqual(static.prefetch_bytes, dynamic.prefetch_bytes)

    def test_large_mapping_overhead_triggers_safe_fallback(self):
        config = replace(load_runner_config(CONFIG), mapping_overhead_per_object=1000)
        rows = {row.baseline: row for row in run_matrix(config)}
        safe = rows[Baseline.MEMDOMAIN_SAFE.value]
        self.assertTrue(safe.fallback_used)
        self.assertEqual(safe.selected_candidate, Baseline.DYNAMIC_NAIVEPF.value)

    def test_matrix_file_is_byte_deterministic_after_p1_p2(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.csv"
            second = Path(directory) / "b.csv"
            run_matrix_file(CONFIG, first)
            run_matrix_file(CONFIG, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(read_matrix(first), read_matrix(second))


if __name__ == "__main__":
    unittest.main()
