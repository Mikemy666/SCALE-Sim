import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scalesim.memory.memdomain_experiment import (
    Baseline,
    ExperimentRow,
    TheoreticalContractViolation,
    derive_selected_row,
    read_matrix,
    validate_theoretical_contract,
    validate_matrix,
    workload_digest,
    write_matrix,
)


class MemDomainExperimentMatrixTests(unittest.TestCase):
    def row(self, baseline, total, candidate_source="measured"):
        return ExperimentRow(
            1, "exp", "switch-tiny", "abc", baseline.value, candidate_source,
            24, 3 * 1024 * 1024, 384.0, 1, 32,
            100, total - 100, 0, 0, 0, 0, 0, 0, total,
        )

    def matrix(self):
        static = self.row(Baseline.STATIC_NOPF, 150)
        static_pf = self.row(Baseline.STATIC_NAIVEPF, 160)
        dynamic = self.row(Baseline.DYNAMIC_NOPF, 140)
        dynamic_pf = self.row(Baseline.DYNAMIC_NAIVEPF, 145)
        raw = self.row(Baseline.MEMDOMAIN_RAW, 130)
        safe = derive_selected_row(
            Baseline.MEMDOMAIN_SAFE, [static, dynamic, raw],
            [Baseline.STATIC_NOPF, Baseline.DYNAMIC_NOPF, Baseline.MEMDOMAIN_RAW],
        )
        oracle = derive_selected_row(
            Baseline.ORACLE, [static, static_pf, dynamic, dynamic_pf, raw, safe],
            [Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF, Baseline.DYNAMIC_NOPF,
             Baseline.DYNAMIC_NAIVEPF, Baseline.MEMDOMAIN_RAW, Baseline.MEMDOMAIN_SAFE],
        )
        return [static, static_pf, dynamic, dynamic_pf, raw, safe, oracle]

    def test_complete_matrix_passes(self):
        self.assertEqual(len(validate_matrix(self.matrix())), 7)
        self.assertEqual(len(validate_theoretical_contract(self.matrix())), 7)

    def test_dynamic_no_prefetch_must_dominate_static(self):
        rows = self.matrix()
        index = next(i for i, row in enumerate(rows)
                     if row.baseline == Baseline.DYNAMIC_NOPF.value)
        rows[index] = self.row(Baseline.DYNAMIC_NOPF, 151)
        with self.assertRaisesRegex(
            TheoreticalContractViolation, "Dynamic-NoPF must not exceed"
        ):
            validate_theoretical_contract(rows)

    def test_dynamic_naive_prefetch_must_dominate_static_naive(self):
        rows = self.matrix()
        index = next(i for i, row in enumerate(rows)
                     if row.baseline == Baseline.DYNAMIC_NAIVEPF.value)
        rows[index] = self.row(Baseline.DYNAMIC_NAIVEPF, 161)
        with self.assertRaisesRegex(
            TheoreticalContractViolation, "Dynamic-NaivePF must not exceed"
        ):
            validate_theoretical_contract(rows)

    def test_naive_prefetch_comparison_requires_identical_workload(self):
        rows = self.matrix()
        index = next(i for i, row in enumerate(rows)
                     if row.baseline == Baseline.DYNAMIC_NAIVEPF.value)
        rows[index] = replace(rows[index], prefetch_requests=1, prefetch_bytes=4096)
        with self.assertRaisesRegex(
            TheoreticalContractViolation, "same prefetch workload"
        ):
            validate_theoretical_contract(rows)

    def test_safe_must_dominate_every_implementable_candidate(self):
        rows = self.matrix()
        static_pf_index = next(i for i, row in enumerate(rows)
                               if row.baseline == Baseline.STATIC_NAIVEPF.value)
        rows[static_pf_index] = self.row(Baseline.STATIC_NAIVEPF, 120)
        oracle_index = next(i for i, row in enumerate(rows)
                            if row.baseline == Baseline.ORACLE.value)
        rows[oracle_index] = derive_selected_row(
            Baseline.ORACLE,
            [row for row in rows if row.baseline not in {
                Baseline.MEMDOMAIN_SAFE.value, Baseline.ORACLE.value
            }] + [next(row for row in rows
                      if row.baseline == Baseline.MEMDOMAIN_SAFE.value)],
            [Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF,
             Baseline.DYNAMIC_NOPF, Baseline.DYNAMIC_NAIVEPF,
             Baseline.MEMDOMAIN_RAW, Baseline.MEMDOMAIN_SAFE],
        )
        with self.assertRaisesRegex(
            TheoreticalContractViolation, "best implementable candidate"
        ):
            validate_theoretical_contract(rows)

    def test_dynamic_naive_baseline_is_mandatory(self):
        rows = self.matrix()
        rows.pop(3)
        with self.assertRaisesRegex(ValueError, "exactly 7"):
            validate_matrix(rows)

    def test_resource_mismatch_is_rejected(self):
        rows = self.matrix()
        rows[0] = replace(rows[0], bank_count=12)
        with self.assertRaisesRegex(ValueError, "resource budget"):
            validate_matrix(rows)

    def test_workload_hash_mismatch_is_rejected(self):
        rows = self.matrix()
        rows[0] = replace(rows[0], workload_hash="different")
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_matrix(rows)

    def test_cycle_accounting_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "accounting mismatch"):
            replace(self.matrix()[0], total_cycles=1)

    def test_safe_row_is_copied_from_real_candidate(self):
        rows = self.matrix()
        safe = next(row for row in rows if row.baseline == Baseline.MEMDOMAIN_SAFE.value)
        self.assertEqual(safe.selected_candidate, Baseline.MEMDOMAIN_RAW.value)
        self.assertEqual(safe.total_cycles, 130)

    def test_fabricated_safe_cycles_are_rejected(self):
        rows = self.matrix()
        index = next(i for i, row in enumerate(rows) if row.baseline == Baseline.MEMDOMAIN_SAFE.value)
        rows[index] = replace(rows[index], bank_stall_cycles=1, total_cycles=101)
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_matrix(rows)

    def test_fabricated_safe_behavior_metric_is_rejected(self):
        rows = self.matrix()
        index = next(i for i, row in enumerate(rows) if row.baseline == Baseline.MEMDOMAIN_SAFE.value)
        rows[index] = replace(rows[index], bank_conflict_count=999)
        with self.assertRaisesRegex(ValueError, "metrics do not match"):
            validate_matrix(rows)

    def test_safe_falls_back_when_raw_and_dynamic_lose(self):
        static = self.row(Baseline.STATIC_NOPF, 120)
        dynamic = self.row(Baseline.DYNAMIC_NOPF, 140)
        raw = self.row(Baseline.MEMDOMAIN_RAW, 150)
        safe = derive_selected_row(
            Baseline.MEMDOMAIN_SAFE, [static, dynamic, raw],
            [Baseline.STATIC_NOPF, Baseline.DYNAMIC_NOPF, Baseline.MEMDOMAIN_RAW],
        )
        self.assertEqual(safe.total_cycles, 120)
        self.assertTrue(safe.fallback_used)

    def test_csv_round_trip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            write_matrix(first, reversed(self.matrix()))
            write_matrix(second, read_matrix(first))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_workload_digest_is_key_order_independent(self):
        self.assertEqual(workload_digest({"a": 1, "b": 2}), workload_digest({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
