import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scalesim.memory.memdomain_experiment import (
    Baseline,
    ExperimentRow,
    derive_selected_row,
    workload_digest,
    write_matrix,
)
from scripts.DATE2.validate_date2_contracts import audit


class Date2ContractAuditTests(unittest.TestCase):
    @staticmethod
    def row(payload, baseline, total):
        return ExperimentRow(
            1, "overall", "model", workload_digest(payload), baseline.value,
            "measured", 24, 1024 * 1024, 384.0, 1, 32,
            0, total, 0, 0, 0, 0, 0, 0, total,
        )

    def write_variant(self, root, name, full_cycles=60):
        payload = {"name": name, "policy": {"prefetch_window": 2}}
        config = root / "configs/MoE/DATE2/overall" / f"{name}.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(payload), encoding="utf-8")
        static = self.row(payload, Baseline.STATIC_NOPF, 200)
        static_pf = self.row(payload, Baseline.STATIC_NAIVEPF, 160)
        dynamic = self.row(payload, Baseline.DYNAMIC_NOPF, 190)
        dynamic_pf = self.row(payload, Baseline.DYNAMIC_NAIVEPF, 150)
        raw = self.row(payload, Baseline.MEMDOMAIN_RAW, full_cycles)
        safe = replace(
            raw, baseline=Baseline.MEMDOMAIN_SAFE.value,
            selected_candidate="Online-Guarded-Full",
        )
        oracle = derive_selected_row(
            Baseline.ORACLE,
            [static, static_pf, dynamic, dynamic_pf, raw, safe],
            [Baseline.STATIC_NOPF, Baseline.STATIC_NAIVEPF,
             Baseline.DYNAMIC_NOPF, Baseline.DYNAMIC_NAIVEPF,
             Baseline.MEMDOMAIN_RAW, Baseline.MEMDOMAIN_SAFE],
        )
        output = root / "outputs/DATE2/overall" / f"{name}.csv"
        write_matrix(
            output, [static, static_pf, dynamic, dynamic_pf, raw, safe, oracle]
        )

    def test_four_model_gain_and_geomean_contract_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("HMoE", "Mixtral", "MoDSE", "Switchtrans"):
                self.write_variant(root, name)
            result = audit(root, suites=("overall",))
            self.assertEqual(result["matrix_count"], 4)
            self.assertGreater(result["min_model_gain"], 0.05)
            self.assertGreater(result["geomean_gain"], 0.10)

    def test_per_model_gain_below_five_percent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("HMoE", "Mixtral", "MoDSE"):
                self.write_variant(root, name)
            self.write_variant(root, "Switchtrans", full_cycles=148)
            with self.assertRaisesRegex(ValueError, "below 5.00%"):
                audit(root, suites=("overall",))

    def test_stale_workload_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("HMoE", "Mixtral", "MoDSE", "Switchtrans"):
                self.write_variant(root, name)
            path = root / "configs/MoE/DATE2/overall/HMoE.json"
            path.write_text(json.dumps({"changed": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale workload hash"):
                audit(root, suites=("overall",))


if __name__ == "__main__":
    unittest.main()
