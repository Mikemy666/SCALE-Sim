"""DATE3 normalized Token trace export and reconstruction contracts."""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scalesim.memory.date3_ep_model import localize_detailed_npu
from scalesim.memory.memdomain_runner import load_runner_config
from scripts.DATE3.token_trace import build_token_trace, export_token_trace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/MoE/DATE3"


class Date3TokenTraceTests(unittest.TestCase):
    def workload(self, relative):
        path = CONFIG / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload, localize_detailed_npu(load_runner_config(path))

    def test_full_trace_is_lossless_and_self_consistent(self):
        payload, detailed = self.workload("overall/HMoE.json")
        routes, stages, index, sample, summary = build_token_trace(payload, detailed)
        self.assertTrue(summary["all_checks_pass"], summary["checks"])
        self.assertEqual(len(routes), len(detailed.routes))
        self.assertEqual(len(index), len(routes))
        self.assertEqual(len(stages), len(detailed.contract.stages))
        self.assertTrue(sample)
        self.assertTrue(all(row["ffn1_stage_id"] and row["ffn2_stage_id"]
                            for row in index))

    def test_multilayer_trace_exposes_layer_and_layer_local_expert(self):
        payload, detailed = self.workload("end_to_end/HMoE.json")
        routes, _, _, _, summary = build_token_trace(payload, detailed)
        self.assertTrue(summary["all_checks_pass"], summary["checks"])
        self.assertEqual(summary["dimensions"]["layers"], 4)
        self.assertEqual(summary["dimensions"]["experts_per_layer"], 8)
        self.assertEqual({row["layer_id"] for row in routes}, {0, 1, 2, 3})
        self.assertTrue(all(0 <= row["layer_expert_id"] < 8 for row in routes))

    def test_export_writes_compressed_full_tables_and_readable_sample(self):
        payload, detailed = self.workload("overall/HMoE.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = export_token_trace(output, payload, detailed, mode="full")
            self.assertTrue(summary["all_checks_pass"])
            for name in (
                "TOKEN_ROUTE_TRACE.csv.gz", "TOKEN_STAGE_TRACE.csv.gz",
                "TOKEN_TRACE_INDEX.csv.gz", "TOKEN_TRACE_SAMPLE.csv",
                "TOKEN_TRACE_SUMMARY.json",
            ):
                self.assertTrue((output / name).exists(), name)
            with gzip.open(output / "TOKEN_TRACE_INDEX.csv.gz", "rt",
                           newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(detailed.routes))


if __name__ == "__main__":
    unittest.main()
