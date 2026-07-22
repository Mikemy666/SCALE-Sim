import csv
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOPOLOGIES = {
    "HMoE": ROOT / "topologies/MoE/HMoE.csv",
    "Mixtral": ROOT / "topologies/MoE/Mixtral.csv",
    "MoDSE": ROOT / "topologies/MoE/MoDSE.csv",
    "Switchtrans": ROOT / "topologies/MoE/Switchtrans.csv",
}
EXPECTED_TOP1_COUNTS = (32, 48, 50, 24, 34, 28, 21, 19)


def expert_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    parsed = {}
    for row in rows:
        match = re.fullmatch(r"MoE-E(\d+)-FF([12])", row[0]) if row else None
        if match:
            parsed[(int(match.group(1)), int(match.group(2)))] = tuple(map(int, row[1:4]))
    return parsed


class MoETopologyControlTests(unittest.TestCase):
    def test_all_models_use_identical_top1_token_counts(self):
        for name, path in TOPOLOGIES.items():
            rows = expert_rows(path)
            counts = tuple(rows[(expert, 1)][0] for expert in range(8))
            self.assertEqual(counts, EXPECTED_TOP1_COUNTS, name)
            self.assertEqual(sum(counts), 256, name)
            for expert, count in enumerate(counts):
                self.assertEqual(rows[(expert, 2)][0], count, name)

    def test_two_homogeneous_and_two_heterogeneous_models_are_preserved(self):
        homogeneous = {"Mixtral", "Switchtrans"}
        for name, path in TOPOLOGIES.items():
            rows = expert_rows(path)
            ff1_shapes = {rows[(expert, 1)][1:] for expert in range(8)}
            if name in homogeneous:
                self.assertEqual(len(ff1_shapes), 1, name)
            else:
                self.assertGreater(len(ff1_shapes), 1, name)


if __name__ == "__main__":
    unittest.main()
