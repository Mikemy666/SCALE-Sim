"""Source-level replacements for the missing cached historical smoke tests."""

import configparser
import unittest
from pathlib import Path

from scalesim.scale_config import scale_config


ROOT = Path(__file__).resolve().parents[1]


class MoEPrefetch1RegressionContracts(unittest.TestCase):
    def test_reference_config_is_parseable_and_round_trips_core_ep_fields(self):
        config_path = ROOT / "configs/MoE/DATE1/exp5/dynamic_prefetch.cfg"
        config = scale_config()
        config.read_conf_file(str(config_path))
        self.assertTrue(config.get_enable_ep_moe())
        self.assertGreater(config.get_num_gpus(), 0)
        self.assertGreater(config.get_experts_per_gpu(), 0)
        self.assertIn(config.get_top_k(), (1, 2))

    def test_reference_experiment_run_names_are_unique_within_each_group(self):
        root = ROOT / "configs/MoE/DATE1"
        for group in sorted(path for path in root.iterdir() if path.is_dir()):
            names = []
            for path in sorted(group.glob("*.cfg")):
                parser = configparser.ConfigParser()
                parser.read(path)
                names.append(parser.get("general", "run_name"))
            self.assertEqual(len(names), len(set(names)), group.name)

    def test_new_namespace_does_not_contain_generated_results(self):
        for relative in (
            "configs/MoE/MoE_prefetch1",
            "topologies/MoE/MoE_prefetch1",
            "scripts/MoE_prefetch1",
        ):
            for path in (ROOT / relative).rglob("*"):
                if path.is_file():
                    self.assertNotIn(path.suffix.lower(), {".csv", ".pdf", ".png"})


if __name__ == "__main__":
    unittest.main()
