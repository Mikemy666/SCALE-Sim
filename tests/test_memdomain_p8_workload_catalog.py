import tempfile
import unittest
from pathlib import Path

from scalesim.memory.memdomain_runner import load_runner_config
from scalesim.memory.moe_workload_catalog import (
    generate_runner_payload,
    load_catalog,
    write_runner_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/MoE/MoE_prefetch1/workloads/catalog.json"


class MoEWorkloadCatalogTests(unittest.TestCase):
    def setUp(self):
        self.specs = load_catalog(CATALOG)
        self.by_id = {spec.model_id: spec for spec in self.specs}

    def test_catalog_contains_homogeneous_and_heterogeneous_networks(self):
        classes = {spec.architecture_class for spec in self.specs}
        self.assertIn("homogeneous", classes)
        self.assertIn("heterogeneous_size", classes)
        self.assertIn("shared_routed", classes)

    def test_homogeneous_experts_are_equal(self):
        for model_id in ("switch-base-8", "mixtral-8x7b-v0.1"):
            self.assertEqual(len(set(self.by_id[model_id].routed_expert_intermediate_sizes)), 1)

    def test_modse_experts_have_published_size_diversity(self):
        sizes = self.by_id["modse-300m-8"].routed_expert_intermediate_sizes
        self.assertEqual(len(sizes), 8)
        self.assertGreater(len(set(sizes)), 2)
        self.assertEqual(sizes[:2], (6912, 768))

    def test_deepseek_keeps_routed_and_shared_structure_separate(self):
        spec = self.by_id["deepseek-moe-16b"]
        self.assertEqual(len(spec.routed_expert_intermediate_sizes), 64)
        self.assertEqual(spec.shared_expert_intermediate_sizes, (2816,))
        self.assertEqual(spec.top_k, 6)

    def test_generated_workload_is_explicitly_derived(self):
        payload = generate_runner_payload(self.by_id["mixtral-8x7b-v0.1"])
        provenance = payload["model_provenance"]
        self.assertTrue(provenance["derived_workload"])
        self.assertFalse(provenance["paper_scale_performance_claim"])
        self.assertTrue(provenance["batch_capacity_inflated"])
        self.assertEqual(provenance["original_hidden_size"], 4096)
        self.assertLess(provenance["scaled_hidden_size"], 4096)

    def test_scaling_preserves_equal_and_unequal_classes(self):
        homogeneous = generate_runner_payload(self.by_id["switch-base-8"])
        heterogeneous = generate_runner_payload(self.by_id["modse-300m-8"])
        self.assertEqual(len(set(homogeneous["model_provenance"]["scaled_routed_expert_intermediate_sizes"])), 1)
        self.assertGreater(len(set(heterogeneous["model_provenance"]["scaled_routed_expert_intermediate_sizes"])), 1)

    def test_generated_payload_round_trips_into_runner(self):
        payload = generate_runner_payload(self.by_id["switch-base-8"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            write_runner_payload(path, payload)
            config = load_runner_config(path)
            self.assertEqual(config.workload_name, "switch-base-8")
            self.assertGreater(len(config.chunks), 0)

    def test_generation_is_byte_deterministic(self):
        payload = generate_runner_payload(self.by_id["modse-300m-8"])
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.json"
            second = Path(directory) / "b.json"
            write_runner_payload(first, payload)
            write_runner_payload(second, payload)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
