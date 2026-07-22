import unittest
from pathlib import Path

from scalesim.memory.memdomain_runner import load_runner_config
from scalesim.memory.moe_workload_catalog import write_runner_payload
from scalesim.memory.topology_workload import generate_topology_runner_payload


ROOT = Path(__file__).resolve().parents[1]


class P10TopologyWorkloadTests(unittest.TestCase):
    def test_four_models_share_top1_input_controls(self):
        classes = {"HMoE": "heterogeneous", "Mixtral": "homogeneous",
                   "MoDSE": "heterogeneous", "Switchtrans": "homogeneous"}
        controls = []
        for model, model_class in classes.items():
            payload = generate_topology_runner_payload(
                ROOT / f"topologies/MoE/{model}.csv", model_class
            )
            provenance = payload["topology_provenance"]
            self.assertEqual(provenance["top_k"], 1)
            self.assertEqual(provenance["total_tokens"], 256)
            self.assertTrue(provenance["streaming_fixed_capacity"])
            self.assertEqual(provenance["weight_scale_divisor"], 8)
            self.assertFalse(provenance["paper_scale_performance_claim"])
            controls.append(tuple(provenance["token_counts"]))
        self.assertEqual(len(set(controls)), 1)

    def test_payload_is_accepted_by_runner_schema(self):
        import tempfile
        payload = generate_topology_runner_payload(
            ROOT / "topologies/MoE/Switchtrans.csv", "homogeneous"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            write_runner_payload(path, payload)
            config = load_runner_config(path)
            self.assertEqual(config.workload_name, "Switchtrans")
            self.assertGreater(len(config.chunks), 0)
            self.assertLess(config.resources.capacity_bytes,
                            sum(chunk.size_bytes for chunk in config.chunks))


if __name__ == "__main__":
    unittest.main()
