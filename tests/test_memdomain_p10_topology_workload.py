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
            self.assertEqual(provenance["original_model_format"], "FP32")
            self.assertEqual(provenance["compute_format"], "INT8xINT8_INT32")
            self.assertEqual(provenance["ia_bytes_per_element"], 1)
            self.assertEqual(provenance["weight_bytes_per_element"], 1)
            self.assertEqual(provenance["accumulator_bytes_per_element"], 4)
            self.assertEqual(provenance["output_bytes_per_element"], 1)
            self.assertEqual(provenance["accumulator_mode"], "local")
            self.assertEqual(provenance["weight_scale_divisor"], 1)
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

    def test_int32_accumulator_is_local_unless_spill_is_explicit(self):
        path = ROOT / "topologies/MoE/MoDSE.csv"
        local = generate_topology_runner_payload(path, "heterogeneous")
        self.assertFalse(any(
            item["tensor_type"] == "accumulator"
            for item in local["compute_requests"]
        ))
        self.assertTrue(any(
            item["tensor_type"] == "oa"
            for item in local["compute_requests"]
        ))
        spill = generate_topology_runner_payload(
            path, "heterogeneous", accumulator_mode="spill"
        )
        accumulator = [
            item for item in spill["compute_requests"]
            if item["tensor_type"] == "accumulator"
        ]
        self.assertGreater(len(accumulator), 0)
        self.assertEqual(
            spill["topology_provenance"]["accumulator_bytes_per_element"], 4
        )


if __name__ == "__main__":
    unittest.main()
