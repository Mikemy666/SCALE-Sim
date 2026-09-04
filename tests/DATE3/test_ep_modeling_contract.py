"""Low-cost contract tests for the EP audit; no simulator state is mutated."""

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validate_ep_modeling", ROOT / "scripts/DATE3/validate_ep_modeling.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


class EPModelingContractTests(unittest.TestCase):
    def test_a_single_npu(self):
        owners = AUDIT.scale_sim_uniform_owner_map(4, 1)
        AUDIT.validate_ownership(owners, 4, 1)
        routes = [[0], [1], [2], [3]]
        self.assertEqual(AUDIT.communication_bytes(routes, owners, 0, 96), 0)

    def test_b_uniform_two_npu_reference_and_date3_contract(self):
        owners = AUDIT.scale_sim_uniform_owner_map(8, 2)
        self.assertEqual(owners, {0: 0, 1: 0, 2: 0, 3: 0,
                                  4: 1, 5: 1, 6: 1, 7: 1})
        AUDIT.validate_ownership(owners, 8, 2)
        payload = json.loads((ROOT / "configs/MoE/DATE3/robustness_factorial"
                              / "expert_parallel__HMoE__2.json").read_text())
        self.assertEqual(payload["system"]["num_gpus"], 2)
        self.assertEqual(payload["ep"]["expert_owner_map"], list(owners.values()))
        self.assertEqual(payload["ep"]["num_experts"], 8)

    def test_c_sixteen_experts_has_no_four_per_npu_assumption(self):
        owners = AUDIT.scale_sim_uniform_owner_map(16, 2)
        self.assertEqual(list(owners.values()).count(0), 8)
        self.assertEqual(list(owners.values()).count(1), 8)
        payload = json.loads((ROOT / "configs/MoE/DATE3/robustness_factorial"
                              / "expert_count__HMoE__16.json").read_text())
        self.assertEqual(len({item["expert_id"] for item in payload["chunks"]}), 16)

    def test_d_non_divisible_is_extension_only(self):
        with self.assertRaises(ValueError):
            AUDIT.scale_sim_uniform_owner_map(10, 3)
        owners = AUDIT.balanced_owner_map(10, 3)
        AUDIT.validate_ownership(owners, 10, 3)
        self.assertEqual(sorted(list(owners.values()).count(i) for i in range(3)),
                         [3, 3, 4])

    def test_e_top2_creates_two_global_replicas_and_more_remote_bytes(self):
        owners = AUDIT.scale_sim_uniform_owner_map(8, 2)
        top1 = [[0], [1], [2], [3]]
        top2 = [[0, 4], [1, 5], [2, 6], [3, 7]]
        AUDIT.validate_routes(top2, 8, 2)
        self.assertEqual(sum(map(len, top2)), 2 * len(top2))
        self.assertGreater(AUDIT.communication_bytes(top2, owners, 0, 96),
                           AUDIT.communication_bytes(top1, owners, 0, 96))
        payload = json.loads((ROOT / "configs/MoE/DATE3/robustness_factorial"
                              / "top_k__HMoE__2.json").read_text())
        provenance = payload["topology_provenance"]
        self.assertEqual(provenance["top_k"], 2)
        self.assertEqual(sum(provenance["token_counts"]),
                         2 * provenance["total_tokens"])
        self.assertNotIn("routes", payload)

    def test_f_heterogeneous_experts_preserve_metrics(self):
        topology = AUDIT.load_topology(
            ROOT / "topologies/MoE/DATE3/models/HMoE.csv"
        )
        metrics = [AUDIT.expert_metrics(stages)
                   for stages in topology["experts"].values()]
        self.assertGreater(len({item["parameter_bytes"] for item in metrics}), 1)
        self.assertGreater(len({item["runtime_macs"] for item in metrics}), 1)

    def test_four_model_router_dimensions_match_real_expert_counts(self):
        for model in AUDIT.MODELS:
            topology = AUDIT.load_topology(AUDIT.MODEL_ROOT / f"{model}.csv")
            self.assertEqual(topology["router_n"], len(topology["experts"]))

    def test_date3_pivot_ep_contract_is_present(self):
        for model in AUDIT.MODELS:
            payload = json.loads((AUDIT.CONFIG_ROOT / f"{model}.json").read_text())
            self.assertEqual(payload["system"]["num_gpus"], 2)
            self.assertEqual(payload["ep"]["num_npus"], 2)
            self.assertEqual(len(payload["ep"]["expert_owner_map"]),
                             payload["ep"]["num_experts"])


if __name__ == "__main__":
    unittest.main()
