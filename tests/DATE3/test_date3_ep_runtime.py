"""DATE3 P0-P3 Expert-Parallel data-flow contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from scalesim.memory.date3_ep_model import (
    EPContract, REQUEST_STAGE, localize_detailed_npu,
)
from scalesim.memory.memdomain_runner import _communication_stall, load_runner_config
from scalesim.memory.pivot_ca_runner import run_pivot_ca_file
from scalesim.memory.pivot_ca_runner import _epoch_groups
from scalesim.memory.date3_ep_system import run_date3_ep_baseline_matrix


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/MoE/DATE3"


class Date3EPContractTests(unittest.TestCase):
    def config(self, relative: str):
        return load_runner_config(CONFIG / relative)

    def test_overall_uses_two_npus_and_unique_complete_ownership(self):
        for model in ("HMoE", "Mixtral", "MoDSE", "Switchtrans"):
            contract = EPContract.from_payload(
                self.config(f"overall/{model}.json").payload
            )
            self.assertEqual(contract.num_npus, 2)
            self.assertEqual(len(contract.owner_by_expert), contract.num_experts)
            self.assertEqual(set(contract.owner_by_expert), {0, 1})
            self.assertEqual(sum(contract.owner_by_expert.count(npu)
                                 for npu in range(contract.num_npus)),
                             contract.num_experts)

    def test_global_top1_routes_close_exact_expert_counts(self):
        contract = EPContract.from_payload(self.config("overall/HMoE.json").payload)
        routes = contract.routes()
        observed = [0] * contract.num_experts
        for route in routes:
            observed[route.global_expert_id] += 1
            self.assertEqual(route.owner_npu,
                             contract.owner_by_expert[route.global_expert_id])
        self.assertEqual(tuple(observed), contract.token_counts)
        self.assertEqual(len(routes), sum(contract.token_counts))

    def test_top2_has_two_distinct_replicas_per_token(self):
        contract = EPContract.from_payload(
            self.config("robustness_factorial/top_k__HMoE__2.json").payload
        )
        routes = contract.routes()
        by_token = {}
        for route in routes:
            by_token.setdefault(route.token_id, []).append(route)
        self.assertEqual(len(routes), 2 * len(by_token))
        for replicas in by_token.values():
            self.assertEqual({item.topk_slot for item in replicas}, {0, 1})
            self.assertEqual(len({item.global_expert_id for item in replicas}), 2)
            self.assertTrue(all(item.routing_weight == 0.5 for item in replicas))

    def test_detailed_workload_contains_only_owned_active_experts(self):
        workload = localize_detailed_npu(self.config("overall/HMoE.json"))
        allowed = set(workload.active_local_experts)
        self.assertTrue(allowed)
        self.assertTrue(all(chunk.expert_id in allowed
                            for chunk in workload.config.chunks))
        for request in workload.config.compute_requests:
            match = REQUEST_STAGE.search(request.request_id)
            if match:
                self.assertIn(int(match.group(1)), allowed)
        self.assertFalse(any(chunk.expert_id not in allowed
                             for chunk in workload.config.chunks))

    def test_detailed_workload_compacts_out_remote_stage_time(self):
        original = self.config("overall/HMoE.json")
        local = localize_detailed_npu(original)
        self.assertLess(local.config.compute_cycles, original.compute_cycles)
        self.assertLess(len(local.config.chunks), len(original.chunks))
        self.assertEqual(local.config.payload["ep_runtime"]["local_experts"],
                         list(local.local_experts))

    def test_ep1_has_no_remote_routes_or_communication(self):
        workload = localize_detailed_npu(self.config(
            "robustness_factorial/expert_parallel__HMoE__1.json"
        ))
        self.assertEqual(workload.remote_route_replicas, 0)
        self.assertEqual(_communication_stall(workload.config), 0)

    def test_ep2_communication_uses_exact_remote_route_replicas(self):
        workload = localize_detailed_npu(self.config(
            "robustness_factorial/expert_parallel__HMoE__2.json"
        ))
        system = workload.config.payload["system"]
        expected = 20 + (
            workload.remote_route_replicas * system["token_payload_bytes"]
            + system["communication_bandwidth_bytes_per_cycle"] - 1
        ) // system["communication_bandwidth_bytes_per_cycle"]
        self.assertGreater(workload.remote_route_replicas, 0)
        self.assertEqual(_communication_stall(workload.config), expected)

    def test_date3_provenance_is_date3_local(self):
        payload = json.loads((CONFIG / "overall/HMoE.json").read_text())
        self.assertIn("/DATE3/", payload["topology_provenance"]["source_path"])
        self.assertNotIn("/DATE2/", payload["topology_provenance"]["source_path"])

    def test_runner_exports_routes_and_local_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = run_pivot_ca_file(
                CONFIG / "unit_cases/MoDSE_minimal.json", Path(directory)
            )
            self.assertGreater(len(execution.routes), 0)
            self.assertEqual(execution.summary["num_npus"], 2)
            self.assertTrue((Path(directory) / "ep_routes.csv").exists())
            self.assertTrue((Path(directory) / "ep_local_workload.csv").exists())
            self.assertTrue((Path(directory) / "ep_peer_workloads.csv").exists())
            self.assertTrue((Path(directory) / "ep_timeline.csv").exists())
            self.assertTrue((Path(directory) / "ep_return_combine.csv").exists())
            self.assertTrue((Path(directory) / "online_incumbent_guard.csv").exists())

    def test_online_guard_applies_minimum_prefix_cost_before_issue(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = run_pivot_ca_file(
                CONFIG / "unit_cases/MoDSE_minimal.json", Path(directory),
            )
            self.assertTrue(execution.guard_rows)
            for row in execution.guard_rows:
                costs = {
                    "adaptive": int(row["proposal_prefix_cost_cycles"]),
                    "fixed_incumbent": int(row["fixed_prefix_cost_cycles"]),
                    "noprefetch_incumbent": int(
                        row["noprefetch_prefix_cost_cycles"]
                    ),
                }
                self.assertEqual(
                    int(row["applied_prefix_cost_cycles"]), min(costs.values())
                )
                self.assertEqual(
                    int(row["incumbent_prefix_cost_cycles"]), min(costs.values())
                )
                self.assertEqual(costs[row["applied_action"]], min(costs.values()))

    def test_peer_path_executes_only_peer_owned_active_experts(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = run_pivot_ca_file(
                CONFIG / "unit_cases/MoDSE_minimal.json", Path(directory),
            )
        contract = EPContract.from_payload(
            self.config("unit_cases/MoDSE_minimal.json").payload
        )
        for row in execution.peer_workloads:
            self.assertNotEqual(row["npu_id"], contract.detailed_npu_id)
            self.assertEqual(contract.owner_by_expert[int(row["expert_id"])],
                             int(row["npu_id"]))
            self.assertGreater(int(row["token_count"]), 0)
            self.assertGreater(int(row["weight_bytes"]), 0)
            self.assertGreater(int(row["weight_load_cycles"]), 0)

    def test_dispatch_return_and_critical_path_are_dependency_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = run_pivot_ca_file(
                CONFIG / "unit_cases/MoDSE_minimal.json", Path(directory),
            )
        summary = execution.summary
        config = self.config("unit_cases/MoDSE_minimal.json")
        system = config.payload["system"]
        remote = int(summary["remote_route_replicas"])
        self.assertEqual(summary["dispatch_bytes"],
                         remote * int(system["token_payload_bytes"]))
        self.assertEqual(summary["return_bytes"],
                         remote * int(system["result_payload_bytes"]))
        self.assertEqual(summary["result_ready_cycle"], max(
            summary["detailed_ready_cycle"], summary["peer_ready_cycle"]
        ))
        self.assertEqual(summary["total_cycles"],
                         summary["result_ready_cycle"] + summary["combine_cycles"])

    def test_top2_combine_waits_for_two_weighted_results(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = run_pivot_ca_file(
                CONFIG / "robustness_factorial/top_k__HMoE__2.json",
                Path(directory),
            )
            self.assertTrue(execution.combine_rows)
            for row in execution.combine_rows:
                self.assertEqual(int(row["expected_results"]), 2)
                weights = [float(value) for value in
                           str(row["routing_weights"]).split("|")]
                self.assertAlmostEqual(sum(weights), 1.0)

    def test_prefetch_epoch_never_crosses_expert_or_ffn_stage(self):
        localized = localize_detailed_npu(self.config("overall/HMoE.json"))
        groups = _epoch_groups(localized.config, 32)
        self.assertTrue(groups)
        for group in groups:
            self.assertEqual(len({(item.expert_id, item.ffn_part)
                                  for item in group}), 1)

    def test_date3_controls_use_the_same_ep_critical_path(self):
        rows = run_date3_ep_baseline_matrix(
            self.config("unit_cases/MoDSE_minimal.json")
        )
        self.assertEqual(len(rows), 7)
        payload = self.config("unit_cases/MoDSE_minimal.json").payload
        combine = (sum(payload["ep"]["token_counts"])
                   // payload["ep"]["top_k"]
                   * payload["system"]["combine_cycles_per_token"])
        for row in rows:
            self.assertEqual(row.other_stall_cycles, combine)
            self.assertEqual(row.total_cycles, sum((
                row.compute_cycles, row.bank_stall_cycles,
                row.weight_load_stall_cycles, row.prefetch_miss_stall_cycles,
                row.prefetch_interference_stall_cycles,
                row.mapping_overhead_cycles, row.communication_stall_cycles,
                row.other_stall_cycles,
            )))


if __name__ == "__main__":
    unittest.main()
