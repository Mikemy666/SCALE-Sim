"""Contracts specific to the non-stationary, multi-layer Exp5 workload."""

import json
import unittest
from pathlib import Path

from scalesim.memory.memdomain_runner import (
    _atomic_noprefetch_config,
    _fixed_issue_schedule,
    critical_path_miss_stalls,
    load_runner_config,
)
from scalesim.memory.streaming_residency import StreamingChunkResult


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/MoE/DATE3/ablation/MoDSE__full.json"


def result(chunk_id, use_cycle, stall, classification):
    return StreamingChunkResult(
        chunk_id=chunk_id,
        planned_kind="prefetch",
        effective_kind="prefetch",
        planned_issue_cycle=0,
        actual_issue_cycle=0,
        completion_cycle=use_cycle + stall,
        use_cycle=use_cycle,
        consume_cycle=use_cycle + stall,
        release_cycle=use_cycle + stall,
        allocation_wait_cycles=0,
        miss_stall_cycles=stall,
        classification=classification,
        physical_banks=(0,),
    )


class Exp5MultiLayerTests(unittest.TestCase):
    def test_exp5_uses_one_persistent_controller_across_four_layers(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        contract = payload["multi_layer_prefetch"]
        self.assertEqual(contract["layer_count"], 4)
        self.assertEqual(contract["controller_state"],
                         "persistent_across_layers")
        self.assertEqual(len(contract["profiles"]), 4)
        self.assertEqual(payload["ep"]["num_experts"], 32)
        owners = payload["ep"]["expert_owner_map"]
        for offset in range(0, 32, 8):
            self.assertEqual(owners[offset:offset + 8], [0] * 4 + [1] * 4)

    def test_exp5_uses_relative_quality_feedback_without_unreachable_floor(self):
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        policy = payload["coverage_accuracy_policy"]
        self.assertEqual(policy["reference_mode"], "shadow_fixed")
        self.assertEqual(policy["min_coverage"], 0)
        self.assertEqual(policy["min_accuracy"], 0)
        self.assertEqual(policy["epsilon_coverage"], 1)
        self.assertEqual(policy["epsilon_accuracy"], 1)
        self.assertGreater(policy["eta_coverage"], 0)
        self.assertGreater(policy["eta_accuracy"], 0)

    def test_multi_layer_stalls_are_critical_path_union(self):
        chunks = (
            result("a", 10, 10, "late"),       # [10, 20]
            result("b", 15, 10, "late"),       # [15, 25]
            result("c", 30, 5, "demand_miss"), # [30, 35]
        )
        # Additive legacy accounting remains unchanged for Exp1--Exp4.
        self.assertEqual(critical_path_miss_stalls(chunks, multi_layer=False),
                         (5, 20))
        demand, late = critical_path_miss_stalls(chunks, multi_layer=True)
        self.assertEqual(demand + late, 20)
        self.assertEqual((demand, late), (4, 16))

    def test_fixed_prefetch_cannot_cross_a_router_boundary(self):
        config = load_runner_config(CONFIG)
        schedule = _fixed_issue_schedule(config, window=8)
        contract = config.payload["multi_layer_prefetch"]
        starts = {
            int(item["layer_id"]): int(item["start_cycle"])
            for item in config.payload["topology_provenance"]["layer_profiles"]
        }
        for chunk in config.chunks:
            layer = chunk.expert_id // int(contract["experts_per_layer"])
            self.assertGreaterEqual(schedule[chunk.chunk_id], starts[layer])

    def test_noprefetch_demand_granularity_is_grid_invariant(self):
        c8 = _atomic_noprefetch_config(load_runner_config(CONFIG))
        c4_path = ROOT / "configs/MoE/DATE3/joint_prefetch/w2_c4.json"
        c4 = _atomic_noprefetch_config(load_runner_config(c4_path))
        self.assertEqual(len(c8.chunks), len(c4.chunks))
        self.assertEqual(sum(item.size_bytes for item in c8.chunks),
                         sum(item.size_bytes for item in c4.chunks))
        self.assertEqual({item.size_bytes for item in c8.chunks}, {256})


if __name__ == "__main__":
    unittest.main()
