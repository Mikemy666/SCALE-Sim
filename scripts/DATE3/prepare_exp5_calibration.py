"""Build the independent calibration set for deployable Exp5 fixed prefetch.

The evaluated MoDSE trace is never used to select Window/Chunk.  A fixed
prefetch implementation is profiled on two distinct routing distributions,
then one pair is frozen for the paper test trace.  The test-trace grid remains
available only as an analysis-only oracle.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.date3_ep_model import attach_ep_contract
from scalesim.memory.topology_workload import generate_topology_runner_payload
from scripts.DATE3.prepare_date3_experiments import POLICY, save

CONFIG_ROOT = ROOT / "configs/MoE/DATE3"
TOPOLOGY_ROOT = ROOT / "topologies/MoE/DATE3/robustness_factorial/routing_seed"
WINDOWS = (1, 2, 4, 8, 16, 32, 64)
CHUNKS = (1, 2, 4, 8)

# These traces are calibration-only and are disjoint from the base MoDSE
# evaluation trace used by configs/MoE/DATE3/joint_prefetch.
CALIBRATION_TRACES = ("light_seed40", "high_seed41")


def payload(trace: str, window: int, chunk: int):
    topology = TOPOLOGY_ROOT / f"MoDSE__{trace}.csv"
    value = dict(generate_topology_runner_payload(
        topology,
        "heterogeneous",
        chunk_size_bytes=chunk * 256,
        top_k=1,
        num_gpus=2,
    ))
    variant = f"{trace}__w{window}_c{chunk}"
    value["experiment_id"] = f"date3-prefetch-calibration-{variant}"
    value["date3_suite"] = "prefetch_calibration"
    value["date3_variant"] = variant
    value["policy"]["prefetch_window"] = window
    value["policy"]["prefetch_policy"] = "coverage_accuracy_constrained"
    value["coverage_accuracy_policy"] = copy.deepcopy(POLICY)
    value["coverage_accuracy_policy"].update({
        "reference_window": window,
        "reference_chunk": chunk,
    })
    value["sweep"] = {
        "variable": "deployable_prefetch_calibration",
        "model": "MoDSE",
        "trace": trace,
        "window": window,
        "chunk_tiles": chunk,
        "role": "calibration_only",
    }
    value["paper_control_contract"] = {
        "selection_protocol": "independent_calibration_then_freeze",
        "calibration_only": True,
        "test_trace_visible_during_selection": False,
        "fixed_prefetch_reference": {
            "window": window,
            "chunk_tiles": chunk,
            "same_workload_for_static_and_dynamic": True,
        },
    }
    return attach_ep_contract(value)


def main() -> None:
    target = CONFIG_ROOT / "prefetch_calibration"
    for trace in CALIBRATION_TRACES:
        for window in WINDOWS:
            for chunk in CHUNKS:
                name = f"{trace}__w{window}_c{chunk}.json"
                save(target / name, payload(trace, window, chunk))

    manifest_path = CONFIG_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["suites"]["prefetch_calibration"] = (
        len(CALIBRATION_TRACES) * len(WINDOWS) * len(CHUNKS)
    )
    manifest["exp5_fixed_prefetch_selection"] = {
        "protocol": "independent_calibration_then_freeze",
        "calibration_traces": list(CALIBRATION_TRACES),
        "evaluation_trace": "MoDSE_base",
        "test_trace_visible_during_selection": False,
        "selection_objective": "mean_normalized_cycles",
        "static_and_dynamic_selected_independently": True,
        "test_grid_role": "analysis_only_oracle",
    }
    save(manifest_path, manifest)
    print(f"generated {manifest['suites']['prefetch_calibration']} calibration configs")


if __name__ == "__main__":
    main()
