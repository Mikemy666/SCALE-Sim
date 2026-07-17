# P17: Final acceptance and handoff

## Implemented architecture

The legacy SCALE-Sim path remains the base simulator. EP-MoE adds fixed
expert-to-GPU placement, Top-K routing, detailed and analytical GPU boundaries,
parallel expert engines, trace-derived detailed tiles, analytical black-box
tiles, chunk-level weight loading/prefetch, private detailed bank modeling,
external background pressure, and a shared latency-plus-bandwidth interconnect.

Static and legacy dynamic bank allocation remain in the original memory model.
The dynamic overhead selection remains `old_model`.

## Primary files

- `scalesim/simulator.py`: EP execution plan, runtime coordinator, reports.
- `scalesim/scale_config.py`: validated EP configuration and round-trip output.
- `scalesim/memory/banked_memory_system.py`: reused bank allocation/conflict and
  prefetch interference model.
- `configs/MoE/`: default, four canonical experiments, and Top-K=2 validation.
- `topologies/MoE/`: eight-expert default and small Top-K=2 topology.
- `run_ep_moe_experiments.py`: canonical four-run experiment matrix.
- `run_ep_moe_sanity.py`: degradation and boundary checks.
- `validate_ep_moe_reports.py`: cross-report and provenance validation.
- `tests/`: legacy golden tests and P1-P17 architecture regressions.

## Commands

Default EP run:

```sh
python -m scalesim.scale \
  -c configs/MoE/ep_default.cfg \
  -t topologies/MoE/test.csv \
  -l layouts/conv_nets/test.csv \
  -p outputs/ep_default -i gemm -s N
```

Canonical experiments, sanity, and report validation:

```sh
python run_ep_moe_experiments.py --output outputs/ep_moe_matrix
python run_ep_moe_sanity.py --output outputs/ep_moe_sanity
python validate_ep_moe_reports.py outputs/ep_moe_matrix/ep_dynamic_prefetch
python -m unittest discover -s tests -v
```

## Accepted invariants

- legacy static, dynamic, and non-bank golden results remain unchanged;
- one GPU has no black-box experts or black-box traffic;
- disabled prefetch issues no prefetch request or traffic;
- disabled background pressure adds no pressure stall;
- sequential MoE cannot finish before the corresponding parallel run;
- MoE finish equals the slowest active expert finish;
- Top-K routing conserves assignments and uses unique experts per token;
- detailed logical weight address ranges do not overlap;
- black-box layers produce no detailed SRAM/DRAM or local bank activity;
- expert/runtime/summary metrics agree;
- repeated validation runs are byte deterministic;
- input and core model source hashes are recorded.

## Deliberate limitations

- Expert placement is fixed; replication, migration, and dynamic placement are
  not modeled.
- Only one configured GPU is detailed. Other GPUs use an analytical workload.
- The interconnect is one serialized latency-plus-bandwidth resource, not a
  topology, packet, collective, or congestion network simulator.
- Black-box compute and memory timing is analytical and is not hardware
  calibrated by this project.
- Detailed chunk load readiness uses the configured byte bandwidth while bank
  interference is evaluated through the legacy bank model; it is not a full
  DRAM controller simulation.
- Dynamic bank remapping intentionally retains the old model rather than adding
  a new cycle-accurate allocator.
- Layer-oriented legacy reports contain analytical/zero-access compatibility
  rows for skipped black-box layers; EP reports are authoritative for EP timing.
- No expert replication, failure behavior, energy model, or hardware validation
  is claimed.

These limits are explicit model boundaries rather than unfinished code paths.
