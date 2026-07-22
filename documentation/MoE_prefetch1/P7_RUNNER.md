# P7 canonical simulator runner

P7 executes five measured rows on one workload and derives Safe/Oracle rows
through the P6 provenance rules. Weight transfers and IA/OA/Accumulator compute
traffic share one P2 service batch, so compute-prefetch interference is measured
from completion-time displacement relative to a compute-only run.

The runner gathers cycle breakdown, physical-Bank conflicts/imbalance/hotspot/
idle/parallelism/queue depth, P4 prefetch metrics, and P3 mapping metrics. Raw
online regressions remain present. Mapping overhead is charged only to dynamic
rows and can cause Safe to select Static-NoPF.

Static rows are not a hand-picked Bank target: the runner exhaustively searches
all physical Weight Bank groups with the configured group width and selects the
lowest end-to-end cycle result. The selected group is recorded in
`candidate_source`. Compute-transfer overlap uses explicit compute intervals
from the workload JSON.

Run the smoke workload with:

```sh
.venv/bin/python scripts/MoE_prefetch1/run_matrix.py \
  configs/MoE/MoE_prefetch1/baseline/tiny_workload.json \
  --output outputs/MoE_prefetch1/p7/tiny.csv
```

P7 uses a JSON workload adapter. Later workload phases will generate this
schema from SCALE-Sim traces and literature-backed model topologies.
