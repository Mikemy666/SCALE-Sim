# P10 topology-backed overall evaluation

P10 implements paper section IV-B on the four controlled topology CSVs. HMoE
and MoDSE are heterogeneous; Mixtral and Switchtrans are homogeneous. All use
Top-1 and the identical 256-token expert-count vector
`[32, 48, 50, 24, 34, 28, 21, 19]`.

The converter preserves each CSV's matrix dimensions and relative expert weight
sizes. For tractable mechanism evaluation it scales every model's weight bytes
by the same factor of 8, materializes the result as 16 KiB Chunks, and records
`paper_scale_performance_claim=false`. It uses the P9 fixed 24 x 64 KiB physical Bank
domain. Capacity is smaller than each model's aggregate weights. The five raw
baselines are executed, while Safe and Oracle are selected from real rows under
the existing theory contract.

The Bank interleave is 1 KiB for this full-topology campaign. Total transferred
bytes, physical bandwidth, capacity, ports, and request-buffer limits are
unchanged; the coarser interleave reduces event count during the 24-way static
Bank-group search. This value is serialized in every generated workload.

Run:

```sh
.venv/bin/python scripts/MoE_prefetch1/run_p10_overall.py
```

Outputs are written under ignored `outputs/MoE_prefetch1/p10/`: four workload
JSON files, four canonical seven-row matrices, and `overall_summary.csv` with
normalized cycles, speedup, memory stalls, and Bank conflict rate.
