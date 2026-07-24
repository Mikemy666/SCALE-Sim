# MoE_prefetch1 configurations

This namespace is reserved for the redesigned simulator architecture. DATE1
configuration files are reference inputs only and must not be modified or
copied here without an explicit provenance note.

Planned subdirectories:

- `baseline/`: frozen static, best-static, oracle, and runtime-dynamic controls;
- `characterization/`: B1 and C1-C3 motivation experiments;
- component ablation reuses DATE2 `overall/` matrices;
- `sensitivity/`: window, chunk, routing, token, expert, and EP sweeps;
- `workloads/`: per-model overrides for homogeneous and heterogeneous experts.

Every configuration must use a unique `run_name` and write beneath
`outputs/MoE_prefetch1/` through the dedicated runner.
