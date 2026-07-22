# P8 literature-backed MoE workload catalog

The catalog contains both equal-weight and unequal-weight expert networks:

- Switch Transformer Base-8: homogeneous, top-1;
- Mixtral 8x7B: homogeneous, top-2;
- MoDSE 300M x 8: heterogeneous expert sizes from the published Table 2 pairs;
- DeepSeekMoE 16B: equal routed experts plus a distinct shared-expert path.

Every entry records a primary paper or official model-config URL and a source
locator. Generated simulator workloads retain original and scaled dimensions,
the dimension divisor, alignment, architecture class, routing parameters, and
an explicit `derived_workload=true` marker. They also set
`paper_scale_performance_claim=false`: scaled workloads may support mechanism
and sensitivity studies but are not passed off as full-model performance.

P8 generated workloads also set `batch_capacity_inflated=true`. The P7 batch
runner retains every Chunk mapping until transfer finalization, so generated
capacity is at least four times total scaled weight bytes to keep the declared
8/24 static Weight partition feasible. These runs validate architecture paths
and model diversity only. Fixed-capacity paper experiments require the later
event-driven streaming/release runner.

The generator emits the P7 JSON schema and is deterministic. Full-scale traces
can reuse the same provenance fields with `dimension_divisor=1`, subject to the
streaming-capacity integration required for weights larger than the on-chip
domain.
