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

P9 replaces the earlier batch-capacity accommodation. Generated workloads set
`streaming_fixed_capacity=true` and use 24 Banks with 64 KiB per Bank,
independent of aggregate model weight bytes. The event-driven runner releases
each Chunk at consumption, so models larger than the on-chip domain execute by
streaming through the fixed physical capacity.

The generator emits the P7 JSON schema and is deterministic. Full-scale traces
can reuse the same provenance fields with `dimension_divisor=1`; run time and
trace size, rather than an artificially enlarged Bank capacity, are then the
practical constraints.
