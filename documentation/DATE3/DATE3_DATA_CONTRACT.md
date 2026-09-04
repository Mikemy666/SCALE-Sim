# DATE3 Data Contract

## Identity

Every DATE3 summary is bound to a canonical SHA-256 `config_hash` and a
policy-independent `workload_hash`. Resumable output metadata also contains an
`implementation_hash` over the DATE3 EP, PIVOT, mapping, and residency source
files, so changed simulator logic cannot silently reuse old results. Detail
and epoch reports are stored beside that summary under
`outputs/DATE3/<suite>/<variant>/`.

## Quality bytes

- `required_bytes`: unique first-demand bytes that were not resident at the
  corresponding decision.
- `prefetched_bytes`: unique bytes moved by real prefetch requests.
- `useful_timely_bytes`: prefetched bytes completed by first use, retained
  until first use, and actually used.
- `late_bytes`, `unused_bytes`, and `evicted_before_use_bytes` are mutually
  exclusive failure classifications and never useful.
- `coverage = useful_timely_bytes / required_bytes` only when required bytes
  are nonzero.
- `accuracy = useful_timely_bytes / prefetched_bytes` only when prefetched
  bytes are nonzero.
- Invalid ratios are empty CSV fields, not fabricated ones.

## Occupancy

`occupancy_byte_cycles` is the sum of resident bytes multiplied by residency
cycles. It is not a percentage.

## Cycles

`total_cycles` is exactly the sum of:

1. `compute_cycles`;
2. `bank_stall_cycles`;
3. `weight_load_stall_cycles`;
4. `prefetch_miss_stall_cycles`;
5. `prefetch_interference_stall_cycles`;
6. `mapping_overhead_cycles`;
7. `communication_stall_cycles`;
8. `other_stall_cycles`.

All components are non-negative. Predicted latency benefit is diagnostic and
is never subtracted from this sum.

`bank_stall_cycles` contains both compulsory expert critical-path Bank service
(IA/Weight/OA transfers and INT32 ACC read-modify-write service under the
executed legal Bank allocation) and any additional observed single-port queue
wait. Counting queue wait alone is invalid because it would make a
conflict-free trace appear to have zero on-chip memory cost.

For P>1, `detailed_ready_cycle` is the cycle-level PIVOT path and
`peer_ready_cycle` is the latest analytical Peer path. The dependency contract
is `result_ready_cycle=max(detailed_ready_cycle,peer_ready_cycle)` and
`total_cycles=result_ready_cycle+combine_cycles`. NPU times are not summed.
`communication_stall_cycles` is only the remote work exposed beyond the
Detailed path; dispatch and return transaction totals remain separate fields.

## Expert Parallelism

- `dispatch_bytes = remote_route_replicas * token_payload_bytes`.
- `return_bytes = remote_route_replicas * result_payload_bytes`.
- Every Token in `ep_routes.csv` has exactly `top_k` distinct expert replicas.
- Each `global_expert_id` maps to exactly one `owner_npu`.
- Detailed and Peer expert workloads are disjoint by owner.
- Top-k combine rows retain expert IDs, owner NPUs, routing weights, and the
  number of results that must arrive.

## Online incumbent protection

Each `online_incumbent_guard.csv` row compares the adaptive proposal, fixed
prefetch reference, and dynamic NoPF action on the same completed prefix plus
current expert/stage. Requests are committed only after this comparison.
`applied_prefix_cost_cycles` must equal the minimum of the three input costs;
`applied_action` identifies the executed path. Ties prefer prefetching so the
quality objective is not weakened for no cycle benefit.

## Reports

- `summary.csv`: one aggregate PIVOT row. Existing raw summaries may retain
  the internal engine identifier `PIVOT-CA`; aggregation normalizes it to
  the single public name `PIVOT`.
- `decision_detail.csv`: every feasible and rejected candidate, thresholds,
  selected action, and fallback provenance.
- `quality_epochs.csv`: measured byte counters, valid flags, and EMA state.
- `metadata.json`: input paths and hashes.
- `ep_routes.csv`: Token-level Top-k routing replicas.
- `ep_local_workload.csv`: Detailed-NPU local expert workload.
- `ep_peer_workloads.csv`: analytical Peer owner-local workloads.
- `ep_timeline.csv`: dispatch/work/return/combine dependencies.
- `ep_return_combine.csv`: per-Token returned-result dependency metadata.
- `online_incumbent_guard.csv`: pre-issue adaptive/fixed comparison.
- `comparison.csv`: per-variant mechanism source. Paper compatibility tables
  use six implementable names: Static-555-NoPF, Static-Opt-NoPF,
  Dynamic-NoPF, Static-Opt-FixedPF, Dynamic-FixedPF, and PIVOT.
  Ideal-NoPF is explicitly reference-only. PIVOT is MemDomain's official name;
  MemDomain-Safe/Raw and the legacy Oracle row remain internal.
- `outputs/DATE3/exp3` through `exp6`: paper-facing aggregate tables. Control
  coverage and accuracy are recomputed from `CHUNK_REPORT.csv` using the same
  unique Tile-lifetime byte definition as PIVOT.
