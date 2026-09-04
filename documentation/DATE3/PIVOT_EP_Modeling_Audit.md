# PIVOT Parameterized Expert-Parallel Modeling Audit

> Scope: DATE3 on `MoE_prefetch2`; DATE2 results and RTL are excluded.

## 1. Repository Snapshot

- Repository: `/home/MikeNotFound/code/SCALE-Sim` — 已验证
- Branch: `MoE_prefetch2` — 已验证
- Audit-start commit: `0b7dace7aba26a69fdd93d7d8073bdb304b4e3a7` — 已验证
- Working tree: dirty; pre-existing changes were preserved — 已验证
- Entry: `run_date3_experiments.py`; runner:
  `pivot_ca_runner.run_pivot_ca` — 已验证
- Config/topology roots: `configs/MoE/DATE3/` and
  `topologies/MoE/DATE3/` — 已验证

## 2. Expected Parameterized EP Contract

`date3_ep_model.EPContract` requires global E, P, Detailed NPU, Top-k, one
owner per expert, controlled per-expert `Ne`, and FF1/FF2 metadata. P defaults
to 2 but is read from configuration. Non-divisible E/P is balanced without
omission or duplication. Tests: `test_overall_uses_two_npus...` and audit tests
A–D. Judgment: 已验证.

## 3. Expert-Count Configuration Semantics

`ep.num_experts` is the global expert count. It is derived from generated
workload experts; it is not experts per NPU. Router topology N and FF1/FF2 stage
counts change in the 4/8/16 expert sweep. No runtime constant assumes E=8.
Judgment: 已验证.

## 4. Expert-to-NPU Ownership

`ep.expert_owner_map[e]` is complete, unique, and range checked by
`EPContract.validate`. The default E=8/P=2 map is E0–E3→NPU0 and E4–E7→NPU1;
E=16/P=2 is 8+8; the E=10/P=3 extension is 4+3+3. Heterogeneous parameter and
routed-MAC imbalance is reported by `validate_ep_modeling.py`, not hidden by
the expert-count balance. Judgment: 已验证.

## 5. Global Router Semantics

`EPContract.routes()` runs over global expert IDs and materializes TokenID,
source, global expert, owner, Top-k slot, routing weight, destination offset,
and remote status. Top-2 creates two distinct global experts per Token. The
routes deterministically realize controlled topology counts rather than neural
logits. Judgment: 已验证 for controlled routing; learned Router prediction 未找到.

## 6. Token Dispatch and Expert Token Counts

`deterministic_routes_from_counts` closes exactly against `ep.token_counts`.
`localize_detailed_npu` and `build_ep_system_timeline` aggregate those same
replicas once at the owner. Inactive experts are omitted. Dispatch bytes depend
on actual remote replicas, not a fixed remote fraction. Judgment: 已验证.

## 7. Local Expert FF1/FF2 Execution

Detailed execution retains only active owned experts and their FF1/FF2 chunks
and IA/OA/ACC requests. Peer rows retain owner, `Ne`, stage compute cycles, and
finish time. HMoE/MoDSE heterogeneous stage cycles remain different.
`test_detailed_workload_contains_only_owned_active_experts` closes the source
side. Numeric FF1 tensor values are not simulated. Judgment: 已验证 for
performance work; tensor arithmetic 部分验证.

## 8. Expert Weight Storage Hierarchy

Detailed weights use `StreamingResidencyEngine`: owner-local off-chip transfer
→ finite-capacity Tile/Chunk residency → release. Peer owners compute nonzero
local weight bytes and HBM startup/bandwidth cycles; they do not move remote
weights to the Detailed NPU. Peer HBM has no packet/request queue model.
Judgment: Detailed 已验证; Peer 部分验证.

## 9. PIVOT Tile/Chunk and Bank Scope

Only the localized Detailed workload constructs `UnifiedBankDomain` and
`VirtualBankMappingTable`. Peer workloads never receive these objects and
cannot share a Bank pool. Chunk identity contains expert, FF stage and tile;
the Detailed instance itself supplies the NPU namespace. Prefetch epochs are
strictly grouped by `(expert_id, ffn_part)`. Tests:
`test_prefetch_epoch_never_crosses_expert_or_ffn_stage`. Judgment: 已验证.

## 10. Peer-NPU Black-Box Model

`date3_ep_system.build_ep_system_timeline` executes only Peer-owned active
experts, retains `Ne`, separates FF1/FF2, loads owner-local weights, returns
results, and affects the system critical path. It returns analytical rows and
scalar timing, not a second cycle-level Bank trace or background Bank pressure.
Judgment: 部分验证, as intentionally permitted by the black-box contract.

## 11. Communication and Combine

Dispatch and return are separate transactions:

- dispatch = remote replicas × `token_payload_bytes`;
- return = remote replicas × `result_payload_bytes`.

`ep_return_combine.csv` preserves Token order, experts, owners, weights, result
count, and completion. Top-2 waits for two weighted results. Numeric weighted
tensor addition is abstracted to `combine_cycles_per_token`. The interconnect
models startup/bandwidth but not packets, topology, link contention, collectives,
or complete All-to-All. Judgment: 部分验证.

## 12. Multi-NPU Cycle Accounting

`result_ready=max(detailed_ready,peer_ready)` and
`total=result_ready+combine`. NPU times are never summed. Only Peer work exposed
beyond the Detailed path enters `communication_stall_cycles`; combine enters
`other_stall_cycles`. The runner asserts equality with the additive component
contract. Test: `test_dispatch_return_and_critical_path_are_dependency_exact`.
Judgment: 已验证.

## 13. HMoE Audit

| Item | Result |
|---|---|
| E/P/Top-k | 8/2/1 — 已验证 |
| Class | heterogeneous — 已验证 |
| Owners | contiguous 4+4 — 已验证 |
| Param load by owner | 334,848 / 556,032 B — 已验证 |
| Routed MAC load | 12,739,584 / 13,819,392 — 已验证 |
| Local/remote execution | disjoint owner paths — 已验证 |
| Peer/communication | analytical — 部分验证 |

## 14. Mixtral Audit

| Item | Result |
|---|---|
| E/P/Top-k | 8/2/1 — 已验证 |
| Class | homogeneous dimensions — 已验证 |
| Param load by owner | 387,072 / 387,072 B — 已验证 |
| Routed MAC load | 14,902,272 / 9,870,336 — 已验证 |
| Local/remote execution | disjoint owner paths — 已验证 |
| Peer/communication | analytical — 部分验证 |

## 15. MoDSE Audit

| Item | Result |
|---|---|
| E/P/Top-k | 8/2/1 — 已验证 |
| Class | heterogeneous dimensions — 已验证 |
| Param load by owner | 184,320 / 184,320 B — 已验证 |
| Routed MAC load | 7,225,344 / 4,755,456 — 已验证 |
| Local/remote execution | disjoint owner paths — 已验证 |
| Peer/communication | analytical — 部分验证 |

## 16. Switch Transformer Audit

| Item | Result |
|---|---|
| E/P/Top-k | 8/2/1 — 已验证 |
| Class | homogeneous dimensions — 已验证 |
| Param load by owner | 294,912 / 294,912 B — 已验证 |
| Routed MAC load | 11,354,112 / 7,520,256 — 已验证 |
| Local/remote execution | disjoint owner paths — 已验证 |
| Peer/communication | analytical — 部分验证 |

## 17. Parameterized Unit Tests

- E=4,P=1: ownership oracle and zero remote communication — 已验证.
- E=8,P=2: complete 4+4 production owner map — 已验证.
- E=16,P=2: complete 8+8 map; no four-per-NPU assumption — 已验证.
- E=10,P=3: balanced extension 4+3+3 — 已验证.
- Top-2: two distinct replicas and two result dependencies — 已验证.
- Heterogeneous experts: parameter/MAC differences retained — 已验证.

Focused final regression: 43 tests passed. Full paper suites were not run.

## 18. Deterministic Top-1 Trace

`outputs/DATE3/validation/ep_model_audit_trace.csv` is the small deterministic
contract trace. Production `ep_routes.csv`, `ep_peer_workloads.csv`, and
`ep_timeline.csv` provide equivalent runtime evidence for generated configs.
Judgment: 已验证.

## 19. Deterministic Top-2 Trace

`ep_model_audit_top2_trace.csv` supplies the reference trace, while
`test_top2_has_two_distinct_replicas_per_token` and
`test_top2_combine_waits_for_two_weighted_results` execute the production path.
Weights and result dependencies close; tensor values are abstract. Judgment:
routes 已验证, numeric combine 部分验证.

## 20. Violations and Risks

The implementation tests exclude fixed E=8, fixed four experts/NPU, P×E expert
duplication, local-only Router, Top-k without replicas, source execution of
remote experts, full expert on-chip residency, cross-NPU Bank/mapping sharing,
fixed remote bytes, summed NPU time, and missing return dependency.

Remaining risks are analytical Peer timing, scalar interconnect constraints,
and absence of numeric tensor values. Online PIVOT does not read Oracle or a
completed full-run result. Its incumbent guard compares only known current-stage
actions before issuing requests and logs both model-predicted prefix costs.

## 21. Final Verdict

**B. 核心参数化 EP 建模成立，但通信和 Peer NPU 为抽象模型。**

Expert ownership, global controlled Top-k routes, actual `Ne`, owner-local
execution and weights, NPU-local PIVOT state, return/combine dependencies, and
critical-path accounting are closed by code, config, trace and tests. Packet
NoC and a second detailed Peer Bank simulation are intentionally absent.

## 22. Required Fixes

No critical fix remains for the intended Detailed-NPU + Peer-black-box paper
model. To claim verdict A, future work must add independent cycle-level Peer
Bank/mapping simulation and a packet/contention-aware interconnect. Numeric
tensor combine is unnecessary for cycle modeling but would be required for a
functional-value simulator.
