# PIVOT-CA Prefetch Simulator Implementation

## 1. Repository Snapshot

The implementation is on branch `MoE_prefetch2` and is an incremental
extension of the DATE2 SCALE-Sim simulator. No Chisel, Verilog, Buckyball RTL,
DC, FPGA, area, power, or timing source is inspected or modified.

## 2. DATE2 Baseline Architecture

DATE2 loads fixed physical Tiles from JSON through
`scalesim/memory/memdomain_runner.py::load_runner_config`. Fixed prefetch uses
`NaivePrefetchPolicy.plan`; Bank-aware placement uses
`BankAwarePrefetchPolicy.decide` and `VirtualBankMappingTable.allocate`.
`StreamingResidencyEngine.run` is the common real request path.

The audit found that DATE2's published `prefetch_coverage` was planned request
count divided by Chunk count, `prefetch_accuracy` was planned requests divided
by itself, and runner-level unused ratio was fixed at zero. DATE2 Safe also has
a matrix-level completed-run incumbent selection. DATE3 does not use that
selection or Oracle information for online actions.

## 3. DATE3 Directory Layout

- Configs: `configs/MoE/DATE3/`
- Topologies: `topologies/MoE/DATE3/`
- Outputs: `outputs/DATE3/`
- Scripts: `scripts/DATE3/`
- Figures: `fig/DATE3/`
- Documentation: `documentation/DATE3/`

`scripts/DATE3/prepare_date3_experiments.py::main` generates Exp1--Exp3
metadata, 4 overall, 4 end-to-end, 8 ablation, 32 fixed Window/Chunk, 32 joint-prefetch,
19 quality sensitivity, 96 robustness, and 1 minimal integration
configuration. DATE2 files are read as sources and never rewritten.

## 4. Modified Simulator Architecture

`scalesim/memory/pivot_ca_prefetch.py` contains the new policy and
`scalesim/memory/pivot_ca_runner.py::run_pivot_ca` integrates it with the
existing unified Bank, virtual mapping, and streaming engine. The public policy
name is `PIVOT-CA`; the JSON selector is
`policy.prefetch_policy=coverage_accuracy_constrained`. Missing selectors in
old DATE2 JSON retain their existing path.

DATE3 additionally wraps PIVOT-CA and every control in the same parameterized
EP contract. The Detailed NPU is simulated at Bank-cycle level; Peer owners use
an analytical owner-local workload; system completion follows dispatch,
parallel local/Peer work, return, and combine dependencies.

## 5. Code Change Map

| File | Main symbol | Status |
|---|---|---|
| `pivot_ca_prefetch.py` | `CoverageAccuracyConstrainedPrefetchPolicy` | implemented |
| `pivot_ca_runner.py` | `run_pivot_ca`, `run_pivot_ca_file` | implemented |
| `streaming_residency.py` | `StreamingLoadPlan`, `StreamingResidencyEngine.run` | backward-compatible lifetime extension |
| `run_date3_experiments.py` | `main` | resumable DATE3 entry point |
| `validate_date3_contracts.py` | `validate_variant` | implemented |
| `test_pivot_ca_prefetch.py` | 20 deterministic tests | passing |

## 6. Tile Lifetime Tracking

`TileLifetime` records required, issue, completion, first use, release, and
eviction cycles. `quality_from_lifetimes` deduplicates by stable Tile ID and
rejects conflicting duplicates. DATE3 opts into unused/eviction controls in
`StreamingLoadPlan`; DATE2 defaults (`will_use=True`, no eviction) are unchanged.

## 7. Coverage and Accuracy Definitions

`PrefetchQualityStats` implements useful-timely/required Coverage and
useful-timely/prefetched Accuracy. Late, unused, and evicted bytes are excluded.
Zero denominators produce invalid flags and empty ratios. Tests:
`test_coverage_is_useful_over_required_bytes`,
`test_accuracy_is_useful_over_prefetched_bytes`,
`test_invalid_denominators_are_not_fabricated`.

## 8. EMA Feedback

`CoverageAccuracyConstrainedPrefetchPolicy.update_feedback` updates quality
EMAs only for valid ratios and also tracks late ratio, timing error, pressure,
and occupancy. Warmup uses `reference_chunk/reference_window` for at least
`ema_warmup_epochs` valid epochs. Test: `test_ema_formula`.

## 9. Reference Prefetch Quality

The default `reference_mode=shadow_fixed`. `run_pivot_ca` evaluates the fixed
reference in a separate Bank/mapping instance over the same known demand
prefix. It contributes zero requests to the real report, asserted by
`shadow_real_request_count=0` and
`test_shadow_reference_has_no_real_requests`.

## 10. Runtime Chunk Selection

At each epoch, `choose` searches configured `candidate_chunks`. Chunk denotes
the number of minimum physical Tiles covered by that online decision; physical
Tile bytes are read from each `WeightChunk`, not assumed to be 2 KB. The
multi-layer path serializes each coalesced request on shared HBM and compares
its completion with every Tile deadline. This preserves the first-byte benefit
of a smaller Chunk under tight slack instead of assigning one epoch-wide
completion timestamp that makes the largest Chunk dominate by construction.
All legal Chunks are evaluated before feedback warm-up; warm-up controls only
feedback-driven pruning and hysteresis.

## 11. Runtime Window Selection

Legacy paths use `minimum_window`; the multi-layer path jointly evaluates every
configured Window with the per-request HBM schedule. The chosen Window changes
actual issue cycles in `run_pivot_ca`. Test:
`test_minimum_window_selects_first_sufficient`.

## 12. Bank Pressure and Group Selection

`_bank_pressure` normalizes queue, busy, conflict, and residency components.
`_groups` constructs a bounded deterministic group set and retains the lowest
pressure `bank_candidate_count`. The minimal run uses three distinct groups.
Test: `test_lower_pressure_group_is_selected`.

## 13. Candidate Score and Constraints

`choose` calculates predicted Coverage, Accuracy, latency benefit, occupancy
byte-cycles, conflict, pressure, and mapping cost. General DATE3 cases may use
hard quality floors. Exp5 instead uses relative shadow feedback with no
unattainable absolute floor, while retaining late/timing EMAs, admission, and
the measured online incumbent guard. Latency is the positive objective. Tests:
`test_quality_constraint_rejects_high_score_candidate` and
`test_capacity_infeasible_candidate_is_removed`.

## 14. Guard and Fallback

Adaptation uses epoch gating, cooldown, hysteresis, bounded Chunk/Window steps,
and emergency overrides. In the multi-layer path, a candidate rejected only by
quality may become a dynamic best-effort proposal, but must still beat FixedPF
and Coalesced-Demand in the measured online prefix guard. True
no-timely-window or capacity failures retain reference, conservative, then
NoPF fallback. Fallback is never rewritten by step limiting. Tests:
`test_fallback_prefers_reference_when_quality_search_fails` and
`test_hysteresis_holds_small_improvement`.

The runner also applies a pre-issue online incumbent guard per expert/FFN
stage. It evaluates the adaptive action, fixed reference, and dynamic NoPF on
the same current prefix using separate simulator state, commits the minimum
prefix cost, and logs all three costs in `online_incumbent_guard.csv`. Ties
retain prefetch quality. It does not select the minimum after the full
experiment has completed.

## 15. Configuration Schema

All policy values live under `coverage_accuracy_policy` in generated JSON.
Defaults are centralized in `CoverageAccuracyPolicyConfig`; candidate sets,
thresholds, EMA rates, pressure/score weights, capacity, stabilization, and
safety margin are configurable and validated.

## 16. CSV and Detail Reports

`run_pivot_ca_file` writes `summary.csv`, `decision_detail.csv`,
`quality_epochs.csv`, EP route/work/timeline/return reports, the online guard
report, and `metadata.json`. Fields and units are fixed by
`documentation/DATE3/DATE3_DATA_CONTRACT.md`. Internal comparison sources keep
diagnostic rows; paper aggregation exposes one proposed scheme named `PIVOT`.
PIVOT is the MemDomain architecture, not a competitor to it. Legacy
MemDomain-Safe/Raw and Oracle rows remain internal and are removed from public
figures.

## 17. Unit Tests

`tests/test_pivot_ca_prefetch.py` and `tests/DATE3/` cover byte
ratios, late/unused/eviction, duplicates, invalid denominators, EMA, minimum
Window, constraints, pressure, capacity, fallback, hysteresis, runtime
adaptation, additive cycles, shadow isolation, determinism/hashes, and DATE2
compatibility, plus ownership, routing, Peer work, return/combine, system
critical path, stage-scoped decisions, and online incumbent protection.

## 18. Minimal Integration Results

Configuration: `configs/MoE/DATE3/unit_cases/MoDSE_minimal.json`.

- decisions: 10; adaptations: 6;
- Chunk range: 4--8 Tiles;
- Window range: 1--8;
- selected Bank groups: 6;
- Coverage: 0.865; Accuracy: 0.928;
- Late/Unused/Evicted-before-use: 2048 B each;
- fallback count: 2;
- PIVOT-CA system total: 72,414 cycles; fixed reference: 72,927 cycles;
- measured local memory-stall reduction in this synthetic case: 513 cycles;
- Detailed/Peer readiness: 72,158 / 65,244 cycles; combine: 256 cycles;
- dispatch and return: 39,168 bytes each;
- additive cycle contract: passed.

These are low-cost integration results, not a claim about full-suite speedup.

## 19. DATE2 Backward Compatibility

`test_date2_baseline_remains_runnable` produces the original seven-row DATE2
matrix. `validate_date3_contracts.py` checks that `outputs/DATE2` and
`fig/DATE2` are not dirty. Current status: passed; DATE2 result files unchanged.

DATE3 paper controls are not run on the old global single-domain workload.
`run_date3_ep_baseline_matrix` first localizes them to the same Detailed NPU and
then applies the same Peer, dispatch, return, combine, and critical-path model
as PIVOT-CA. Thus policy comparisons change only the Bank/prefetch mechanism.

## 20. Known Limitations

- Router-known current-stage Tiles use probability 1; speculative integration
  Tiles use configured probabilities. There is no neural predictor.
- Online prefix feedback is deterministically replayed through the same
  Bank/mapping engine to expose only already completed epochs. This prioritizes
  correctness and is slower than a future persistent incremental session.
- Bank group candidates are bounded cyclic groups, not all combinations.
- Peer NPUs use owner-local analytical timing; the Detailed NPU alone uses the
  cycle-level Bank simulation.
- Dispatch and return use transaction startup/bandwidth without packet-level
  NoC contention.
- The complete overall, ablation, sensitivity, fixed sweep, and robustness
  suites are generated but intentionally not run.
- No performance benefit beyond the synthetic minimal case is claimed.

## 21. Full Experiment Commands

```bash
.venv/bin/python scripts/DATE3/prepare_date3_experiments.py
.venv/bin/python run_date3_experiments.py --exp exp1
.venv/bin/python run_date3_experiments.py --exp exp2
.venv/bin/python run_date3_experiments.py --exp exp3
.venv/bin/python run_date3_experiments.py --exp exp4
.venv/bin/python run_date3_experiments.py --exp exp5
.venv/bin/python run_date3_experiments.py --exp exp6
.venv/bin/python run_date3_experiments.py --exp exp7
.venv/bin/python scripts/DATE3/plot_experiments.py --exp all
.venv/bin/python scripts/DATE3/validate_paper_experiments.py --require-outputs --require-figures
.venv/bin/python scripts/DATE3/validate_date3_contracts.py --suite overall --require-runtime-variation
```

Use `--variant <name>` for one configuration, `--dry-run` to inspect work, and
omit `--force` to resume hash-valid PIVOT-CA outputs.
