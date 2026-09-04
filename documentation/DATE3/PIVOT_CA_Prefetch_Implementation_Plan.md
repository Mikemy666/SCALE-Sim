# PIVOT-CA Prefetch Simulator Implementation Plan

## Scope

This plan covers only the SCALE-Sim performance simulator. DATE2 inputs and
published outputs remain backward compatible and are not rewritten. DATE3
adds an online policy on top of the existing unified Bank, virtual mapping,
and streaming-residency implementation.

DATE3 also adds a parameterized Expert-Parallel system envelope: one global
expert set and Router, unique owner mapping, Token-level Top-k replicas,
Detailed-NPU cycle simulation, analytical Peer owner execution, dispatch,
return/combine, and dependency-based system completion. DATE2 artifacts remain
outside this contract.

| Mechanism | DATE2 file location | Current implementation | DATE3 modification |
|---|---|---|---|
| Chunk selection | `memdomain_runner.py::_effective_prefetch_window`, workload JSON `chunks` | Physical tiles are fixed in JSON; Chunk is selected by an offline Window/Chunk sweep | `CoverageAccuracyConstrainedPrefetchPolicy` chooses a configured tile count at every adaptation epoch during one run |
| Window selection | `NaivePrefetchPolicy.plan`, `_effective_prefetch_window` | Fixed JSON look-ahead, or one capacity-derived value for the whole run | Candidate Window search selects the smallest sufficient look-ahead and applies cooldown, hysteresis, and bounded steps |
| Bank selection | `BankAwarePrefetchPolicy`, `VirtualBankMappingTable.allocate` | Pressure-aware placement for a fixed prefetch plan | Normalize queue/busy/conflict/residency pressure, retain bounded Top-K groups, and jointly score `(Chunk, Window, Group)` |
| Coverage | `memdomain_runner.py` | `prefetch request count / all chunk count` | Unique useful timely bytes / unique required non-resident bytes; invalid when denominator is zero |
| Accuracy | `memdomain_runner.py` | `prefetch request count / prefetch request count` | Unique useful timely bytes / unique actually prefetched bytes; invalid when denominator is zero |
| Timely/Late/Unused | `streaming_residency.py` | Timely and late are completion relative to use; every normal plan is consumed | Track byte-accurate unique Tile lifetimes; add unused and evicted-before-use outcomes without changing DATE2 defaults |
| Occupancy | `streaming_residency.py` | bytes multiplied by issue-to-release cycles | Preserve byte-cycles; expose epoch and aggregate values in DATE3 reports |
| Runtime feedback | none for quality | One whole-run adaptive Window heuristic | Valid-only EMA for coverage, accuracy, late ratio, timing error, pressure, and occupancy; prior completed epochs only; Exp5 uses relative shadow feedback without an unreachable absolute floor |
| Guard/Fallback | DATE2 Safe selection and Bank-aware decision guard | Safe may retain measured incumbents; DATE2 matrix also derives an end-of-run safe row | DATE3 candidate rejection is online; multi-layer quality-only rejection may propose dynamic best-effort before measured FixedPF/Coalesced-Demand protection; true timing/capacity failure falls back to reference, conservative action, then NoPF |
| CSV export | `ExperimentRow`, `write_matrix` | DATE2 seven-row schema | Independent DATE3 summary, decision-detail, and quality-epoch CSV schemas plus config/workload hashes |

## Online-information boundary

The policy may read only the current decision cycle, already completed Tile
lifetimes, current Bank snapshot, configured candidates, and the known router
demand/deadline stream. Oracle rows and future total cycles are never policy
inputs. A fixed-reference shadow estimator creates no real Bank requests.

## Implementation sequence

1. Add quality/lifetime types, valid-only EMA, pressure normalization, bounded
   two-level search, action stabilization, and fallback.
2. Extend streaming lifetime results with opt-in unused/evicted semantics while
   preserving defaults used by DATE2.
3. Add a DATE3 runner which performs several decisions in one simulation run,
   executes the chosen requests through the existing Bank/mapping model, and
   writes independent summary/detail/epoch reports.
4. Generate parallel DATE3 suites and commands without running the full set.
5. Validate hashes, ratios, additive cycles, runtime variation, shadow
   isolation, and DATE2 backward compatibility.
6. Localize PIVOT and all controls to the same Detailed NPU, model Peer owners
   without shared Bank state, and validate route/owner/communication/critical
   path contracts.
7. Add a stage-scoped pre-issue incumbent guard and record both evaluated costs
   for every applied action.
