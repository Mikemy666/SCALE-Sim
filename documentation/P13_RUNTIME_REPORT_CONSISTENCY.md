# P13: Runtime metric consistency across reports

Per-expert prefetch fields in `EP_MOE_REPORT.csv` now come from the event-driven
runtime state, matching `EP_MOE_RUNTIME_STATE.csv` and the aggregates in
`EP_MOE_SUMMARY.csv`. The superseded analytical prefetch estimator was removed;
black-box compute and runtime weight loading are now counted exactly once.

The unified runtime fields include chunk count, initial and demand weight stall,
prefetch hit/miss/rate, miss stall, bandwidth overhead, interference, useful and
useless traffic, and runtime cycles. `WeightLoadingStall` is explicitly emitted
per expert instead of being inferable only from the group summary.

The cross-report validator checks each per-expert field against runtime state and
recomputes group hit, miss, weight stall, miss stall, and bandwidth totals. A
simulation now fails validation if any of these three report layers diverge.
