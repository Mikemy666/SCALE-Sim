# P16: Final experiment and performance audit

The canonical matrix and architectural sanity suite were rerun after enforcing
the P15 black-box boundary. Every run passed cross-report validation.

| Configuration | MoE cycles | Weight stall | Prefetch miss stall | Bank conflict |
|---|---:|---:|---:|---:|
| static, no prefetch | 730 | 74 | 0 | 1216 |
| static, prefetch | 726 | 0 | 39 | 1216 |
| dynamic, no prefetch | 474 | 74 | 0 | 192 |
| dynamic, prefetch | 487 | 0 | 39 | 192 |

After removal of the duplicate black-box estimator, prefetch produced four hits
and four misses (0.5 hit rate). It improved the
static case by four cycles but slowed the dynamic case by thirteen cycles due to
the higher modeled prefetch bank interference. This is retained as a model
result rather than imposing an assumption that prefetch must improve latency.

The experiment runner now emits `EP_MOE_EXPERIMENT_COMPARISONS.csv` with signed
cycle deltas and speedups for prefetch vs. no-prefetch and dynamic vs. static.
Signed deltas preserve regressions as positive cycle costs.

The full four-run matrix completed in about 30 seconds after P15, and the four
sanity cases completed in about 57 seconds on the validation host.
