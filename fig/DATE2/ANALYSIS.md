# DATE2 results analysis

## What to read for each experiment

### IV-B Overall performance and Bank behavior

Read the four files under `outputs/DATE2/overall/`. Use `total_cycles` for
normalized performance, the seven `*_stall_cycles` columns for breakdowns,
and `bank_conflict_rate`, `bank_imbalance`, `hotspot_bank_ratio`,
`idle_bank_ratio`, and `effective_bank_parallelism` for Bank behavior. Do not
use `summary_all.csv` when a field is absent there.

Safe speedups over Static-NoPF are 1.562x (HMoE), 1.476x (Mixtral), 1.190x
(MoDSE), and 1.343x (Switchtrans). In all four cases `selected_candidate` is
`Dynamic-NoPF`. Thus the present data support dynamic virtualization and the
Safe non-regression contract, but not an additional prefetch benefit.

### IV-C Component ablation

Read `outputs/DATE2/ablation/MoDSE_components.csv`. Compare the measured rows
Static-NoPF, Static-NaivePF, Dynamic-NoPF, Dynamic-NaivePF, and MemDomain-Raw.
MemDomain-Safe is a copied selection row, not an independently measured design.
For MoDSE, Raw reaches 1.180x while Safe reaches 1.190x by selecting
Dynamic-NoPF. The gap shows the current Bank-aware prefetch path adds overhead
rather than benefit for this schedule.

### IV-D Window and Chunk sensitivity

Read all 20 `outputs/DATE2/window_chunk/wW_cC.csv` files. Use MemDomain-Raw for
mechanism sensitivity and MemDomain-Safe only for end-to-end non-regression.
Relevant fields are total cycles, prefetch miss/interference stalls, timely and
late ratios, occupancy byte-cycles, mapping count/failures, and overlap cycles.
The lowest Raw cycle count occurs at W=8 and C=8 (1,482,332 cycles). However,
the timely-prefetch ratio is 0 for every point. Therefore this heatmap currently
measures Chunk granularity and transfer scheduling effects, not successful
timely-prefetch behavior.

### IV-E Robustness

Read the 16 files under `outputs/DATE2/robustness/`. Variants cover Top-K,
expert count, token count, routing severity, homogeneous/heterogeneous models,
and 1/2-GPU EP. Safe speedup ranges from 1.084x (4 experts) to 2.305x (32
tokens). Top-2 is 1.276x versus 1.476x for Top-1. The 2-GPU model records 788
communication-stall cycles.

Balanced/light/high routing currently produce identical results. With equal
expert weight sizes and a serialized analytical compute envelope, changing the
assignment distribution does not alter the modeled memory request stream.
These three rows cannot yet substantiate routing-skew robustness and must be
fixed before publication.

## Publication readiness

The DATE2 results are reproducible and satisfy `Safe <= Static-NoPF` and the
Oracle minimum contract. They are suitable for architecture regression and for
showing dynamic mapping gains. They are not yet sufficient for the paper's
Bank-aware-prefetch or routing-skew claims because timely prefetches are zero,
Raw is consistently below the Safe-selected no-prefetch candidate, and routing
severity is behaviorally inert. Also retain the existing
`paper_scale_performance_claim=false` disclosure: weight traffic is uniformly
scaled by eight for mechanism-level evaluation.
