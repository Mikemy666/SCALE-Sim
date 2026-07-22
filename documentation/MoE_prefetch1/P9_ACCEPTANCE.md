# P9 acceptance report

P9 acceptance requires batch/session equivalence, chronological and priority
ordering, persistent contention, fixed-capacity reuse, deadline-crossing demand
fallback, shared compute/load service, zero capacity leaks, and peak occupancy
bounded by physical capacity.

Acceptance completed with 93 repository tests passing. The four catalog
matrices also pass the canonical seven-row validator under fixed capacity:

| Workload | MemDomain-Safe | Oracle |
|---|---:|---:|
| Switch Base-8 | 774 | 758 |
| Mixtral 8x7B | 61,693 | 61,005 |
| MoDSE 300M x 8 | 7,510 | 7,302 |
| DeepSeekMoE 16B | 21,213 | 18,109 |

For every matrix, MemDomain-Safe is no slower than the exhaustively selected
Static-NoPF candidate, and Oracle equals the minimum measured candidate. Raw
ablation rows remain unmodified even when mapping overhead makes one lose.
