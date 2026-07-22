# P6 canonical experiment matrix

Every workload/configuration point produces exactly seven rows:

1. Static-NoPF;
2. Static-NaivePF;
3. Dynamic-NoPF;
4. Dynamic-NaivePF;
5. MemDomain-Raw;
6. MemDomain-Safe;
7. Oracle.

Rows share one workload hash and hardware resource budget. Total cycles must
equal the common P1 component sum. Safe and Oracle rows are derived by copying
a real candidate row and recording `selected_candidate`; fabricated lower
cycles fail validation. Raw online regressions remain in their own rows.

The schema includes cycle breakdown, Bank conflicts/utilization proxies,
queue depth, prefetch coverage/accuracy/timeliness/occupancy/overlap, and mapping
counts/failures/peak occupancy. CSV row ordering and line endings are
deterministic.
