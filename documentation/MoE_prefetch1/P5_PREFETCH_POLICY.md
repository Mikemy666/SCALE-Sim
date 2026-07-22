# P5 prefetch policies

P5 separates policy decisions from the P2-P4 execution mechanism:

- `none`: demand load at the declared use cycle;
- `naive`: fixed Chunk window and fixed Bank group, ignoring pressure;
- `bank_aware`: online pressure/capacity/deadline decision;
- `oracle`: offline upper bound whose candidates include no-prefetch;
- `safe`: Bank-aware outcome with transparent no-prefetch fallback.

Bank-aware decisions consume one current `BankSnapshot`. They may prefetch,
redirect to a cooler Bank group, delay for re-evaluation, or cancel speculation
and preserve correctness with a demand load. They do not inspect a future
pressure trace.

Only Oracle and Safe carry the invariant that cycles cannot exceed the
no-prefetch baseline. Raw Naive and raw Bank-aware regressions remain visible.
This prevents the simulator from forcing every online policy to win while still
providing the theoretical relationship required for controlled comparisons.
