# P11 timing and online-pressure correction

Experiments 1-3 exposed two invalid abstractions. Weight Chunks were assigned a
fixed eight-cycle use spacing unrelated to FFN work, making every transfer late,
and Bank-aware decisions used whole-run cumulative pressure, making every Bank
permanently hot.

P11 derives each FFN stage duration from its MxNxK work on the configured 64x64
array, distributes Chunk deadlines across that stage, and emits stage-level IA
and accumulator requests on rotating physical-Bank groups. The prefetch policy
now observes only compute services overlapping the next 64 cycles. This creates
real late/timely regions as Window changes and gives placement a time-varying
pressure signal.

Regression contracts require: non-constant deadline spacing; more than two
compute requests; both late and timely sweep regions; and MemDomain-Raw to
reduce interference and total cycles relative to Dynamic-NaivePF on the
controlled MoDSE workload. Safe/Oracle selection contracts remain unchanged.
