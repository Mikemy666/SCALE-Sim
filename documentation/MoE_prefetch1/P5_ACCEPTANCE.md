# P5 acceptance report

P5 acceptance requires tests for disabled prefetch, fixed-window Naive behavior,
zero-window equivalence, hotspot redirection, high-pressure delay, deadline
cancellation, group-capacity checks, enforcement of selected physical Banks,
Safe fallback, and Oracle candidate containment.

P6 will turn these mechanisms into the canonical experiment baselines and
common CSV schema. It must report raw Runtime/Bank-aware results alongside Safe
and Oracle rather than replacing observed regressions.
