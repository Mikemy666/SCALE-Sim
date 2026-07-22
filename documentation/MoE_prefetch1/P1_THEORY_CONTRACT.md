# P1 theoretical comparison contract

## Claim boundary

An unconstrained online dynamic policy is not theoretically guaranteed to beat
an offline best-static policy. This branch therefore reports three distinct
objects instead of modifying the model until every dynamic run wins:

1. **Runtime Dynamic**: an implementable online policy; regressions are retained.
2. **Oracle Dynamic**: a common-objective candidate search whose feasible set
   explicitly includes every static candidate.
3. **Safe Dynamic**: Runtime Dynamic with Best-Static as a legal fallback.

Only Oracle Dynamic and Safe Dynamic carry the invariant

```text
TotalCycles <= BestStaticTotalCycles
```

Safe Dynamic is a hybrid policy and must be labelled as such in reports.

## Fair-comparison resources

Every candidate in one comparison must have identical:

- physical Bank count;
- total SRAM capacity;
- total bandwidth;
- ports per Bank;
- request-buffer depth.

An IA/Weight/OA allocation must contain only positive groups and conserve the
physical Bank count. Later unified-domain policies may use a different mapping
representation, but they must conserve the same resource budget.

## Common objective

Selection minimizes end-to-end cycles, not Bank stall alone:

```text
TotalCycles = Compute
            + BankStall
            + WeightLoadStall
            + PrefetchMissStall
            + PrefetchInterferenceStall
            + MappingOverhead
            + CommunicationStall
            + OtherStall
```

All components are non-negative and use cycles. Overlap will be represented by
an explicit timeline in later phases rather than subtracting an unvalidated
negative cost.

## Executable contract

The definitions live in `scalesim/memory/memdomain_policy.py`. Tests in
`tests/test_memdomain_p1_policy.py` reject resource mismatch, non-conserving
allocations, negative costs, objective mismatch, and theoretical-order
violations.
