# P1 acceptance report

## Result

P1 passed its standalone theory-contract milestone. This does not claim that
the existing DATE1 dynamic allocator already implements Safe Dynamic. It means
the redesigned architecture now has one common resource/objective contract that
later simulator integration must use.

## Implemented

- immutable fair-comparison resource budget;
- non-negative end-to-end cycle breakdown;
- Bank-count conservation for allocation candidates;
- Best-Static selection using total cycles;
- Oracle candidate selection whose feasible set contains all statics;
- Runtime Dynamic plus Best-Static safe fallback;
- explicit theoretical-order assertion;
- source-level replacements for missing cached smoke tests;
- generated-figure isolation in `.gitignore`.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m py_compile scalesim/memory/memdomain_policy.py \
  tests/test_memdomain_p1_policy.py \
  tests/test_moe_prefetch1_regression_contracts.py
```

Acceptance requires:

- all source tests pass;
- the seeded 100-case candidate sweep never violates Oracle/Safe ordering;
- resource mismatches and non-conserving allocations fail;
- selection uses end-to-end cycles rather than Bank stall alone;
- runtime regressions remain observable through metadata even when Safe Dynamic
  falls back.

## Next integration boundary

P2 must connect actual per-layer/per-chunk simulator results to
`PolicyResult`. Until that integration is complete, DATE1 reports must not be
renamed Oracle Dynamic or Safe Dynamic.
