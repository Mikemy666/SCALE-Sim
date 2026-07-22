# P0 acceptance report

## Result

P0 passed on 2026-07-22 (Asia/Shanghai). The branch and inherited worktree are
now identifiable, the new architecture has isolated namespaces, and the
machine-readable baseline can be regenerated without modifying DATE1 outputs.

## Executed checks

```sh
.venv/bin/python scripts/MoE_prefetch1/validate_p0.py
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m py_compile scripts/MoE_prefetch1/validate_p0.py
```

Observed result:

- branch: `MoE_prefetch1`;
- HEAD: `e5675c5b7155b01d3f095d3f9d153fd74c6a4734`;
- inherited tracked changes: 76 paths;
- manifest inputs hashed: 112 before adding the P0 requirements/setup inputs;
- discoverable source test files: 1;
- executed tests: 3;
- passed tests: 3;
- failed tests: 0.

The generated runtime artifact is
`outputs/MoE_prefetch1/p0/baseline_manifest.json`. The `outputs/` tree is
intentionally ignored by Git.

## Risks carried into P1

1. The branch is not a clean commit boundary. It inherited unfinished DATE1
   source, configuration, notebook, and test changes.
2. Historical P0-P17 tests appeared only as Python bytecode cache files and
   were absent from every reachable Git revision. This risk is mitigated by
   new tracked source-level contract tests; coverage must continue growing in
   every phase.
3. Generated DATE1 figures were mixed with untracked source assets. P1 now
   ignores `fig/generated/`; notebooks and plotting sources remain visible.
4. Existing DATE1 dynamic results are characterization inputs only. They are
   not evidence that the redesigned runtime mapping or bank-aware prefetch has
   been implemented.
5. No P0 commit was created because doing so would also stage unrelated
   inherited user changes unless files are selected deliberately.

## P1 entry condition

P1 may modify only explicitly selected source files and the new
`MoE_prefetch1` namespaces. It must first encode the candidate-set, resource
conservation, objective-consistency, and safe-fallback invariants as executable
tests.
