# MoE_prefetch1 P0 baseline

## Scope

This branch owns the simulator-side MemDomain model and experiments. RTL,
Verilator, Design Compiler, area, power, and timing validation are performed in
another system and are outside this repository's acceptance criteria.

## Inherited state

- Branch: `MoE_prefetch1`
- Base commit: `e5675c5b7155b01d3f095d3f9d153fd74c6a4734`
- Base subject: `add something for exp`
- The branch was created while the worktree already contained DATE1 changes.
- Those pre-existing changes are an inherited development baseline, not new
  MemDomain evidence and not a clean Git checkpoint.
- No inherited file may be deleted, reverted, or committed as new work without
  first reviewing its provenance.

At P0 audit time, the tracked diff contained 76 paths, 537 insertions, and 827
deletions. It included SCALE-Sim source changes, DATE1 configuration changes,
deleted static configurations, and experiment preparation changes. Untracked
assets included DATE1 scripts, notebooks, generated figures, tests, selected
static configurations, and paper PDFs.

## Namespace contract

New simulator work must use these paths:

- configs: `configs/MoE/MoE_prefetch1/`
- topologies: `topologies/MoE/MoE_prefetch1/`
- runners: `scripts/MoE_prefetch1/`
- ignored runtime outputs: `outputs/MoE_prefetch1/`
- tracked documentation: `documentation/MoE_prefetch1/`

DATE1 inputs and outputs are read-only references for the new architecture.
New runs must not write to `outputs/DATE1/`.

## Reproducibility contract

Every experiment run must record:

- Git branch, commit, and dirty-worktree state;
- SHA-256 hashes of simulator sources, configuration, topology, and runner;
- Python and dependency versions;
- all routing seeds;
- the exact command line;
- the output schema version.

Run the P0 manifest generator with:

```sh
.venv/bin/python scripts/MoE_prefetch1/validate_p0.py
```

The generated manifest is written below `outputs/MoE_prefetch1/p0/`, which is
ignored by Git. A non-clean worktree is recorded but does not fail P0 because
the branch intentionally inherited unfinished DATE1 work.

## Baseline tests

P0 uses the currently discoverable source tests as its executable baseline.
Compiled `__pycache__` files are not accepted as test coverage. At branch
creation, only `tests/test_date1_extensions.py` was present as Python source and
contained three tests:

- Chunk-size rechunking conservation;
- routed-token-aware detailed trace scaling;
- physical-bank metric collection.

The missing historical P0-P17 source tests were not present in any reachable
Git revision and therefore could not be restored. They are being replaced with
tracked source-level contract tests. P1 begins with configuration/namespace
smoke tests and executable theoretical-order tests; later phases must add their
own contracts rather than relying on stale bytecode.

## P0 acceptance

- [x] New branch exists and is active.
- [x] Inherited dirty state is documented rather than overwritten.
- [x] New config/topology/output namespaces are reserved.
- [x] A machine-readable manifest generator exists.
- [x] Manifest generator passes on the current worktree.
- [x] Discoverable source tests pass.
- [x] P0 acceptance report records exact results and remaining risks.

See `documentation/MoE_prefetch1/P0_ACCEPTANCE.md` for the executed result.
