# DATE2 simulator experiment package

DATE2 contains all simulator-side experiments required by paper section IV:

- `exp1`: B1/C1 layer, expert-stage, IA/Weight/OA flow characterization;
- `exp2`: C2 exhaustive static IA:Weight:OA Bank ownership sweep;
- `exp3`: C3 NoPF versus NaivePF interference characterization;

- `overall`: four controlled Top-1 networks, two homogeneous and two heterogeneous;
- `ablation`: all static/dynamic, no-prefetch/naive/bank-aware components;
- `window_chunk`: W={0,1,2,4,8} x Chunk={1,2,4,8} tiles;
- `robustness`: Top-K={1,2}, experts={4,8,16}, tokens={32,128,256,512},
  balanced/light/high skew, homogeneous/heterogeneous, and 1/2-GPU EP.

Locations are fixed to `configs/MoE/DATE2`, `topologies/MoE/DATE2`, and
`outputs/DATE2`. Run `python3 scripts/prepare_date2_experiments.py` after source
topology edits, then `python3 run_date2_experiments.py --suite all`. Every
matrix is schema-validated and asserts Safe <= best Static-NoPF and Oracle equal
to the minimum real candidate. RTL and synthesis are explicitly out of scope.

Run individual experiments with `python3 run_date2_experiments.py --exp expN`.
For exp4-exp7 the command writes both the canonical matrix and real execution
detail at expert, FFN-stage, Chunk, physical-Bank, and request granularity.
The exp3 command first executes all 20 Window x Chunk matrices and only then
aggregates NoPF/NaivePF rows. Aggregation verifies every matrix workload hash
against its current JSON and rejects stale pre-architecture results.
