# P6 acceptance report

P6 acceptance requires a complete seven-row matrix, identical workload and
resource provenance, exact cycle accounting, mandatory Dynamic-NaivePF,
candidate-derived Safe/Oracle rows, theoretical ordering, deterministic CSV
round trips, and deterministic workload hashing.

`scripts/MoE_prefetch1/validate_matrix.py` validates generated matrix files.
P7 will provide the actual simulator runner that produces raw candidate rows
for every workload; P6 defines and enforces the result contract it must satisfy.
