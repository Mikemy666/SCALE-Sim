# P11: Input provenance and reproducible runs

Every EP-MoE run now writes `EP_MOE_RUN_MANIFEST.csv`. It records the absolute
path, byte size, and SHA-256 digest of the configuration, topology, and layout
inputs used by that run. `validate_ep_moe_reports.py` verifies the digest when
the original input remains available.

The result directory therefore contains both:

- `EP_MOE_CONFIG.csv`, the normalized effective parameter values;
- `EP_MOE_RUN_MANIFEST.csv`, the identity of the source input files.

Together they distinguish an intentional parameter change from editing a file
while retaining the same filename. Temporary generated sanity configurations
are still reproducible by digest even after their temporary path disappears.

The main README contains commands for a default run, the four canonical
experiments, architectural sanity checks, and cross-report validation.
