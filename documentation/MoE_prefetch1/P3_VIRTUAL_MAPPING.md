# P3 virtual Bank mapping and object lifetime

P3 introduces virtual IA Tile, Weight Chunk, OA Tile, and Accumulator objects.
Each object records its tensor class, byte size, requested Bank-group width,
expert, FF1/FF2 stage, Tile, and Chunk identity.

The mapping table enforces:

- allocation before access;
- a stable physical Bank group throughout the live interval;
- exact per-Bank capacity accounting;
- atomic allocation failure;
- explicit release and exact capacity recovery;
- no access at or after the release cycle;
- no object-ID reuse or implicit migration;
- deterministic placement and statistics.

Supported placement policies are round-robin, least occupied, least queue
pressure, and conflict aware. Pressure-aware policies consume only supplied
current/history counters; they do not inspect future requests.

`make_request()` resolves the mapping and emits a P2 `UnifiedMemoryRequest`
whose preferred Bank group is the stable physical mapping. P3 therefore joins
capacity residency and access service without restoring tensor-specific Bank
ownership.
