# P5: Runtime weight requests and chunk prefetch

P5 connects the P4 chunk plan to the P3 event-driven coordinator. Each GPU has
one ordered weight-memory timeline shared by its active experts. Requests from
different experts therefore contend for weight bandwidth instead of completing
independently.

## Runtime sequence

1. At MoE-group start, every active expert requests `InitialChunk` chunks.
2. A compute tile can start only after its matching chunk is ready and a compute
   engine is available.
3. At tile `i` start, a configured prefetch requests chunk
   `i + ChunkPrefetchWindow`.
4. A chunk without an earlier request is loaded on demand and blocks its tile.
5. A prefetched chunk ready before use is a hit; otherwise only its exposed
   remaining latency is counted as miss stall.

Detailed-GPU prefetch traffic continues to use the existing bank simulation for
aggregate normal/prefetch interference. The event coordinator supplies the
missing per-chunk readiness and bandwidth serialization. Black-box experts use
the same runtime request protocol with analytical chunk sizes and load times,
but do not enter the detailed GPU bank model.

## Events and metrics

`EP_MOE_EVENTS.csv` now includes:

- `initial_weight_request` / `initial_weight_ready`;
- `weight_prefetch_request` / `weight_prefetch_ready`;
- `weight_demand_request` / `weight_demand_ready`.

`EP_MOE_RUNTIME_STATE.csv` adds runtime hit, miss, miss stall, bandwidth
overhead, useful/useless prefetch traffic, and initial weight stall fields.
These are derived from requests actually issued by the coordinator rather than
the older layer-shift estimate.
