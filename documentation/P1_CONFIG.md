# P1 EP-MoE configuration

P1 centralizes the public EP-MoE controls in `[run_presets]`. Legacy configs do
not need to add these fields because `EnableEPMoE` defaults to `False`.

The canonical EP controls are:

```ini
EnableEPMoE = True
EnableParallelMoE = True
NumGPUs = 2
DetailedGPUId = 0
BlackBoxGPUIds = 1
ExpertsPerGPU = 4
ComputeEnginesPerGPU = 4
TopK = 1
MoERoutingMode = balanced
MoETokens = 32
RoutingFile =
RoutingSeed = 40
RoutingSkewFactor = 1.0
EnableChunkPrefetch = False
InitialChunk = 1
ChunkPrefetchWindow = 0
PrecisionBytes = 2
AllowCommPrefetchOverlap = True
```

`BlackBoxGPUIds` may be `auto`, a comma-separated list, or bracketed list such
as `[1]`. In the current one-detailed-GPU architecture it must contain every GPU
except `DetailedGPUId` exactly once.

Legacy next-layer prefetch remains controlled by `EnablePrefetch` and
`PrefetchWindow`. EP weight-chunk prefetch is independently controlled by
`EnableChunkPrefetch` and `ChunkPrefetchWindow`. Enabling both models in one run
is rejected because their metrics and request semantics differ.

`EnableCommunicationOverlap` remains accepted as an alias for
`AllowCommPrefetchOverlap`. New configurations and generated config snapshots
use only the canonical name. `PrecisionBytes` is the unified element size for
both token dispatch and output combination.

Available configurations:

- `configs/MoE/ep_default.cfg`
- `configs/MoE/ep_experiments/static_no_prefetch.cfg`
- `configs/MoE/ep_experiments/static_prefetch.cfg`
- `configs/MoE/ep_experiments/dynamic_no_prefetch.cfg`
- `configs/MoE/ep_experiments/dynamic_prefetch.cfg`

The small `topologies/MoE/test.csv` workload contains eight complete experts and
matches the default two GPUs × four experts mapping.
