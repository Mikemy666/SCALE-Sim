# P3 MoE runtime coordinator

P3 replaces one-shot expert completion with an event-driven coordinator. Active
experts are split into their current chunk/tile metadata and scheduled onto a
configurable number of compute engines per GPU.

```ini
EnableParallelMoE = True
ComputeEnginesPerGPU = 4
```

When parallel execution is disabled, one global engine is used to provide a
deterministic sequential baseline. When enabled, each GPU has
`ComputeEnginesPerGPU` independent engines. A tile can start only after its
expert's preceding tile completes and an engine becomes available. Time waiting
for an engine is accumulated as `ExpertWaitingTime` in runtime state.

The coordinator records:

- dispatch completion;
- initial-weight-ready transition;
- compute tile start and completion;
- expert completion;
- inactive experts.

These events are written to `EP_MOE_EVENTS.csv`. P5 connects initial and
prefetched weight chunks to the shared weight-memory timeline. P6 adds token
dispatch and output-combine dependencies.

Detailed expert work retains the existing SCALE-Sim cycle result and distributes
that duration across current tile metadata without changing the total work.
Black-box experts use their analytical duration. Detailed tile boundaries are
extracted from demand traces by P4.
