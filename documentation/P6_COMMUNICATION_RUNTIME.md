# P6: Communication dependencies in the EP runtime

P6 moves communication from an analytical report-only term into the EP event
timeline. The model remains intentionally simple: each transfer costs fixed
latency plus bytes divided by configured bandwidth.

## Dependencies

- Token dispatch starts at the MoE-group boundary and must finish before the
  expert's first compute tile.
- With `AllowCommPrefetchOverlap=True`, initial weight loading starts at the
  group boundary and may overlap dispatch.
- With overlap disabled, initial loading starts only after dispatch completes.
- Output combine starts after the expert's final compute tile and extends its
  finish time.

Black-box `local_work_cycles` excludes communication so dispatch/combine are not
counted both analytically and by the runtime coordinator. Black-box GPUs remain
outside the detailed GPU's on-chip bank model.

## Events and metrics

`EP_MOE_EVENTS.csv` adds token-dispatch and output-combine start/completion
events. `EP_MOE_RUNTIME_STATE.csv` records dispatch cycles, combine cycles, and
the measured dispatch/initial-load overlap. The MoE group time in expert reports
is taken from the authoritative EP timeline, so it includes pre-compute dispatch
and post-compute combine.
