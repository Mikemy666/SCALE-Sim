# P4: Trace-derived compute tiles and weight chunks

P4 replaces provisional analytical tile metadata on the detailed GPU with
metadata extracted from the demand matrices produced by `single_layer_sim`.
Black-box GPUs intentionally retain the analytical chunk model.

## Tile boundary rule

A consecutive run of non-empty rows in the filter demand matrix is a weight
access burst. Each burst starts one compute tile. The next burst starts the
next tile; for the final tile, the end is the longest demand-matrix length.
This preserves the timing and request counts observed by SCALE-Sim instead of
reconstructing tiles only from tensor dimensions.

Each detailed chunk records the raw filter addresses, trace-cycle boundaries,
IA/W/OA request counts, precision-aware byte count, and load-cycle estimate.
The runtime coordinator uses the trace tile durations as weights when dividing
the measured expert latency across its tile events.

## Weight address isolation

SCALE-Sim normally restarts every layer's filter addresses at `FilterOffset`.
P4 retains those raw addresses for trace auditing and additionally assigns a
topology-wide logical range using the prefix sum of preceding layers' weight
sizes. The extractor rejects a trace address outside its layer's declared
weight range. Consequently, different layers and experts cannot accidentally
alias in the logical address space used by later prefetch work.

## Report

`EP_MOE_CHUNKS.csv` contains both models:

- `detailed_demand_trace` for experts on `DetailedGPUId`;
- `analytical` for experts on black-box GPUs.

P4 does not issue weight-memory or prefetch requests. Those requests and their
contention timing belong to P5.
