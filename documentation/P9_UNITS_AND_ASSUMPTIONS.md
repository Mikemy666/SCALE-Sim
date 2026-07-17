# P9: Unit consistency and model assumptions

All traffic fields use bytes. Tensor sizes from topology are element counts and
are converted exactly once using `PrecisionBytes`:

- detailed trace addresses remain element addresses;
- detailed chunk `WeightBytes` is unique weight elements times precision;
- analytical expert weight, chunk, bandwidth, prefetch, and background-pressure
  traffic use the same conversion;
- dispatch/combine bytes use routed elements times precision;
- detailed DRAM access counts are converted to bytes in the EP summary.

Compute MAC counts and systolic compute cycles do not depend on precision in the
current model. `PrecisionBytes` changes storage and bandwidth costs only. This
is an explicit abstraction: the array dimensions and arithmetic throughput are
not automatically changed when precision changes.

Other important boundaries remain:

- black-box GPUs generate analytical workload and external pressure only;
- only the detailed GPU contributes on-chip bank conflict/utilization;
- logical weight addresses isolate layers, while raw trace addresses retain
  SCALE-Sim's per-layer offset convention;
- communication is latency-plus-bandwidth, not a packet/network simulator;
- dynamic bank remapping continues to use the legacy `old_model` overhead.
