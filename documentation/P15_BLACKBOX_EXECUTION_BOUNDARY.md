# P15: Enforced black-box execution boundary

Remote black-box expert layers are no longer executed through
`single_layer_sim.run()`. Their runtime, chunk, memory, and communication costs
come only from the analytical black-box model. Normal layers and expert layers
owned by `DetailedGPUId` continue through the detailed SCALE-Sim compute and
bank models. Non-EP simulations are unchanged.

Legacy layer-oriented CSV files retain one row per topology layer. For skipped
black-box layers, compute rows contain analytical compute cycles while bandwidth
and detailed-access rows contain zero physical accesses. Black-box layers are
not emitted into the detailed bank-model report.

`EP_MOE_LAYER_EXECUTION.csv` makes this boundary explicit with
`detailed_scalesim` or `analytical_blackbox` and a boolean indicating whether a
detailed simulation actually ran. This prevents analytical remote layers from
being mistaken for contributors to local bank conflicts.
