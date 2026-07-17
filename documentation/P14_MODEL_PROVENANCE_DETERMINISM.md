# P14: Model provenance and deterministic output

The run manifest now fingerprints both experiment inputs and the implementation
that interprets them. In addition to config, topology, and layout, it hashes:

- `scalesim/simulator.py`;
- `scalesim/scale_config.py`;
- `scalesim/memory/banked_memory_system.py`.

This prevents two results produced by different runtime or bank-model code from
appearing equivalent merely because their input files match. The report
validator checks any manifest source still available on disk.

The small Top-K=2 integration workload is also run twice in separate output
directories. Routing, events, runtime state, chunks, timeline, expert summary,
group summary, and manifest must be byte-identical. This covers seeded routing,
heap tie-breaking, set serialization, report ordering, and source provenance.
