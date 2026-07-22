# P9 fixed-capacity streaming execution

P9 adds a persistent P2 Bank session and a chronological Chunk streaming
engine. Bank ports, outstanding queues, conflict counters, and busy history
survive across request submissions. The original batch API remains available
and is regression-tested against the session report.

The streaming engine allocates each Chunk at its planned issue cycle, schedules
its unified-Bank transfer, and releases its P3 mapping at the actual consumption
cycle. If capacity is unavailable, the load waits for the earliest legal
release. A speculative load delayed beyond its use deadline becomes a demand
load and records allocation wait and miss stall. Peak occupancy may never
exceed the fixed physical capacity, and all mappings must be released at exit.

The P7 runner now executes all five measured baselines through this event
engine. Catalog workloads use a fixed 24 x 64 KiB physical capacity and declare
`streaming_fixed_capacity=true`; capacity is not derived from total weights.
