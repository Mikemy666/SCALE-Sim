# P3 acceptance report

P3 acceptance requires source tests for stable lifetime mappings, exact release,
pre-allocation/post-release rejection, atomic out-of-capacity failure, unique
object identity, deterministic placement, pressure-aware avoidance, mapping to
the P2 request path, and mapping/occupancy statistics.

P3 does not yet schedule future Chunk loads. P4 will add Chunk states,
prefetch/load completion, timely/late/unused classification, and residency
release decisions on top of this mapping table.
