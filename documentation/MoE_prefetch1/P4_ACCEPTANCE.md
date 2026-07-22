# P4 acceptance report

P4 acceptance requires tests for timely and late prefetches, exact late stall,
demand misses, unused release, cancellation, independent coverage/accuracy
definitions, interval-union overlap, duplicate scheduling, early-release
rejection, and byte conservation.

P4 provides mechanism and metrics, not a policy that chooses which Chunk to
prefetch. P5 will compare no-prefetch, naive fixed-window, and pressure-aware
Bank-aware scheduling using the same P2-P4 execution semantics.
