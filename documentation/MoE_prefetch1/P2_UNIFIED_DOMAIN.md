# P2 unified physical Bank domain

## Implemented boundary

`scalesim/memory/unified_bank_domain.py` provides one deterministic physical
Bank pool for IA, Weight, OA, Accumulator, and Weight-prefetch traffic. Tensor
type remains metadata for breakdown reports; it no longer grants ownership of
a separate physical Bank model.

The P2 service path models:

- one shared physical Bank namespace;
- address interleaving across Banks;
- total bandwidth divided across physical Banks;
- configurable ports per Bank;
- finite outstanding-request buffers;
- deterministic arbitration with compute reads/writes ahead of prefetch;
- per-Bank accesses, busy cycles, conflicts, queue wait, and maximum queue
  depth;
- per-tensor request counts and byte/beat conservation.

P2 intentionally does not model resident-object capacity allocation. The
resource budget records total capacity, while P3 introduces virtual-object
lifetime allocation and enforces occupied-byte capacity per physical Bank.

## Old simulator boundary

The inherited `banked_memory_system.py` remains unchanged by P2 because it has
uncommitted DATE1 edits. `memdomain_adapter.py` converts a common-schema report
to the P1 `PolicyResult` and rejects any mismatch between `TotalCycles` and its
components. Later integration must explicitly select the new model; it must not
silently replace legacy DATE1 behavior.
