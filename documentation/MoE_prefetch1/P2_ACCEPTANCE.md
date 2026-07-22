# P2 acceptance report

P2 is accepted when the full source test suite proves:

- IA, Weight, OA, and Accumulator can contend for the same physical Bank;
- prefetch and compute share the same path, with compute priority;
- larger requests stripe across multiple physical Banks;
- bytes and Bank beats are conserved;
- port count changes same-cycle conflict behavior;
- request-buffer occupancy never exceeds its configured depth;
- invalid Bank placement is rejected;
- simulation is independent of input iteration order;
- report accounting drift is rejected before policy comparison.

The executable checks are in `tests/test_memdomain_p2_unified_bank.py`. P2 is a
standalone simulator component; end-to-end configuration wiring is deferred
until the virtual mapping semantics in P3 are available.
