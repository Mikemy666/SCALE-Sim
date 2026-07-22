# P4 Chunk residency and prefetch outcomes

P4 models Weight Chunks as explicit runtime objects rather than inferring
prefetch quality from a hit counter. Every Chunk follows a legal state path:

```text
registered -> loading -> consumed -> released
                      -> released (unused prefetch)
           -> canceled (before transfer finalization)
```

Transfer completion comes from the P2 unified Bank service path. Capacity is
reserved and released through the P3 virtual mapping table. Consumption uses a
declared use cycle and classifies a prefetched Chunk as timely or late; an
unconsumed completed prefetch becomes unused only when explicitly released.

Reported metrics include demand/prefetch traffic, timely/late/unused counts and
ratios, demand misses, exact miss stall, coverage, accuracy, prefetch occupancy
byte-cycles, and compute-transfer overlap computed from interval unions.
