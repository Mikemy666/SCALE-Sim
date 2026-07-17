# P2 MoE topology and routing

P2 makes token routing the source of truth for active experts and
`tokens_per_expert`. The removed `topk_prefix` behavior did not model token
routing and is now rejected in EP mode. `ActiveExpertIds` is also rejected; use
an explicit routing file when only a subset of experts should be active.

## Topology names

Legacy names remain supported:

```text
MoE-E3-FF1
MoE-E3-FF2
```

For multiple adjacent MoE layers, use explicit layer IDs:

```text
MoE-L0-E3-FF1
MoE-L0-E3-FF2
MoE-L1-E3-FF1
MoE-L1-E3-FF2
```

Every MoE layer must define every configured expert exactly once with one FF1
and one FF2 entry. Validation runs before detailed layer simulation.

## Routing modes

- `topology_counts`: preserves legacy GEMM M values as per-expert token counts.
- `balanced`: distributes `MoETokens` deterministically across experts.
- `seeded_skewed`: samples a reproducible skewed distribution using
  `RoutingSeed` and `RoutingSkewFactor`.
- `explicit`: reads exact token assignments from `RoutingFile`.

An explicit routing CSV has this format:

```csv
MoELayerID,TokenID,ExpertIDs
0,0,0|4
0,1,1|5
```

`ExpertIDs` contains exactly `TopK` distinct IDs separated by `|`. For Top-1 it
contains one ID. For Top-2 each token contributes two expert assignments, so the
sum of `tokens_per_expert` is `2 × MoETokens`.

Every run writes `EP_MOE_ROUTING.csv`, allowing token counts and active experts
to be independently reconstructed. Balanced and seeded routing are deterministic
for a fixed configuration and seed.
