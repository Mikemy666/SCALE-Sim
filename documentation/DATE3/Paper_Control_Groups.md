# DATE3 paper control groups

DATE3 uses one public identity per mechanism. PIVOT is MemDomain; the mapping
and fixed-prefetch variants below are ablations of PIVOT, not separate proposed
architectures.

| CSV identity | Paper label | Mapping | Prefetch | Role |
|---|---|---|---|---|
| `Static-555-NoPF` | Static-555 | fixed 5/5/5/15 | none | legacy anchor |
| `Static-Opt-NoPF` | Static-Opt | one model-wide fixed four-domain plan | none | strong static control |
| `Dynamic-NoPF` | PIVOT-Map | dynamic unified mapping | none | mapping ablation |
| `Static-Opt-FixedPF` | Static-Opt+PF | same Static-Opt plan | fixed | static prefetch control |
| `Dynamic-FixedPF` | PIVOT-Map+PF | dynamic unified mapping | the same fixed work | non-joint ablation |
| `PIVOT` | PIVOT | dynamic unified mapping | online joint selection | final scheme |
| `Ideal-NoPF` | Ideal | conflict-free NoPF reference | none | non-implementable reference |

`Static-Opt` is compiled once from all expert FFN GEMMs and remains frozen for
the complete execution. It may not choose a new allocation per layer. Dynamic
mapping contains that fixed plan as its measured incumbent. Static and Dynamic
fixed-prefetch controls issue the same Window/Chunk workload.

Dynamic executes the compiler plan at every expert FFN boundary: IA, Weight,
OA, and ACC receive disjoint physical candidate pools whose widths follow that
stage's plan. A model-level guard permits a small local FFN regression only
when the complete-model cycle count improves over Static-Opt. PIVOT additionally
coordinates physical pool rotations with future prefetch residency; the
non-joint Dynamic-FixedPF control performs the same fixed prefetch work without
that coordination.

Exp4 publishes the four mapping-only rows. Exp5, Exp6, and Exp7 publish the six
implementable rows. The Ideal row is excluded from implementable win/tie/loss
statistics.
