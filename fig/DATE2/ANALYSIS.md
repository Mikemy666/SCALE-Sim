# DATE2 results analysis

## Current modeling contract

- Original model format: FP32.
- Simulator IA and Weight: INT8.
- MAC: INT8 x INT8 with INT32 PE-local accumulation.
- Requantized output: INT8.
- INT32 accumulator spill is reported only by Exp1 sensitivity analysis.
- Weight traffic is not divided by eight.

Outputs generated before this contract are stale and must not be used in paper
figures. Workload-hash validation rejects those matrices after configs are
regenerated.

## Experiment mapping

- Exp1: layer bottleneck, flow characterization, and accumulator sensitivity.
- Exp2: exhaustive static IA:Weight:OA Bank partition sweep.
- Exp3: naive-prefetch interference over the Exp5 parameter matrices.
- Exp4: four-model overall comparison and component ablation.
- Exp5: Window x Chunk sensitivity, 32 parameter points.
- Exp6: model/routing/expert/token/Top-k/EP robustness.

For Exp4-Exp6, each variant additionally exports expert, FFN-stage, chunk,
physical-Bank, request, routing-input, dominance, and measured-selection
reports beneath `outputs/DATE2/expN/<variant>/`.

## Publication readiness

The revised Exp1 local-accumulator result supports the intended motivation
without relying on spill traffic: MoE mean exposed memory stall is 3,084
cycles versus 2,925 compute cycles, while mean stall ratio is 53.5% for MoE
and 32.1% for non-MoE.

Exp2 also regenerates successfully under the same precision contract. Exp3-
Exp6 must be rerun before their old numerical conclusions or figures are
quoted.
