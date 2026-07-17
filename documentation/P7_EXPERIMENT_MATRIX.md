# P7: Canonical static/dynamic and prefetch experiment matrix

The canonical matrix is driven by `run_ep_moe_experiments.py` and the four
tracked configurations under `configs/MoE/ep_experiments`:

- static without prefetch;
- static with chunk prefetch;
- dynamic without prefetch;
- dynamic with chunk prefetch.

The runner writes `EP_MOE_EXPERIMENT_MATRIX.csv` and rejects an incomplete
matrix, a changed `DynamicBankOverhead` model, or prefetch traffic in a
no-prefetch run.

Summary prefetch fields now come from requests issued by the runtime event
coordinator. Demand loading with prefetch disabled is reported separately as
`WeightLoadingStall`; it is not mislabeled as a prefetch miss. Detailed bank
conflict and prefetch interference values still come from the existing bank
model.

Run from the repository root:

```sh
python run_ep_moe_experiments.py --output outputs/ep_moe_matrix
```
