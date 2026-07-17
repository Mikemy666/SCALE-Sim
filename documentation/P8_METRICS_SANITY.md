# P8: Experiment metrics and architectural sanity checks

P8 completes the main EP summary with metrics derived from runtime state and
the existing detailed memory reports:

- GPU load imbalance and average/minimum workload utilization;
- average/maximum detailed bank capacity utilization;
- detailed DRAM traffic and estimated black-box DRAM traffic;
- separate weight-loading, prefetch-miss, bank-interference, communication,
  expert-waiting, and background-pressure stalls.

`run_ep_moe_sanity.py` runs four degradation cases and fails on violated model
boundaries:

1. one GPU produces no black-box experts;
2. disabled prefetch produces no hit, miss, or prefetch traffic;
3. disabled black-box pressure produces no background stall;
4. enabled pressure produces a positive background stall;
5. sequential MoE does not finish earlier than parallel MoE;
6. dynamic bank overhead remains `old_model`.

The runner resumes from complete case reports in its output directory, which is
useful for longer simulations.

```sh
python run_ep_moe_sanity.py --output outputs/ep_moe_sanity
```
