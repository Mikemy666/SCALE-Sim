# P10: Cross-report validation and Top-K=2 integration

`validate_ep_moe_reports.py` validates one completed EP run as a coherent model,
not merely as independently readable CSV files. It checks:

- timeline contiguity and duration arithmetic;
- MoE finish equals the slowest active expert finish;
- summary group time equals timeline duration;
- every routing row contains exactly `TopK` unique experts;
- runtime chunk counts equal emitted chunk rows;
- detailed layers have non-overlapping logical weight ranges;
- event rows are cycle ordered.

Use it after any experiment:

```sh
python validate_ep_moe_reports.py outputs/ep_moe_matrix/ep_dynamic_prefetch
```

The tracked `ep_validation_topk2.cfg` and `validation_topk2.csv` form a small,
fast end-to-end Top-K=2 workload. It is executed by the test suite and covers
routing, detailed/black-box execution, chunk prefetch, communication, reports,
and the validator itself.
