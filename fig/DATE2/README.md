# DATE2 figure/data mapping

| Paper experiment | Source data | Key fields | Generated figure |
|---|---|---|---|
| B1/C1 exp1 | `outputs/DATE2/exp1/layer_characterization.csv`, `accumulator_sensitivity.csv` | per-layer compute/exposed stall, INT8 IA/W/OA bytes, local/spill INT32 accumulator sensitivity, conflict/imbalance, ideal Bank demand | `exp1_cycle_breakdown.pdf`, `exp1_accumulator_sensitivity.pdf` |
| C2 exp2 | `outputs/DATE2/exp2/static_bank_sweep.csv`, `per_stage_best.csv` | all 253 positive IA:W:OA partitions per stage | `exp2_best_static_ratio.pdf` |
| C3 exp3 | `outputs/DATE2/exp3/naive_prefetch_interference.csv` | NoPF/NaivePF by Window and Chunk | `exp3_naive_interference.pdf` |
| IV-B Overall | `outputs/DATE2/overall/{HMoE,Mixtral,MoDSE,Switchtrans}.csv` | `total_cycles`, all stall fields, `bank_conflict_rate`, `bank_imbalance`, hotspot/idle ratios | `exp4_overall_performance.pdf` |
| IV-C Ablation | Reuses `outputs/DATE2/overall/*.csv` | component-wise cumulative and incremental gains | `exp4_cross_model_ablation.pdf` |
| IV-D Exp5 Window/Chunk | `outputs/DATE2/window_chunk/w{0,1,2,4,8,16,32,64}_c{1,2,4,8}.csv` | Raw/Safe cycles, timely/late ratio, interference, occupancy, mapping count | `exp5_window_chunk_heatmap.pdf` |
| IV-E Exp6 Robustness | `outputs/DATE2/robustness/*.csv` | Safe speedup, conflict/imbalance, memory and communication stalls | `exp6_robustness.pdf` |

`outputs/DATE2/summary_all.csv` is the cross-suite index. Use the individual
matrix CSVs for paper plots because they retain all 41 canonical fields.

For exp4-exp6, each variant additionally exports `EXPERT_REPORT.csv`,
`FFN_STAGE_REPORT.csv`, `CHUNK_REPORT.csv`, `BANK_REPORT.csv`,
`REQUEST_REPORT.csv`, `EXPERT_INPUT_REPORT.csv`, and
`MEASURED_SELECTIONS.csv` beneath `outputs/DATE2/expN/<variant>/`.

Regenerate with `MPLCONFIGDIR=/tmp/matplotlib-date2 python3 scripts/DATE2/analyze_date2.py`.
