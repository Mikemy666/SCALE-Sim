# DATE2 figure/data mapping

| Paper experiment | Source data | Key fields | Generated figure |
|---|---|---|---|
| IV-B Overall | `outputs/DATE2/overall/{HMoE,Mixtral,MoDSE,Switchtrans}.csv` | `total_cycles`, all stall fields, `bank_conflict_rate`, `bank_imbalance`, hotspot/idle ratios | `exp4_overall_performance.pdf` |
| IV-C Ablation | `outputs/DATE2/ablation/MoDSE_components.csv` | `total_cycles`, stall breakdown, mapping overhead, selected candidate | `exp5_ablation.pdf` |
| IV-D Window/Chunk | `outputs/DATE2/window_chunk/w{0,1,2,4,8}_c{1,2,4,8}.csv` | Raw/Safe cycles, timely/late ratio, interference, occupancy, mapping count | `exp6_window_chunk_heatmap.pdf` |
| IV-E Robustness | `outputs/DATE2/robustness/*.csv` | Safe speedup, conflict/imbalance, memory and communication stalls | `exp7_robustness.pdf` |

`outputs/DATE2/summary_all.csv` is the cross-suite index. Use the individual
matrix CSVs for paper plots because they retain all 41 canonical fields.

Regenerate with `MPLCONFIGDIR=/tmp/matplotlib-date2 python3 scripts/DATE2/analyze_date2.py`.
