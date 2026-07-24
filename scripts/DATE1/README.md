# DATE1 实验执行脚本

以下脚本可以在任意工作目录下执行。每个脚本会运行对应实验组中的全部配置：

```bash
./scripts/DATE1/run_exp1.sh
./scripts/DATE1/run_exp2.sh
./scripts/DATE1/run_exp3.sh
./scripts/DATE1/run_exp4.sh
./scripts/DATE1/run_exp5.sh
./scripts/DATE1/run_exp6.sh
./scripts/DATE1/run_exp7.sh
```

每个脚本都会将额外的命令行参数传递给 `run_date1_experiments.py`。

`run_exp2.sh` 会先执行 24-Bank 空间内全部 253 种静态 Bank 组合的单次 trace 重放扫描，随后自动选择性能差异明显的代表配置。实验3--7统一采用实验2得到的静态最优基线 `IA:Weight:OA = 4:14:6`（总计24个Bank）。

实验3--7中依赖静态基线的配置统一使用 `4:14:6`。实验4对应方案名称为 `static_best_4_14_6` 和 `dynamic_24`；三组方案的 Bank 总数均为24。

实验7启用 `EnableRoutedTokenAwareTrace=True`。详细GPU上每个活跃专家的GEMM M维会使用实际 `RoutedTokens`，并输出 `EP_MOE_ROUTED_TRACE.csv` 记录 `OriginalM`、`EffectiveM` 和是否缩放。架构升级前的实验7结果不能与新结果混用，需要完整重跑实验7。

只查看将要执行的命令，不实际运行仿真：

```bash
./scripts/DATE1/run_exp6.sh --dry-run
```

只运行实验组中的某一个方案：

```bash
./scripts/DATE1/run_exp5.sh --variant dynamic_prefetch
```

实验结果将按照以下结构保存：

```text
outputs/DATE1/expN/方案名称/
```

例如：

```text
outputs/DATE1/exp5/dynamic_prefetch/
```
