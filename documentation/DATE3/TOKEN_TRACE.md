# DATE3 Token 轨迹说明

Token 轨迹是用于验证确定性专家并行（Expert Parallelism，EP）数据流正确性的辅助产物，不参与论文性能数据和加速比的计算。

## 数据语义

- `TOKEN_ROUTE_TRACE.csv.gz` 是无损的完整路由记录，每一行对应一个 Top-k 路由副本。
- `TOKEN_STAGE_TRACE.csv.gz` 对模拟器中的专家级 FFN1/FFN2 执行区间进行归一化记录。每个阶段只保存一次，避免为路由到该专家的每个 Token 重复保存相同信息。
- `TOKEN_TRACE_INDEX.csv.gz` 将每个路由副本与对应的 FFN1/FFN2 阶段 ID，以及符号化的 Dispatch、Return 和 Combine 事件 ID 关联起来。
- 路由到同一专家的 Token 继承该专家的阶段时间区间。当前模拟器没有为每个 Token 单独建模流水线时间戳，因此轨迹不会虚构不存在的逐 Token 精确时间。
- 对于多层工作负载，`token_id` 是路由契约使用的全局路由实例 ID，`layer_token_id` 是便于阅读和检查的层内 Token ID。

`TOKEN_TRACE_SUMMARY.json` 会检查路由副本数量、Top-k 唯一性、各专家 Token 数量、专家所有权、目标偏移连续性、FFN 阶段完整性与顺序，以及多层路由一致性。将轨迹作为正确性验证依据前，必须确认其中的 `all_checks_pass` 为 `true`。

## 不重新运行性能仿真，单独导出轨迹

```bash
.venv/bin/python scripts/DATE3/export_date3_token_trace.py \
  configs/MoE/DATE3/end_to_end/HMoE.json \
  outputs/DATE3/exp7/HMoE \
  --token-trace full
```

`export_date3_details.py` 在导出 DATE3 详细结果时也会自动生成 Token 轨迹。轨迹提供以下四种详细程度：

- `none`：不生成 Token 轨迹；
- `summary`：只生成检查摘要；
- `sampled`：生成摘要、专家阶段表和少量代表性 Token 样本；
- `full`：生成完整的归一化轨迹，且使用 gzip 压缩，是默认选项。

如果此前使用 `--skip-details` 跳过了详细结果导出，可以使用上面的独立导出命令补充 Token 轨迹，不需要重新运行性能仿真。

## 查看单个 Token

```bash
.venv/bin/python scripts/DATE3/show_token_trace.py \
  outputs/DATE3/exp7/HMoE --token 0
```

可以增加 `--layer N` 或 `--topk-slot N`，分别按照网络层或 Top-k 副本编号筛选重建结果。

## 建议检查顺序

1. 首先查看 `TOKEN_TRACE_SUMMARY.json`，确认 `all_checks_pass` 为 `true`。
2. 查看 `TOKEN_TRACE_SAMPLE.csv`，人工检查具有代表性的本地路由、远程路由和不同专家的 Token。
3. 使用 `show_token_trace.py` 重建指定 Token 的路由与 FFN1/FFN2 执行过程。
4. 需要全面审计时，再解压并检查 `TOKEN_ROUTE_TRACE.csv.gz` 和 `TOKEN_TRACE_INDEX.csv.gz`。
