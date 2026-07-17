# SCALE-Sim Expert Parallel MoE 架构使用说明

本工程在 SCALE-Sim 原有计算阵列、片上存储和 Bank conflict 模型上，增量加入了面向多 GPU Expert Parallelism（EP）的 MoE 仿真能力。原有非 EP 配置和执行入口仍然可用；新架构通过 `EnableEPMoE` 显式启用。

## 主要功能

新架构支持：

1. 固定 expert-to-GPU 映射；
2. Top-K=1 和 Top-K=2 token routing；
3. 一张 detailed GPU 和多张 analytical black-box GPU；
4. MoE 层内多 expert 并行执行；
5. 每 GPU 多 compute engine 调度；
6. 从 SCALE-Sim demand matrix 提取真实 compute tile 和 weight chunk；
7. initial weight load 与 chunk-level prefetch；
8. static/dynamic Bank allocation 及 prefetch interference；
9. black-box GPU 的外部带宽背景压力；
10. token dispatch、output combine 和共享 interconnect 仲裁；
11. expert、GPU、Bank、DRAM、通信及负载不均衡指标；
12. 输入和核心模型源码的 SHA-256 provenance；
13. 四组标准实验、sanity check 和跨报表验证。

## 快速开始

### 安装依赖

推荐在仓库根目录使用可编辑安装：

```bash
pip install -e .
```

也可以直接从源码运行，确保当前目录是仓库根目录即可。

### 运行默认 EP-MoE 实验

```bash
python -m scalesim.scale \
  -c configs/MoE/ep_default.cfg \
  -t topologies/MoE/test.csv \
  -l layouts/conv_nets/test.csv \
  -p outputs/ep_default \
  -i gemm \
  -s N
```

结果位于：

```text
outputs/ep_default/ep_default/
```

参数含义：

- `-c`：架构和实验配置；
- `-t`：网络拓扑；
- `-l`：数据 layout；
- `-p`：输出根目录；
- `-i gemm`：拓扑采用 M/N/K 格式；
- `-s N`：不保存体积较大的逐周期 trace 文件。

## 默认架构

默认配置为：

```text
NumGPUs = 2
DetailedGPUId = 0
BlackBoxGPUIds = 1
ExpertsPerGPU = 4
ComputeEnginesPerGPU = 4
TopK = 1
InitialChunk = 1
ChunkPrefetchWindow = 0
BlackBoxWorkloadMode = analytical
DynamicBankOverhead = old_model
CommunicationModel = latency_plus_bandwidth
PrecisionBytes = 2
```

默认 expert 映射：

```text
GPU 0（detailed）: Expert 0, 1, 2, 3
GPU 1（black-box）: Expert 4, 5, 6, 7
```

GPU 0 的 expert 层进入完整 SCALE-Sim 和本地 Bank 模型。GPU 1 的 expert 层不会执行详细 `single_layer_sim`，只使用 analytical workload，不参与 GPU 0 的片上 Bank conflict。

## 配置说明

EP 相关参数位于配置文件的 `[run_presets]`。

### EP 和并行调度

```ini
EnableEPMoE = True
EnableParallelMoE = True
NumGPUs = 2
DetailedGPUId = 0
BlackBoxGPUIds = 1
ExpertsPerGPU = 4
ComputeEnginesPerGPU = 4
```

- `EnableParallelMoE=False` 时使用单个全局 compute engine，作为顺序基线；
- `NumGPUs × ExpertsPerGPU` 必须与拓扑中的 expert 数一致；
- 当前不支持 expert replication、migration 和动态 placement。

### Routing

```ini
TopK = 1
MoERoutingMode = balanced
MoETokens = 32
RoutingSeed = 40
RoutingSkewFactor = 1.0
RoutingFile =
```

支持的 routing mode：

- `balanced`：确定性的均衡路由；
- `seeded_skewed`：按固定随机种子生成偏斜负载；
- `explicit`：从 CSV 读取逐 token 路由；
- `topology_counts`：兼容旧拓扑中按 expert 指定 token 数量的方式。

显式路由文件格式：

```csv
MoELayerID,TokenID,ExpertIDs
0,0,0|1
0,1,2|3
```

`ExpertIDs` 使用 `|` 分隔，其数量必须等于 `TopK`，同一 token 不能重复选择相同 expert。

### Weight chunk 和 prefetch

```ini
EnableChunkPrefetch = True
InitialChunk = 1
ChunkPrefetchWindow = 1
BlackBoxBandwidthBytesPerCycle = 128
```

执行 tile `i` 时，请求 chunk `i + ChunkPrefetchWindow`。如果 chunk 在使用前完成则为 hit；否则记录 miss 和暴露的等待周期。`ChunkPrefetchWindow=0` 表示不进行有效预取，后续 chunk 按需加载。

旧的跨层 `EnablePrefetch` 与新的 `EnableChunkPrefetch` 不能同时启用。

### Bank 模型

```ini
EnableBankModel = True
EnableDynamic = False
BankConflictPenalty = 4
DynamicBankOverhead = old_model
```

- `EnableDynamic=False`：static Bank allocation；
- `EnableDynamic=True`：复用工程原有 dynamic allocator；
- dynamic mapping overhead 保持原有 `old_model`；
- black-box GPU 不进入 detailed GPU 的 Bank allocation 和 conflict 统计。

### Black-box 背景压力

```ini
EnableBlackBoxBackgroundPressure = False
GlobalMemoryBandwidthBytesPerCycle = 1024
```

启用后，远端 expert 的 analytical weight 和通信流量会形成外部带宽压力，并增加 detailed GPU runtime stall。它不会改变 detailed GPU 的片上 Bank 分配。

### 通信

```ini
CommunicationModel = latency_plus_bandwidth
PrecisionBytes = 2
CommunicationLatencyCycles = 20
CommunicationBandwidthBytesPerCycle = 128
AllowCommPrefetchOverlap = True
```

远端 expert 的通信延迟为：

```text
fixed latency + transfer bytes / bandwidth
```

token dispatch 必须在 expert compute 前完成；output combine 在最后一个 tile 后执行。所有远端传输共享简化的 interconnect 时间线。本地 detailed expert 不产生跨卡通信。

## 拓扑格式

EP-MoE 使用 GEMM M/N/K 拓扑：

```csv
Layer,M,N,K,
Router,32,16,16,
MoE-E0-FF1,4,32,16,
MoE-E0-FF2,4,16,32,
MoE-E1-FF1,4,32,16,
MoE-E1-FF2,4,16,32,
```

expert 层命名支持：

```text
MoE-E3-FF1
MoE-E3-FF2
MoE-L1-E3-FF1
MoE-L1-E3-FF2
```

- `E3` 表示 Expert 3；
- `FF1/FF2` 表示 expert FFN 的两个 GEMM；
- `L1` 用于区分拓扑中连续出现的多个 MoE 层；
- 每个 expert 必须恰好包含一个 FF1 和一个 FF2。

## 执行模型

一次 MoE group 的主要阶段为：

```text
token routing
  -> remote token dispatch
  -> initial weight chunk load
  -> 多 expert / 多 engine tile 调度
  -> chunk prefetch 或 demand load
  -> remote output combine
  -> 最慢 active expert 完成后结束 MoE group
```

detailed GPU 的 tile 边界来自真实 filter demand burst。black-box GPU 根据拓扑、路由 token 数、阵列吞吐、precision 和带宽生成 analytical chunk。两类 chunk 最终进入同一个事件驱动 runtime coordinator。

## 输出文件

除 SCALE-Sim 原有报告外，新架构生成：

- `EP_MOE_CONFIG.csv`：有效 EP 配置；
- `EP_MOE_ROUTING.csv`：逐 token routing；
- `EP_MOE_TIMELINE.csv`：普通层和 MoE group 时间线；
- `EP_MOE_EVENTS.csv`：dispatch、weight、compute、combine 事件；
- `EP_MOE_RUNTIME_STATE.csv`：逐 expert 最终 runtime 状态；
- `EP_MOE_CHUNKS.csv`：tile/chunk、地址、字节和访问统计；
- `EP_MOE_LAYER_EXECUTION.csv`：每层是 detailed 还是 black-box；
- `EP_MOE_BANK_ALLOCATION.csv`：detailed expert 的 Bank 分配；
- `EP_MOE_REPORT.csv`：逐 expert 综合结果；
- `EP_MOE_SUMMARY.csv`：逐 MoE group 汇总；
- `EP_MOE_RUN_MANIFEST.csv`：输入和核心模型源码 SHA-256。

EP 实验应以 `EP_MOE_TIMELINE.csv`、`EP_MOE_RUNTIME_STATE.csv`、`EP_MOE_REPORT.csv` 和 `EP_MOE_SUMMARY.csv` 为权威结果。旧层级报告中，black-box 层为 analytical/零物理访问兼容行。

## 四组核心实验

一键运行 static/dynamic × prefetch on/off：

```bash
python run_ep_moe_experiments.py --output outputs/ep_moe_matrix
```

使用的配置：

```text
configs/MoE/ep_experiments/static_no_prefetch.cfg
configs/MoE/ep_experiments/static_prefetch.cfg
configs/MoE/ep_experiments/dynamic_no_prefetch.cfg
configs/MoE/ep_experiments/dynamic_prefetch.cfg
```

汇总结果：

```text
outputs/ep_moe_matrix/EP_MOE_EXPERIMENT_MATRIX.csv
outputs/ep_moe_matrix/EP_MOE_EXPERIMENT_COMPARISONS.csv
```

comparison 文件提供 signed cycle delta 和 speedup，不假设 prefetch 一定带来加速。

## Sanity check

```bash
python run_ep_moe_sanity.py --output outputs/ep_moe_sanity
```

自动检查：

- 单 GPU 时不存在 black-box expert；
- 关闭 prefetch 时不存在 prefetch hit/miss/traffic；
- 关闭背景压力时 stall 为零；
- 开启背景压力后产生正的 pressure stall；
- sequential MoE 不会快于 parallel MoE；
- dynamic overhead 保持 `old_model`。

脚本支持复用输出目录中已经完成的 case，从中断位置继续。

## 报告一致性验证

```bash
python validate_ep_moe_reports.py \
  outputs/ep_moe_matrix/ep_dynamic_prefetch
```

验证器检查：

- timeline 连续性和周期计算；
- MoE finish 是否等于最慢 active expert；
- routing 是否包含 Top-K 个不同 expert；
- runtime 与 chunk 数量是否一致；
- detailed 权重逻辑地址是否重叠；
- event 是否按周期排序；
- expert、runtime、summary 指标是否一致；
- manifest 中的输入和模型源码哈希是否匹配。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 legacy static/dynamic/non-bank golden、配置 round-trip、Top-K routing、并行调度、trace chunk、prefetch、通信、实验矩阵、sanity、precision、报告一致性和确定性。

## 常见问题

### 配置提示 expert 数量不一致

检查：

```text
NumGPUs × ExpertsPerGPU == topology 中的 expert 数
```

并确保每个 expert 都同时存在 FF1 和 FF2。

### 开启 prefetch 后反而变慢

这是允许出现的结果。prefetch 可以减少 weight loading stall，也会增加 Bank/bandwidth interference。请同时查看：

```text
PrefetchHitRate
PrefetchMissStall
PrefetchBankInterferenceStall
PrefetchBandwidthOverhead
```

### 为什么 black-box 层在旧报告中访问量为零

black-box 层不会执行详细 SCALE-Sim，这是架构边界。其 workload、weight traffic 和通信结果位于 EP 专用报告中。

### 如何确认两次实验使用了相同代码

比较两个输出目录中的 `EP_MOE_RUN_MANIFEST.csv`。该文件同时记录配置、拓扑、layout、runtime、config parser 和 Bank model 的 SHA-256。

## 模型边界

当前模型有意不支持：

- expert replication、migration、动态 placement；
- 多张 detailed GPU；
- packet-level 或拓扑感知网络模拟；
- 完整 DRAM controller timing；
- energy/power 和故障行为；
- 自动硬件校准。

通信模型是共享的 latency-plus-bandwidth 约束模型，black-box workload 是 analytical 估算。使用实验结果时应在论文或报告中明确这些假设。
