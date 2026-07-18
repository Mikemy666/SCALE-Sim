# 毕业设计 + DATE 论文统一实验方案（简化版）

## 1. 总体实验逻辑

本文只研究单核上的 MoE 片上存储优化。其他 GPU 保持黑箱建模，用于补全专家执行时间，不分析 GPU 间通信优化。

整篇 DATE 论文的实验逻辑如下：

1. MoE 层是主要访存瓶颈。
2. 瓶颈的主要原因是 MoE 的动态访存需求与静态 Bank 分配不匹配。
3. 普通预取虽然能提前搬运数据，但也会增加 Bank 访问压力，甚至与正常请求发生冲突。
4. MemDomain 使用动态 Bank 分配缓解冲突。
5. 动态 Bank 分配与预取协同后取得更好的性能。
6. 最后验证方案对预取窗口、Chunk 粒度、路由分布和专家配置是否稳定。

实验 1、2 使用完整 MoDSE 网络，包括 Attention、Router 和 MoE 层。实验 3–7 只运行 MoDSE 的 MoE 专家层。实验 8 使用已有 DC 脚本，不在本方案中展开。

---

## 2. 当前架构能否完成这些实验

| 实验 | 当前能否完成 | 判断 |
|---|---|---|
| 实验1：MoE 层是主要访存瓶颈 | 可以 | 已有完整 MoDSE topology、逐层计算报告和 Bank 报告 |
| 实验2：静态分配无法适应动态数据流 | 可以，但需要准确表述 | 可以证明不同 MoE 层需要不同 IA/Weight/OA Bank 比例，固定比例造成更多冲突；当前不能直接画逐物理 Bank 的热点和空闲时间图 |
| 实验3：普通预取产生新的 Bank 冲突 | 可以证明“预取请求增加 Bank 争用”；不能完整证明“预取数据占据错误位置” | 当前预取请求与正常 Weight 请求共享 Bank，并输出 prefetch interference；但没有建模预取数据驻留、替换和物理放置冲突 |
| 实验4：动态分配缓解 Bank 冲突 | 可以 | 已有 Static/Dynamic 开关和逐层 Bank allocation/conflict 数据 |
| 实验5：动态分配与预取协同消融 | 可以 | 已有 Static/Dynamic × Prefetch On/Off 四组配置和批量运行脚本 |
| 实验6：预取窗口敏感性 | 可以 | `ChunkPrefetchWindow` 可以直接修改 |
| 实验6：Chunk 粒度敏感性 | 当前不能严格完成 | 当前有 Chunk 数和窗口，但没有独立 `ChunkSize` 配置参数 |
| 实验7：路由分布敏感性 | 可以 | 支持 balanced、seeded_skewed、explicit、topology_counts |
| 实验7：专家配置敏感性 | 可以，但要准备对应 topology | 支持 Top-k=1/2、Token 数和专家数；改变专家数时必须生成对应的专家层 topology |
| 实验8：DC | 已有独立流程 | 本文档不展开 |

---

# II. Background and Motivation

## 实验1：验证访存瓶颈主要集中在 MoE 层

### 目的

比较完整 MoDSE 网络中的 Attention、Router 和 MoE 层，证明 MoE 专家层消耗了更多访存等待周期。

### 使用网络

运行完整 MoDSE 网络，不能只运行 MoE 层。

沿用毕业设计配置：

- 8 个专家。
- 输入 Token 数为 256。
- Hidden dimension 为 384。
- Systolic Array 为 64×64。
- 总 Bank 数为 24。
- Static Bank 分配为 8:8:8。
- 单 Bank 带宽为 128 bit/cycle。
- 单 Bank 端口数为 1。
- Dataflow 为 WS。

### 需要改变的参数

这个实验不需要扫参数，只运行一组固定静态基线：

```ini
EnableDynamic = False
EnableChunkPrefetch = False
```

### 需要读取的数据

从 `COMPUTE_REPORT.csv` 读取：

- `LayerID`
- `Total Cycles`
- `Stall Cycles`
- `Compute Util %`

从 `BANK_MODEL_REPORT.csv` 读取：

- `stall_cycles_due_to_bank_conflict`
- `total_bank_conflict_delay`
- `ifmap_bank_conflict_delay`
- `filter_bank_conflict_delay`
- `ofmap_bank_conflict_delay`

计算：

```text
Stall Ratio = Stall Cycles / Total Cycles
Bank Conflict Share = stall_cycles_due_to_bank_conflict / Stall Cycles
```

### 画什么图

直接采用一张分组柱状图，横轴只分三类模块：

- Attention
- Router
- MoE

每类画两个柱：

- 总 Stall Cycles
- 平均 Stall Ratio

再加一个小表格列出三类模块的 Total Cycles、Stall Cycles 和 Bank Conflict Share。

### 采用图还是表

**正文用柱状图，旁边放一个小表格。**

不建议继续画 23 个层的大柱状图作为正文主图，因为信息过多。逐层图放到附录。

### 预期结论

MoE 层的总 Stall Cycles 和平均 Stall Ratio 明显高于 Attention 与 Router，且大部分 Stall 与 Bank conflict 有关，因此后续实验只研究 MoE 专家层是合理的。

---

## 实验2：验证静态分配无法适应动态数据流

### 目的

证明不同专家、不同 FFN 层对 IA、Weight、OA 的需求比例不同，因此不存在一个固定 Bank 比例可以同时适合所有 MoE 层。

### 使用网络

运行完整 MoDSE 网络，但分析时重点读取其中的 MoE 层。

### 需要改变的参数

固定总 Bank 数为 24，关闭预取，扫描静态 Bank 比例。

建议保留以下 7 组，数量足够且容易解释：

| 类型 | IA:Weight:OA |
|---|---:|
| 均衡 | 8:8:8 |
| Weight-heavy | 5:15:4 |
| Weight-heavy | 7:14:3 |
| IA-heavy | 14:7:3 |
| IA-heavy | 11:7:6 |
| OA-heavy | 5:7:12 |
| OA-heavy | 3:7:14 |

所有组设置：

```ini
EnableDynamic = False
EnableChunkPrefetch = False
```

### 需要读取的数据

从 `BANK_MODEL_REPORT.csv` 读取：

- `LayerID`
- `ifmap_banknum`
- `filter_banknum`
- `ofmap_banknum`
- `ifmap_elements`
- `filter_elements`
- `ofmap_elements`
- `ifmap_capacity_utilization`
- `filter_capacity_utilization`
- `ofmap_capacity_utilization`
- `ifmap_bank_conflict_delay`
- `filter_bank_conflict_delay`
- `ofmap_bank_conflict_delay`
- `stall_cycles_due_to_bank_conflict`
- `total_cycles`

### 画什么图

画一张热力图：

- 横轴：7 种静态 Bank 比例。
- 纵轴：MoE 专家层。
- 颜色：该层的归一化 `total_cycles` 或 `stall_cycles_due_to_bank_conflict`。

每一行使用该层所有静态配置中的最小值归一化为 1。

再画一张小型堆叠柱状图：

- 横轴：MoE 专家层。
- 纵轴：该层表现最好的 IA/Weight/OA Bank 比例。

### 采用图还是表

**正文使用热力图和最佳比例堆叠柱状图，不使用大表。**

### 预期结论

如果不同 MoE 层的最佳静态比例不同，并且同一个静态比例在部分层表现很好、在另一些层表现很差，就能证明固定 Bank 划分无法适应 MoE 的动态访存需求。

### 注意

当前报告中的三类 `capacity_utilization` 是 IA/Weight/OA 三个逻辑 Bank 池的容量压力，不是逐物理 Bank 利用率。因此本实验可以证明“资源比例失配”，但不要直接写成已经测得“某个具体物理 Bank 长期空闲或热点”。

---

## 实验3：验证普通预取会产生新的 Bank 冲突

### 目的

证明普通预取会提前发出 Weight Bank 请求。预取请求与正常 IA/Weight/OA 请求共享 Bank 资源时，可能增加 Bank 冲突和 Stall。

### 使用网络

只运行 MoDSE 的 MoE 专家层。

### 当前架构可以证明的内容

当前 `banked_memory_system.py` 会让低优先级 Weight prefetch 请求与正常 Weight 请求共享 Bank 模型，并输出：

- 预取 Bank 请求数。
- 预取导致的 Bank interference cycles。
- 预取导致的额外 Stall。
- 预取请求自身的 Bank conflict cycles。

因此可以证明：

> 普通预取增加了 Bank 访问压力，并可能与正常请求发生 Bank 争用。

当前不能严格证明：

> 预取数据被放入了一个错误物理位置，因为数据驻留或替换，导致后续数据无处存放。

这是因为当前模型主要模拟访问请求的排队冲突，没有完整模拟预取数据在 SRAM 中的驻留、替换和覆盖。

### 需要改变的参数

使用 Static 8:8:8，保持动态分配关闭，改变预取开关和窗口：

| 方案 | EnableChunkPrefetch | ChunkPrefetchWindow |
|---|---:|---:|
| No Prefetch | False | 0 |
| Prefetch W1 | True | 1 |
| Prefetch W2 | True | 2 |
| Prefetch W4 | True | 4 |

其他参数保持不变：

```ini
EnableDynamic = False
InitialChunk = 1
```

### 需要读取的数据

从 `EP_MOE_SUMMARY.csv` 读取：

- `MoEGroupTime`
- `TotalPrefetchHit`
- `TotalPrefetchMiss`
- `AvgPrefetchHitRate`
- `TotalPrefetchMissStall`
- `TotalWeightLoadingStall`
- `TotalPrefetchBankInterferenceStall`
- `TotalPrefetchBandwidthOverhead`
- `TotalUsefulPrefetchTraffic`
- `TotalUselessPrefetchTraffic`

从 `EP_MOE_BANK_ALLOCATION.csv` 读取：

- `LayerBankConflictStall`
- `RuntimePrefetchBankRequests`
- `RuntimePrefetchBankInterferenceStall`
- `RuntimePrefetchBankInterferenceCycles`
- `PrefetchAwareCombinedBankCycles`

### 画什么图

画一张双轴图：

- 横轴：No Prefetch、W1、W2、W4。
- 左轴柱状图：`MoEGroupTime`。
- 右轴折线：`TotalPrefetchBankInterferenceStall`。

再画一张分专家或分层柱状图，展示 W=0 与出现最大 interference 的窗口下 `LayerBankConflictStall` 的差值。

### 采用图还是表

**正文使用两张图，不使用大表。**

### 预期结论

预取窗口增大后，Weight loading stall 可能下降，但 prefetch bank interference 和 bandwidth overhead 上升。当预取过于激进时，总周期可能不再下降，甚至变差，说明普通预取会引入新的 Bank 争用。

---

# IV. Evaluation

## 实验4：验证动态分配是否缓解 Bank 冲突

### 目的

直接比较静态和动态 Bank 分配，验证动态分配是否降低冲突和执行时间。

### 使用网络

只运行 MoDSE 的 MoE 专家层。

### 需要改变的参数

关闭预取，只改变 Bank 分配方式：

| 方案 | EnableDynamic | Static 比例 |
|---|---:|---:|
| Static-Equal | False | 8:8:8 |
| Best-Static | False | 从实验2选出的整体最佳固定比例 |
| Dynamic | True | 总 Bank 数 24，运行时动态选择 |

共同设置：

```ini
EnableChunkPrefetch = False
ChunkPrefetchWindow = 0
```

### 需要读取的数据

从 `EP_MOE_SUMMARY.csv` 读取：

- `MoEGroupTime`
- `TotalExpertWaitingCycles`
- `ExpertCycleImbalance`
- `AverageDetailedBankCapacityUtilization`
- `TotalDynamicFallbackCount`

从 `EP_MOE_BANK_ALLOCATION.csv` 读取：

- `LayerTotalCycles`
- `LayerStallCycles`
- `LayerBankConflictStall`
- `IfmapBankNum`
- `FilterBankNum`
- `OfmapBankNum`
- `AllocationRatio`

### 画什么图

主图采用三个方案的分组柱状图，分别展示：

- 归一化 `MoEGroupTime`。
- 归一化 `LayerBankConflictStall` 总和。
- `TotalExpertWaitingCycles`。

第二张图画 Dynamic 每个 MoE 层实际选择的 IA/Weight/OA Bank 数堆叠柱状图。

### 采用图还是表

**正文使用两张柱状图。**

表格只列出三个方案的原始数值和 Dynamic speedup。

### 预期结论

Dynamic 的 Bank conflict stall 和总周期低于 Static-Equal，并且优于整体 Best-Static。动态方案在不同层选择不同的 IA/Weight/OA 比例，说明收益来自按需分配，而不是单纯选择了另一个固定比例。

---

## 实验5：预取协同优化消融实验

### 目的

分别测量动态 Bank 分配和预取的独立收益，再验证二者组合是否最好。

### 使用网络

只运行 MoDSE 的 MoE 专家层。

### 需要改变的参数

直接使用当前已有的四组配置：

| 方案 | EnableDynamic | EnableChunkPrefetch | Window |
|---|---:|---:|---:|
| Static-NoPrefetch | False | False | 0 |
| Static-Prefetch | False | True | 1 |
| Dynamic-NoPrefetch | True | False | 0 |
| Dynamic-Prefetch | True | True | 1 |

对应目录已经存在：

```text
configs/MoE/ep_experiments/
```

批量脚本已经存在：

```text
run_ep_moe_experiments.py
```

### 需要读取的数据

优先读取脚本生成的：

- `EP_MOE_EXPERIMENT_MATRIX.csv`
- `EP_MOE_EXPERIMENT_COMPARISONS.csv`

核心指标：

- `MoEGroupTime`
- `TotalExpertWaitingCycles`
- `AvgPrefetchHitRate`
- `TotalPrefetchMissStall`
- `TotalPrefetchBankInterferenceStall`
- `TotalPrefetchBandwidthOverhead`
- `DetailedBankConflictStall`

### 画什么图

画一张四柱图，横轴为四种方案，纵轴为归一化 `MoEGroupTime`。

柱子内部建议堆叠：

- 基础计算时间。
- Weight loading/prefetch miss stall。
- Bank conflict/interference stall。

如果这些 Stall 分量在当前定义中有重叠，不能直接堆叠，应改为并列三张小柱状图。

### 采用图还是表

**正文用四柱图，旁边放四行消融表。**

### 预期结论

- Static-Prefetch vs Static-NoPrefetch：普通预取的独立效果。
- Dynamic-NoPrefetch vs Static-NoPrefetch：动态分配的独立效果。
- Dynamic-Prefetch vs Dynamic-NoPrefetch：动态环境中的预取效果。
- Dynamic-Prefetch 最好：说明两种机制可以协同工作。

注意：如果 Dynamic-Prefetch 的组合收益没有超过两个独立收益之和，仍然可以写“组合后性能最好”，但不要写成数学意义上的超加性协同。

---

## 实验6：预取窗口和 Chunk 粒度敏感性

### 6A. 预取窗口敏感性——当前可以直接完成

只运行 Dynamic 模式，扫描：

```text
ChunkPrefetchWindow = 0, 1, 2, 4, 8
InitialChunk = 1
```

其中 Window=0 对应关闭预取。

读取：

- `MoEGroupTime`
- `AvgPrefetchHitRate`
- `TotalPrefetchMissStall`
- `TotalWeightLoadingStall`
- `TotalPrefetchBankInterferenceStall`
- `TotalPrefetchBandwidthOverhead`
- `TotalUsefulPrefetchTraffic`
- `TotalUselessPrefetchTraffic`

画一张两部分组合图：

- 上图：Window vs normalized MoEGroupTime。
- 下图：Window vs miss stall、bank interference、bandwidth overhead 三条折线。

**使用图片展示。**

### 6B. Chunk 粒度敏感性——当前不能严格完成

当前 `InitialChunk` 表示初始加载多少个 Chunk，`ChunkPrefetchWindow` 表示提前多少个 Chunk；它们都不表示每个 Chunk 的大小。

需要新增配置：

```ini
ChunkSizeBytes = 4096 / 8192 / 16384 / 32768
```

或者：

```ini
TilesPerChunk = 1 / 2 / 4 / 8
```

修改后进行二维扫描：

- Chunk size：4/8/16/32 KB。
- Window：1/2/4/8。

读取相同的 prefetch 指标，并额外读取：

- `WeightChunkCount`
- `WeightBytes`
- `WeightLoadCycles`
- 每个 Chunk 的 issue、ready、consume 周期。

主图使用二维热力图：

- 横轴：Window。
- 纵轴：Chunk size。
- 颜色：normalized MoEGroupTime。

再用一张小图显示最优点的 hit rate 和 interference。

**使用热力图，不使用大表。**

---

## 实验7：路由分布与专家配置敏感性

### 目的

验证 Dynamic-Prefetch 相对 Static-NoPrefetch 的优势是否能保持在不同路由和专家配置下。

### 使用网络

只运行 MoDSE 的 MoE 专家层。

### 7A. 路由分布

固定 8 experts、Top-k=1、Tokens=256，改变：

| 路由类型 | 配置 |
|---|---|
| 均衡 | `MoERoutingMode = balanced` |
| 轻度不均衡 | `seeded_skewed`，较小 skew |
| 中度不均衡 | `seeded_skewed`，中等 skew |
| 重度不均衡 | `seeded_skewed`，较大 skew |
| MoDSE 分布 | `explicit` 或 `topology_counts` |

每个 seeded_skewed 配置至少运行 5 个 `RoutingSeed`。

不要只使用输入的 `RoutingSkewFactor` 作为横轴，应读取 `EP_MOE_SUMMARY.csv` 中实际得到的：

- `ExpertTokenImbalance`
- `ExpertCycleImbalance`
- `GPULoadImbalanceCycles`

主图：

- 横轴：实际 `ExpertTokenImbalance`。
- 纵轴：Dynamic-Prefetch 相对 Static-NoPrefetch 的 speedup。
- 每个点代表一个 routing seed。

**使用散点图。**

### 7B. Top-k

改变：

```text
TopK = 1, 2
```

保持专家数和输入 Token 数不变。

读取：

- `MoEGroupTime`
- `TotalExpertCycles`
- `TotalExpertWaitingCycles`
- `DetailedDRAMTrafficBytes`
- `TotalPrefetchBandwidthOverhead`
- `LayerBankConflictStall`

画两组柱状图，比较 Static-NoPrefetch 与 Dynamic-Prefetch 在 Top-k=1/2 下的归一化周期。

**使用柱状图。**

注意 Top-k=2 会增加总 expert assignments，必须同时报告总工作量或 traffic，不能仅把周期增加解释为架构退化。

### 7C. Token 数

改变：

```text
MoETokens = 32, 128, 256, 512
```

画折线图：

- 横轴：Token 数。
- 纵轴：Static-NoPrefetch 和 Dynamic-Prefetch 的 MoEGroupTime。
- 另一小图画 speedup。

**使用折线图。**

### 7D. Expert 数量

改变：

```text
NumExperts = 4, 8, 16
```

当前专家总数由 `NumGPUs × ExpertsPerGPU` 和 topology 共同决定，因此不能只修改一个数字。需要准备 4、8、16 expert 对应的 MoE topology，并保证 LayerID 与 ExpertID 映射正确。

为了符合单核研究：

- `DetailedGPUId = 0`。
- GPU 0 使用详细 SCALE-Sim 模型。
- 其他 GPU 保持 black-box expert model。
- 不分析 inter-GPU communication 优化。
- 所有实验固定相同 communication 配置，避免它成为变量。

建议主图：

- 横轴：4/8/16 experts。
- 纵轴：Dynamic-Prefetch 相对 Static-NoPrefetch 的 speedup。
- 同图或副图显示 `ExpertTokenImbalance`。

**使用折线图。**

### 实验7最终需要读取的公共数据

从 `EP_MOE_SUMMARY.csv`：

- `MoEGroupTime`
- `NumExperts`
- `NumActiveExperts`
- `TopK`
- `ExpertTokenImbalance`
- `ExpertCycleImbalance`
- `TotalExpertWaitingCycles`
- `AverageDetailedBankCapacityUtilization`
- `TotalPrefetchBankInterferenceStall`

从 `EP_MOE_ROUTING.csv`：

- `TokenID`
- `ExpertIDs`
- `TopK`
- `RoutingMode`

从 `EP_MOE_BANK_ALLOCATION.csv`：

- `LayerBankConflictStall`
- `AllocationRatio`
- `IfmapBankNum`
- `FilterBankNum`
- `OfmapBankNum`

---

## 3. 最终建议保留的 DATE 图表

| 编号 | 实验 | 最终展示形式 |
|---|---|---|
| Fig. 1 | 实验1：MoE vs Attention/Router | 模块级分组柱状图 + 小表 |
| Fig. 2 | 实验2：静态比例失配 | Layer × Static Ratio 热力图 |
| Fig. 3 | 实验2：每层最佳 Bank 比例 | IA/W/OA 堆叠柱状图 |
| Fig. 4 | 实验3：普通预取产生 Bank interference | latency 柱 + interference 折线 |
| Fig. 5 | 实验4：Static vs Best-Static vs Dynamic | latency/conflict 分组柱状图 |
| Fig. 6 | 实验4：Dynamic 实际分配 | 每层 IA/W/OA Bank 堆叠图 |
| Fig. 7 | 实验5：四组消融 | 四柱图 + 四行表 |
| Fig. 8 | 实验6：Window sensitivity | latency 与 interference 折线图 |
| Fig. 9 | 实验6：Chunk × Window | 二维热力图，需先修改架构 |
| Fig. 10 | 实验7：Routing imbalance | imbalance-speedup 散点图 |
| Fig. 11 | 实验7：Top-k/Tokens/Experts | 三个小型 sensitivity 子图 |

正文建议控制在 11 张图以内，原始数据表、逐层完整结果和多个 routing seed 放入附录。

---

## 4. 架构修改 TODO

### 必须修改

1. **增加 Chunk 粒度参数。**
   - 新增 `ChunkSizeBytes` 或 `TilesPerChunk`。
   - 修改 Chunk 构造逻辑，使相同权重 trace 可以按不同粒度切分。
   - 在 `EP_MOE_CHUNKS.csv` 和 `EP_MOE_CONFIG.csv` 中记录实际 Chunk 大小。

### 如果实验3必须证明“预取数据位置冲突”，则必须修改

2. **增加预取数据驻留和物理 Bank 放置模型。**
   - 记录每个 prefetched chunk 被放入哪些物理 Bank。
   - 记录 chunk 的 ready、consume、evict 周期。
   - 模拟预取数据占用容量以及覆盖/替换。
   - 区分“请求排队冲突”和“数据放置/容量冲突”。

3. **增加 Naive 与 Bank-aware 两种预取策略。**
   - `PrefetchPolicy = naive / bank_aware`。
   - Naive：只按固定窗口发起预取。
   - Bank-aware：根据目标 Bank 当前压力、可用容量或队列情况决定发起、延迟或重映射。
   - 输出每个预取请求的 issue/suppress/remap 原因。

如果实验3只需要证明“普通预取请求会增加 Bank 争用”，则第 2、3 项不是实验3的前置条件，当前架构已经可以完成。

### 建议修改，但不阻塞主要实验

4. **导出逐物理 Bank 利用率。**
   - 当前内存模型内部已有 `per_bank_access_count` 和 `per_bank_cycle_utilization`。
   - 建议展开为 `BANK_UTILIZATION_REPORT.csv`。
   - 字段至少包括 LayerID、TensorType、BankID、AccessCount、BusyCycles、Utilization、ConflictCount。
   - 有了该报告，实验2和实验4可以直接展示 hotspot 和 idle，而不再依赖容量利用率代理指标。

5. **准备 4/8/16 expert topology。**
   - 不需要重构模拟器。
   - 需要生成对应的 MoE 层，并增加 topology/config 一致性检查。

---

## 5. 推荐执行顺序

1. 先重跑实验1，确定完整网络中 MoE 确实是主要瓶颈。
2. 使用静态比例扫描完成实验2。
3. 使用现有 prefetch interference 字段完成实验3的“请求争用”版本。
4. 运行 Static/Dynamic 对比完成实验4。
5. 使用现有四组配置和 runner 完成实验5。
6. 先完成 Window sensitivity；增加 ChunkSize 参数后再完成 Chunk sensitivity。
7. 最后生成路由、Top-k、Token 和 Expert 数量的批量配置，完成实验7。
8. DC 使用现有独立脚本完成，不与 SCALE-Sim 数据处理混合。

这套顺序可以先利用当前架构得到实验1–5、实验6的窗口部分以及实验7的大部分结果，只把 Chunk 粒度和严格的数据放置型 Bank-aware prefetch 留作最小架构补充。
