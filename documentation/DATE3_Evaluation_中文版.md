# IV. Evaluation

## A. Methodology

我们在 SCALE-Sim v3 上实现周期级、Bank 冲突感知的事件模拟器，以显式刻画 IA、INT8 Weight、INT8 OA 和 INT32 ACC 请求的排队、服务、冲突及 Weight Chunk 生命周期。默认加速器包含 30 个单端口 Bank，每个 Bank 含 128 个 128-bit 条目，即 2 KB 容量和 16 B/cycle 带宽；片上总容量和聚合带宽分别为 60 KB 和 480 B/cycle。普通写直接覆盖目标 Bank，ACC 更新采用 read-modify-write。所有方案共享相同的阵列、Bank 数量、容量、带宽和工作负载。

我们评估 HMoE、Mixtral、MoDSE 和 Switch Transformer 四种经统一缩放的 MoE 网络，覆盖同构和异构专家。默认配置为 8 个专家、256 个 Token 和 Top-1 路由。映射实验比较固定 5:5:5 分区（Static-555）、经全模型离线搜索后冻结的单一静态分区（Static-Opt）、无预取的 PIVOT 动态映射（Dynamic）以及无 Bank 冲突的不可实现下界（Ideal）。预取实验进一步加入在独立校准轨迹上选定并在测试前冻结参数的 Static-FixedPF 与 Dynamic-FixedPF；测试轨迹不参与参数选择。完整 PIVOT 则在线联合选择 Window、Chunk 和 Bank-group。主要指标为总周期和局部访存停顿。

## B. Dynamic Bank Mapping

图 4(a)隔离预取，仅比较 Bank 映射。相对 Static-555，Dynamic 在 HMoE、Mixtral、MoDSE 和 Switch Transformer 上分别减少 17.26%、17.40%、15.43%和 15.40%的总周期，平均为 16.37%。这表明固定 5:5:5 所有权不能持续匹配不同专家阶段的 IA、Weight、OA 与 ACC 压力。Static-Opt 已通过离线全模型搜索消除了大部分明显失配，因此 Dynamic 相对它的额外收益为 0.65%--0.90%，平均 0.78%；对应的局部访存停顿仍降低 1.67%--2.22%。换言之，动态映射的价值不仅是超过一个较弱静态配置，还在于无需为每个网络重新搜索分区，并能在运行时逼近或略优于最强可部署静态映射。Ideal 仍比 Dynamic 低 14.1%--26.6%，说明 Bank 冲突之外仍存在数据搬运和关键路径开销，也为预取留下进一步空间。

## C. Joint Prefetching and End-to-End Performance

图 5(a)在四层非平稳 MoDSE 轨迹上比较可部署方案。两个固定预取参数对仅由两条独立校准轨迹选出：Static-FixedPF 使用 W=8、C=4，Dynamic-FixedPF 使用 W=2、C=8。测试时，PIVOT 的周期为 172,158，分别比 Static-Opt、Dynamic、Static-FixedPF 和 Dynamic-FixedPF 减少 25.56%、23.63%、15.29%和 11.13%；若以 Static-555 为基线，降幅为 43.42%。固定预取虽可隐藏部分 Weight 搬运，但同一 Window/Chunk 无法同时适配阶段间变化的计算余量、驻留容量和 Bank 热点。

图 5(b)给出一次在线决策轨迹，而不是对 PIVOT 做离线参数扫描。运行期间，Chunk 主要在 4 与 8 tiles 间切换，Window 从 64 逐步收缩到 16、2 或 1，目标 Bank-group 也随阶段在多个组间迁移。该轨迹直接说明 PIVOT 的收益来自联合适配：Window 控制预取提前量，Chunk 控制单次搬运与驻留粒度，Bank-group 则避免预取流与当前计算流集中到相同物理资源。在线保护在候选预取不能改善安全 incumbent 时拒绝或回退该候选。

Exp7 将每个网络扩展为四个完整 MoE Transformer block，共包含 28 个 Attention/Router 非 MoE 算子和 16 个专家阶段。非 MoE 层采用同一周期模型并计入其计算与访存停顿；PIVOT 仅优化 MoE 路径。图 5(c)显示，PIVOT 相对 Static-555 在 HMoE、Mixtral、MoDSE 和 Switch Transformer 上分别获得 1.272x、1.316x、1.190x 和 1.266x 的近似端到端加速，平均为 1.261x，即平均周期减少 20.59%。相对 Dynamic-FixedPF 的端到端额外降幅为 1.22%--3.01%，其小于 MoE 层收益是因为静态执行中 MoE 路径仅占总周期的 16.62%--37.69%。该实验不包含 embedding、normalization、softmax、residual 和 sampling，因此结果应表述为完整 Transformer block 的近似加速，而非整模型推理吞吐率。

## D. Sensitivity and Limitation

Exp6 共包含 96 个“模型--配置”点，改变专家数、Token 数、Top-k、EP 度、路由倾斜和随机种子。相对 Static-555，PIVOT 在 80/96 个点上获益，中位周期降幅为 30.03%。Top-k 和 EP 两组全部获益，Top-k 的降幅为 30.19%--37.67%；路由强度与随机种子分别在 11/12 和 37/40 个点上获益。这些结果说明 PIVOT 对路由复制和大多数输入相关扰动具有适应性。

当前实现同时暴露出明确边界：16 专家、32/128 Token 以及少数高倾斜 HMoE 路由发生显著回退。图 6 显示这些配置中预取请求被放大约 1.94--4.93 倍，HBM queue wait 可放大 3.02--195.28 倍，最终使预取及时率下降并扩大局部停顿。原因是现有保护主要观察片上 Bank 压力和预取质量，尚未把全局 HBM 排队作为硬约束。因此本文不宣称 PIVOT 在所有配置下无回退；Exp6 同时界定了其有效范围，并指出未来控制器需要加入 HBM 队列反馈或拥塞触发的 NoPF 回退。

## E. Hardware Overhead

DC 综合设置与原稿保持不变。PIVOT 的标准单元面积由 47,353.355 增至 53,877.754 μm²，开销为 13.78%；总功耗由 3.818 增至 4.281 mW，开销为 12.14%。增量主要来自后端映射查询、Bank 选择、统一请求控制和 AccPipe。结果表明，PIVOT 能以可控硬件代价获得显著的 MoE 路径和近似端到端收益。
