你是一名熟悉 IEEE DATE、计算机体系结构、MoE 加速器和学术可视化的论文图表设计专家。我正在撰写一篇 DATE 论文，提出的架构名为 PIVOT。PIVOT 通过动态 Bank 映射以及 Window、Chunk size、Bank-group 的在线联合选择，优化 MoE 加速器中的片上存储分配和 Weight 预取。

我会向你上传论文前三章、当前第四章 Evaluation 正文、原始实验 PDF，并在需要时上传 CSV。你的任务包含两部分：首先检查并修订第四章，使其与前三章严格对齐；然后根据修订后的第四章设计 Evaluation 图表。你不能修改、虚构、隐藏或选择性排除数据。在读完全部文件前不要直接给出最终设计；应先检查正文与实验文件中的方案名称、归一化基线和数值是否一致，如有冲突先报告冲突。

## 0. 第四章与前三章的一致性审查及词数约束

在设计图片前，必须完整阅读论文前三章和当前第四章，并执行一次逐项一致性审查。前三章是研究动机、问题定义、架构设计和机制语义的权威来源；第四章只能验证前三章已经提出的设计，不能临时引入前三章从未定义的新机制，也不能改变前三章的研究范围。

至少检查以下内容：

1. **研究问题对齐**：第四章每个实验是否分别验证了前三章提出的访存失配、动态 Bank 映射、统一资源池、虚拟到物理映射以及协同预取问题。
2. **贡献对齐**：前三章列出的每项贡献是否在第四章中有对应证据；第四章是否出现没有对应贡献或设计描述的结论。
3. **机制语义对齐**：PIVOT、Dynamic、Static-555、Static-Opt、Static-FixedPF、Dynamic-FixedPF、Ideal、Window、Chunk、Bank-group、online guard、ACC Buffer、read-modify-write 等术语的含义必须与前三章完全一致。
4. **数据流对齐**：第四章对 IA、Weight、OA、INT32 ACC、Weight Chunk、片上 Bank、HBM 和 MoE/非 MoE 数据流的描述必须符合前三章的架构与数据流定义。
5. **因果归属对齐**：exp4 只能把收益归因于无预取动态 Bank 映射；exp5 可以归因于动态映射和预取协同；exp7 的 PIVOT 已包含协同预取，不能把其完整端到端收益归因于纯动态映射。
6. **实验范围对齐**：默认专家数、Token 数、Top-k、数据精度、Bank 数量、Bank 容量、端口、带宽和模型缩放方式应与前三章及配置表一致。
7. **命名对齐**：PIVOT 是最终架构名称，不得把 PIVOT 和 MemDomain 写成两个独立方案；内部诊断名称不能作为额外公开对照组。
8. **结论强度对齐**：不得把近似 block-level 端到端结果写成完整模型吞吐率；不得把 Ideal 写成可部署方案；不得声称 exp6 全部配置无回退。
9. **图文对齐**：正文引用的 Figure 编号、panel 标号、方案名称、数值和图注必须与最终绘图方案一致。
10. **DC 对齐**：DC 综合数据沿用当前第四章给出的结果，不重新推导或修改；只检查文字表述是否与前三章硬件模块一致。

完成检查后，先输出一份“前三章—第四章对齐问题清单”。每个问题必须指出：前三章的定义、第四章当前写法、冲突原因和建议修订。若发现任何问题，必须据此重新修改第四章；只能修改第四章，不能反向修改前三章来迁就实验结果。

修订后的第四章只输出**中文版正文**，不要输出英文正文。但是，为严格控制最终论文长度，你必须在内部制作一份忠实、完整、学术风格的英文翻译，并使用常规英文空格分词规则实际统计词数。要求如下：

- 英文翻译后的正文必须为 **900--1000 words（含900和1000）**。
- 统计范围包含 Section IV 的引言以及 A--E 各小节正文。
- Figure caption、Table caption、表格内容、标题和参考文献不计入900--1000词。
- 不允许只根据中文字数估算英文词数；必须先形成完整英文对照稿再实际计数。
- 如果英文词数少于900或超过1000，必须继续压缩或补充中文版，并重新翻译、重新计数，直到满足范围。
- 最终不要展示英文正文，只在中文版正文末尾报告：`Verified English translation word count: XXX words`。
- 不能为了满足词数删除实验限制、基线定义、端到端近似范围或exp6失败原因等必要信息。

第四章建议保持以下结构；如果前三章逻辑要求调整标题，可以小幅修改标题，但不能改变实验归属：

- A. Methodology
- B. Dynamic Bank Mapping（exp4）
- C. Joint Prefetching and End-to-End Performance（exp5和exp7）
- D. Sensitivity and Limitation（exp6）
- E. Hardware Overhead（沿用现有DC结果）

## 1. 强制版面约束

- 投稿会议为 IEEE DATE，Evaluation 剩余篇幅约两页。
- **所有最终 Figure 都必须是单栏图，最大宽度约 88--89 mm（3.5 in），禁止跨双栏。**
- 一张单栏 Figure 内允许包含两个左右排列的子图，但左右排列后仍必须保证坐标、标签和数据在最终印刷尺寸下清晰可读。
- 如果两个子图左右排列后不可读，必须改为上下排列、压缩信息或拆分成两个单栏 Figure，不能通过缩小字体勉强容纳。
- 图中文字在最终尺寸下建议为 7--8 pt，panel 标号和主要标注不得小于 7 pt。
- DC 综合结果继续使用现有表格，不设计 DC 图片。
- 避免 3D、渐变、装饰背景、冗余图例和过密柱状图。
- 使用色盲安全配色，并通过颜色、纹理、线型或 marker 的组合保证黑白打印可区分。

## 2. 正文实验结构

- Section B：Dynamic Bank Mapping，只包含 exp4 的无预取映射实验。
- Section C：Joint Prefetching and End-to-End Performance，包含 exp5 的协同预取和 exp7 的近似端到端实验。
- Section D：Sensitivity and Limitation，包含 exp6 的六类敏感性及失败原因。

请根据该结构组织图片，不要把 exp7 的完整 PIVOT 端到端收益归因于纯动态 Bank 映射，因为 exp7 的 PIVOT 已包含协同预取。

## 3. 每个目标图与源 PDF 的对应关系

### 候选 Figure 4：Dynamic Bank Mapping

主要源文件：

1. `exp4_mapping_four_way.pdf`
   - 对应 mapping-only 总周期结果。
   - 包含 Static-555、Static-Opt、Dynamic/PIVOT-Map 和 Ideal。
   - 最终命名统一为 Static-555、Static-Opt、Dynamic、Ideal。
   - Static-555 归一化为 1，纵轴为 normalized cycles，lower is better。

2. `exp4_memory_stall_comparison.pdf`
   - 对应 Static 与 Dynamic 的 local memory stall 对比。
   - 仅当它能在单栏范围内提供不重复的新信息时，才作为 Figure 4(b)。
   - 如果与 normalized total cycles 结论重复，可只在正文报告数值而不放该子图。

Figure 4 要回答：动态 Bank 映射相对固定 5:5:5 是否有效；相对离线 Static-Opt 的增益有多大；与不可实现 Ideal 尚有多少距离。Ideal 必须使用空心柱、斜线或虚线，不能伪装成可部署方案。Dynamic 相对 Static-Opt 仅提升 0.65%--0.90%，不得通过截断纵轴夸大。

### 候选 Figure 5：Joint Prefetching and End-to-End Performance

Figure 5(a) 的源文件：

3. `exp5_public_sensitivity.pdf`
   - 只提取可部署方案比较，不照搬其中两个含义重复的子图。
   - 使用 Static-Opt、Dynamic-NoPF、Static-FixedPF、Dynamic-FixedPF、PIVOT 五组。
   - 对应周期依次为 231,286、225,422、203,228、193,724、172,158。
   - Static-FixedPF 的冻结参数为 W=8、C=4；Dynamic-FixedPF 为 W=2、C=8。
   - 固定参数来自独立校准轨迹，测试轨迹未参与选择。

Figure 5(b) 的源文件：

4. `exp5_online_adaptation.pdf`
   - 只使用 Runtime Chunk、Runtime Window、Runtime Bank placement 三部分。
   - 不使用原图的 quality/accuracy 子图，因为未发出预取时 accuracy 未定义，而不是 0。
   - 新图需要用共享的阶段/决策序号横轴，以三行窄带、离散阶梯线或 event timeline 表达 Chunk、Window、Bank-group 的联合变化。
   - Chunk 主要在 4 和 8 tiles 间变化；Window 在 64、16、2、1 等值间变化；Bank-group 随阶段迁移。

Figure 5(c) 的源文件：

5. `exp7_end_to_end_speedup.pdf`
   - 对应四个完整 MoE Transformer block 的近似端到端加速。
   - 包含 Static-555、Static-Opt、Dynamic-NoPF、Static-FixedPF、Dynamic-FixedPF、PIVOT。
   - PIVOT 在 HMoE、Mixtral、MoDSE、Switch Transformer 上分别为 1.272x、1.316x、1.190x、1.266x，平均 1.261x。
   - 图注必须说明实验包含 28 个 Attention/Router 非 MoE 算子和 16 个专家阶段；不包含 embedding、normalization、softmax、residual、sampling，因此是 approximate block-level end-to-end speedup，不是完整模型吞吐率。

Figure 5 有三个信息单元，但仍必须服从单栏宽度。请比较以下布局并推荐最终方案：

- (a)(b) 左右排列、(c) 在下方占满单栏；
- (a)(c) 左右排列，在线轨迹作为独立的单栏 Figure；
- 三个子图上下排列；
- 删除重复信息后改成两个子图。

你必须根据最终 88--89 mm 宽度判断可读性，而不是按屏幕显示大小判断。

### 候选 Figure 6：Sensitivity and Failure Analysis

Figure 6(a) 的六个源文件：

6. `exp6_expert_count.pdf`：Expert count。
7. `exp6_token_count.pdf`：Token count。
8. `exp6_top_k.pdf`：Top-k。
9. `exp6_expert_parallel.pdf`：Expert Parallelism。
10. `exp6_routing_severity.pdf`：Routing severity。
11. `exp6_routing_seed.pdf`：Routing seed。

这六张 PDF 只是数据参考，禁止原样全部放入正文。应重绘成一张单栏汇总图。纵轴定义为：

`Gain = (Cycles_Static-555 - Cycles_PIVOT) / Cycles_Static-555 * 100%`。

统计值如下：

- Expert count：8/12 获益，中位数 28.14%，范围 -167.30%--37.67%。
- Token count：8/16 获益，中位数 -20.28%，范围 -133.97%--37.67%。
- Top-k：8/8 获益，中位数 32.10%，范围 30.19%--37.67%。
- EP：8/8 获益，中位数 35.08%，范围 23.50%--37.67%。
- Routing severity：11/12 获益，中位数 30.37%，范围 -7.83%--37.11%。
- Routing seed：37/40 获益，中位数 29.53%，范围 -14.83%--36.41%。
- 全部：80/96 获益，总体中位数 30.03%。

必须展示全部负收益点，不能截断负轴或隐藏异常值。请比较 broken axis、局部 inset、对称对数轴、蜂群图加异常值标记等方案，选择在单栏尺寸下最清楚且最不容易误导审稿人的方案。必须画 0% 参考线。

Figure 6(b) 的源文件：

12. `exp6_material_failure_mechanism.pdf`
   - 提供 cycle inflation、request amplification 和 HBM queue-wait amplification。
   - 最终优先保留 request amplification（约 1.94--4.93x）与 HBM queue-wait amplification（约 3.02--195.28x）。
   - HBM queue wait 必须使用对数轴。

13. `exp6_failure_diagnosis.pdf`
   - 提供 Token=32 的性能回退、fallback/quality 以及 local-stall inflation 证据。
   - 用于补充因果解释或标注，不建议整张照搬。

Figure 6(b)应清楚表达：

`aggressive/inaccurate prefetch -> request amplification -> HBM queue buildup -> local-stall inflation -> performance regression`。

失败主要发生在 16 experts、32/128 tokens 及少数 high-skew HMoE 路由。当前保护主要监控片上 Bank 压力和预取质量，未把全局 HBM 排队作为硬约束。

## 4. 可选原始 CSV

如果 PDF 无法支持准确重绘，应要求我上传以下文件，而不是从图片中估读：

- `outputs/DATE3/exp4/mapping_comparison.csv`
- `outputs/DATE3/exp4/system_breakdown.csv`
- `outputs/DATE3/exp5/deployable_selection.csv`
- `outputs/DATE3/exp5/joint_prefetch.csv`
- PIVOT 对应的 `MEASURED_SELECTIONS.csv`
- `outputs/DATE3/exp6/robustness_comparison.csv`
- `outputs/DATE3/exp7/end_to_end_summary.csv`

## 5. 统一视觉编码

请为所有图提出一致的方案顺序、颜色、纹理和缩写。建议顺序为：

`Static-555 -> Static-Opt -> Dynamic-NoPF -> Static-FixedPF -> Dynamic-FixedPF -> PIVOT -> Ideal`。

- PIVOT 应有统一的强调色，但不能依靠高饱和红色制造视觉夸张。
- Static 系列、Dynamic 系列和 PIVOT 应分别属于一致的颜色家族。
- Ideal 使用空心或斜线样式。
- 所有性能纵轴必须明确标注 higher/lower is better。
- 不允许用不同图中不一致的归一化基线。

## 6. 需要你输出的内容

请严格依次输出：

1. 前三章—第四章对齐问题清单：逐项列出冲突位置、原因和修改方法；没有问题也要明确说明检查过哪些项目。
2. 修订后的第四章中文版正文：只输出论文原文，不在正文中混入制图说明；正文末尾报告经过实际翻译验证的英文词数。
3. 输入实验文件一致性检查：逐项检查方案名称、基线、图号和数值冲突。
4. 单栏版面总方案：最终保留几张 Figure，每张的宽高和两页内排布。
5. 每张 Figure 的 ASCII 线框草图。
6. 每个目标图/子图对应的源 PDF 文件名，不能遗漏。
7. 每个子图的图形类型及选择理由。
8. 坐标轴、归一化基线、颜色、纹理、marker、参考线和数值标注。
9. Figure 6 极端负值的无误导处理方式。
10. 可直接用于 IEEE DATE 的英文 caption。
11. Visio 重绘步骤，包括 88--89 mm 画布、字体、最终字号、线宽、子图间距和 PDF/SVG 导出设置。
12. 审稿风险检查，特别关注数据挑选、隐藏回退、Oracle 公平性、端到端近似范围、与前三章不一致及视觉夸大。
13. 如果两页无法容纳全部图，给出按重要性排序的删减方案，但任何最终保留的图都必须为单栏图。
