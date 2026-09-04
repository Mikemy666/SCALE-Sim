# PIVOT Evaluation Engineering Reading

> 论文：*PIVOT: Memory-Flow-Aware Bank Virtualization and Adaptive Chunk Prefetching for MoE Accelerators*  
> 阅读快照：2026-08-01；仓库 commit `0b7dace7aba26a69fdd93d7d8073bdb304b4e3a7`。  
> 可信度标签：**已验证**=代码/配置/当前原始数据可闭环；**部分验证**=有实现或数据但合同/版本不完整；**推断**=仅由实现逻辑推得；**未找到**=仓库内无证据；**需要作者确认**=存在冲突或论文定义不充分。  
> 最重要的使用限制：本报告把 PIVOT architectural mechanism、DATE2 simulator abstraction 和旧 Buckyball/MemDomain RTL 分开；不得把其中一层的证据移作另一层的实现证明。

## 1. Executive Summary

仓库包含两条相关但不同的实验链：旧 EP-MoE/DATE1（继承的 `banked_memory_system.py`、一张详细 GPU 加黑箱 GPU）和当前 DATE2 MemDomain/PIVOT 性能模型（`unified_bank_domain.py`、显式 virtual mapping、streaming Chunk residency、在线压力快照）。论文 IV-B–IV-E 应优先引用 `configs/MoE/DATE2`、`outputs/DATE2/{overall,window_chunk,robustness_factorial}` 及其当前 hash 匹配的 CSV；不要引用 `fig/DATE2/analysis.json` 或 `outputs/DATE2/summary_all.csv` 中的旧数值。

当前可追溯 Overall 数据显示，公开的 Full 行实际是 `MemDomain-Safe`，且 `selected_candidate=Online-Guarded-Full`。对 Static-NoPF，四模型总周期下降 20.60%–27.54%，加速 1.260–1.380×；Raw online 行下降 18.93%–28.02%。但 Full 并非每模型都严格优于最强 conventional：Mixtral 与 Dynamic-NaivePF 相同，Switchtrans 也相同，MoDSE/HMoE 才有额外收益。四模型 `Dynamic-NaivePF` 又与 `Static-NaivePF` 完全相同，导致 `scripts/DATE2/validate_date2_contracts.py` 于 2026-08-01 实测失败；因此不能写“全部 publication contracts passed”。

Window×Chunk 当前 32 点数据完整且 hash 匹配。以每点的 Dynamic-NoPF 为 NoPF 参照，Safe 最优为 W=32、C=1：91,262 cycles、20.16% reduction；无 Safe 回退，最低收益为 0%。窗口或 Chunk 均明显非单调。占用量单位是 byte-cycles，不是百分比。

Factorial sensitivity 有 96 个配置（4 模型×24：Experts 3、Top-k 2、Tokens 4、Routing severity 3、EP 2、Routing seed 10）；Safe 对 Static-NoPF 的 reduction 为 12.09%–47.94%，96/96 无回退。这个“无回退”主要由在线/最终 guard 合同保障，不能单独证明未经保护的 Raw policy 鲁棒；Raw 在 11 个点出现回退，极端值非常大。

仓库没有当前 PIVOT/Buckyball Chisel/RTL、综合 Tcl 或 `.rpt`。唯一硬件数字在本科毕设 PDF：28 nm、2 ns、旧 MemDomain 标准单元面积 +16.06%、逻辑功耗 +12.86%。DATE 论文框架 PDF 明确要求加入 Prefetch Queue、Chunk Metadata 等后重新综合。因此这些数字只能作旧版本参考，不能写成当前 PIVOT overhead。

## 2. Repository Version and Environment

| 项目 | 观测值 | 状态与证据 |
|---|---|---|
| 仓库 | SCALE-Sim（`setup.py` package `scalesim`） | 已验证：`README.md:1`、`setup.py:10-20` |
| Git branch | `MoE_prefetch2` | 已验证：`git branch --show-current` |
| Commit | `0b7dace7aba26a69fdd93d7d8073bdb304b4e3a7`，2026-07-27 22:06:28 +0800 | 已验证：`git log -1`；subject=`Complete DATE2 sensitivity experiments and analysis` |
| 未提交修改（阅读前） | 无 | 已验证：`git status --short` 为空；本报告是本任务唯一新增文件 |
| 主要实验目录 | `configs/MoE/DATE2`、`topologies/MoE/DATE2`、`outputs/DATE2`、`scripts/DATE2`、`fig/DATE2` | 已验证 |
| 系统 Python | 3.12.7 | 已验证：`python3 --version` |
| 项目 venv Python | 3.8.10 | 已验证：`.venv/bin/python --version`；实验命令在文档中多使用该 venv |
| 主要 venv 包 | numpy 1.24.4；pandas 2.0.3；matplotlib 3.7.5；numba 0.58.1；Cython 3.2.4 | 已验证：`.venv/bin/pip list --format=freeze`；`requirements.txt` 未锁版本，复现应以此环境快照为准 |
| SCALE-Sim | package version 3.0.0；README 称 SCALE-Sim v3 | 已验证：`setup.py:12`、`README.md:26-34`；代码中个别 docstring 仍写 v2，属于历史文本冲突 |
| Scala / sbt | Scala 2.13.12；sbt 1.12.4 | 已验证：本机命令；仓库无当前 PIVOT `build.sbt`，不是 RTL 工程版本证据 |
| Verilator | 5.024 (2024-04-05) | 已验证：本机命令；未找到本仓库 DATE2 RTL 仿真日志，不能证明结果由此版本生成 |
| Chisel | 未找到版本 | 未找到：无 PIVOT Chisel 源码或依赖清单 |
| 综合工具 | 旧 PDF 称 Synopsys Design Compiler，版本未给出 | 部分验证：本科毕设 PDF §4.5；仓库无可复核日志/脚本 |
| 结果与 commit | 164 个主 CSV 的 workload hash 与当前 JSON 一致；未记录生成 commit | 部分验证：hash 只覆盖 JSON payload，不覆盖全部 Python 源；`outputs/DATE2/overall/MoDSE.csv` 早于当前 commit |

版本污染检查：`outputs/DATE2/summary_all.csv`（2026-07-22）和 `fig/DATE2/analysis.json`（2026-07-23）含 HMoE Static-NoPF=5,251,816 等旧值；当前 `outputs/DATE2/overall/HMoE.csv` 为 230,257。`fig/DATE2/ANALYSIS.md` 又明确写 Exp3–Exp6 旧数值必须重跑。故本文所有定量表直接读取各 suite 当前矩阵 CSV，未使用旧汇总。当前 JSON/CSV 的 SHA-256 检查结果为 Overall 4/4、Window 32/32、Joint 32/32、Robustness 26/26、Factorial 96/96 匹配。

## 3. Evaluation File Index

| 类型 | 文件路径 | 文件作用 | 对应论文小节/图片 | 状态 |
|---|---|---|---|---|
| 架构合同 | `configs/MoE/DATE2/architecture.json` | 30 Banks、精度、AccPipe、off-chip、静态池划分 | IV-A | 已使用 |
| suite 清单 | `configs/MoE/DATE2/manifest.json` | suite 数量、实验映射、模型摘要 | IV-A–E | 已使用 |
| 运行器 | `run_date2_experiments.py` | 分 suite 执行矩阵、聚合与 detail export | IV-B–E | 已使用 |
| 配置生成 | `scripts/prepare_date2_experiments.py` | 从 topology 构造 JSON、requests、Chunk 与 hash 输入 | IV-A、D、E | 已使用 |
| 核心模型 | `scalesim/memory/{unified_bank_domain,virtual_bank_mapping,streaming_residency,prefetch_policy,memdomain_runner,memdomain_experiment}.py` | Bank service、mapping、residency、policy、七行矩阵 | IV-A–E | 已使用 |
| Overall 配置/数据 | `configs/MoE/DATE2/overall/*.json`；`outputs/DATE2/overall/*.csv` | 四模型七 baseline 原始矩阵 | IV-B、IV-C | 已使用 |
| Detail 数据 | `outputs/DATE2/exp4/<model>/{EXPERT,FFN_STAGE,CHUNK,BANK,REQUEST,...}_REPORT.csv` | expert/stage/chunk/bank/request 细节 | IV-B、C | 已使用；部分目录缺 `DETAILS_META.json`，不影响主矩阵 |
| Window/Chunk | `configs/MoE/DATE2/window_chunk/*.json`；`outputs/DATE2/window_chunk/*.csv` | 8×4 sensitivity | IV-D | 已使用 |
| Joint prefetch | `configs/MoE/DATE2/joint_prefetch/*.json`；`outputs/DATE2/joint_prefetch/*.csv` | 与 Exp5 detail export 对应的 32 点 | IV-D | 可能相关；与 window_chunk 命名并存，写图前确认口径 |
| Factorial | `configs/MoE/DATE2/robustness_factorial/*.json`；对应 topology/outputs | 四模型 96 点 | IV-E | 已使用 |
| Characterization | `outputs/DATE2/exp1/*.csv`、`exp2/*.csv`、`exp3/*.csv` | 动机、静态划分、naive interference | Motivation/IV-A/C | 已使用 |
| 绘图 | `scripts/DATE2/analyze_date2.py`、`create_exp*.py`、`fig/exp*.ipynb` | 二次计算及出图 | 各图 | 已使用；`analyze_date2.py` 的输出现已陈旧 |
| 图片 | `fig/DATE2/*.pdf` | 已生成论文候选图 | 各图 | 过期版本风险；不能脱离源 CSV 引用 |
| 综合 | `面向MoE模型的片上内存总线统一虚拟化与预取优化联合设计.pdf` | 旧 MemDomain 论文及综合表 | IV-F 参考 | 过期版本 |
| DATE 规划 | `DATE_MemDomain_论文框架_中文版.pdf` | PIVOT Evaluation 规划/缺口 | IV-A–F | 已使用；不是实验结果 |
| 当前 RTL/报告 | Chisel/RTL、DC Tcl、area/power/timing `.rpt` | 当前 PIVOT 实现/开销 | IV-F | 未找到对应结果 |

## 4. Simulator–Architecture Relationship

| 论文机制 | 模拟器中的抽象方法 | RTL/Buckyball 中的实现证据 | 模拟器简化 | Evaluation 验证内容 |
|---|---|---|---|---|
| Unified On-Chip Memory Domain | `UnifiedBankDomain` 让 IA/W/OA/ACC/prefetch 共用一个 Bank namespace | 旧 PDF 描述统一路径；当前源码未找到 | 请求/容量统一的软件事件模型 | Bank-level contention 策略 |
| Unified SP/ACC Resource Pool | DATE2 为 30-Bank 统一容量；static 才固定 SP 0–14、ACC 15–29 | 旧 PDF 声称 SRAM/ACC 统一；当前未复核 | 是统一容量/Bank service abstraction，不是物理宏连接证明 | 容量与分配结果 |
| Unified BankRead/BankWrite | `UnifiedMemoryRequest` 统一 read/write/prefetch | 旧 PDF 描述接口 | 未模拟 Frontend/Midend handshake | 共享端口/仲裁 |
| AccPipe Direct Write | `wmode=0` 普通 overwrite | 旧 PDF 描述 AccPipe | 固定 service duration | 访问成本抽象 |
| AccPipe Accumulate | `wmode=1` 限定 ACC write；1 read+1 INT32 add+1 write=3×transfer | 旧 PDF 有 AccPipe | 无 RTL pipeline bubbles | atomic RMW contention |
| Dynamic Bank Allocation | runtime mapping 与 candidate pool；dynamic no-PF 含 static incumbent guard | 当前 RTL未找到 | 搜索/软件决策替代硬件控制时序 | 分配策略效果 |
| Virtual-to-Physical Mapping | `VirtualBankMappingTable` 真正保存 live record、稳定 group、容量与 release | 旧 PDF 声称映射表 | lookup 默认 0 exposed cycles | 生命周期映射结果 |
| Multi-Bank Group Mapping | combinations 搜索满足 `bank_group_size` 的物理组 | 当前 RTL未找到 | 组合搜索可能远强于有限硬件 allocator | group placement 上界/抽象 |
| Bank Pressure Monitoring | queue depth/busy/conflicts；64-cycle local horizon | 当前 RTL未找到 | 从已生成 compute service 读取，不是硬件计数器时序 | pressure signal utility |
| Pressure-Aware Selection | cost 依次考虑 late、occupancy queue、interference、queue、incumbent、capacity | 当前 RTL未找到 | 软件全组合可行组搜索 | policy abstraction |
| Chunk-Level Prefetch | explicit Chunk state、issue/use/release、streaming capacity | DATE 扩展 RTL未找到 | weight transfer event abstraction | timeliness/occupancy/interference |
| Adaptive Window | `_effective_prefetch_window=min(max_window, capacity-derived window)`；Overall 常得到 7 | 未找到 controller | 容量规则，不是多候选在线学习 | rule-based adaptation |
| Adaptive Chunk | 单次运行 Chunk size 固定；Sensitivity 离线 sweep C={1,2,4,8} tiles | 未找到 | **没有在线 Chunk size selection** | 离线 sensitivity |
| Runtime Feedback | 每个 decision 使用局部 `BankSnapshot` | 未找到 | pressure 来自模型内部 report | online snapshot-based placement |
| Multi-GPU / EP | DATE2 `latency + remote_bytes/bandwidth` 标量；旧 EP 路径仅一 GPU detailed | 未找到当前 PIVOT RTL | 无 packet/NoC/all-to-all | 固定通信项敏感性 |

结论：DATE2 确实模拟了一个 mapping-table 数据结构，不只是直接写最终 Bank 编号；但表查询、握手、仲裁流水与硬件面积没有逐周期实现。Dynamic 也不是“first free”：它包含压力/期限/容量排序及 static incumbent guard。Adaptive Window 是预定义容量规则；Adaptive Chunk 仅离线 Sweep。所有七行共享 Bank count/capacity/bandwidth/ports/queue/compute workload，资源公平性由 `memdomain_experiment.py:144-150` 检查；但策略搜索强度不同，且静态仅搜索 cyclic contiguous Weight groups（`P7_RUNNER.md`），相对比较仍需保守。

## 5. Evaluation Methodology

七行 canonical matrix 顺序为 Static-NoPF、Static-NaivePF、Dynamic-NoPF、Dynamic-NaivePF、MemDomain-Raw、MemDomain-Safe、Oracle（`memdomain_experiment.py:16-26`）。前五行原则上是 measured；Safe/Oracle 可复制真实候选，当前 adaptive suite 的 Safe 是带最终 online guard 的 measured 行。Oracle 是同一矩阵真实候选的离线最小值，不是逐阶段无限先知。

周期模型使用加法合同：`Total = Compute + BankStall + WeightLoadStall + PrefetchMissStall + PrefetchInterferenceStall + MappingOverhead + CommunicationStall + OtherStall`（`memdomain_experiment.py:109-123`）。因此 CSV 中 Compute 与各 exposed stall 不重叠；transfer/compute overlap 另作指标，不从 Total 中再次相减。Compute 来源为 SCALE-Sim trace/转换得到的 interval，Bank service 与 streaming event 逐事件推进；DRAM 和通信是带宽/固定启动或固定延迟解析项，不是 DRAM timing 或 NoC 模拟。

2026-08-01 最小验证结果：`.venv/bin/python scripts/DATE2/validate_date2_contracts.py` 在 Overall contract 失败，报告四模型 Dynamic-NaivePF 对 Static-NaivePF improvement 均 0.00%，违反验证器“strictly beat”要求。结构/哈希可读不等于发布合同通过。

## 6. Experimental Setup

### 6.1 Simulator

- 名称：SCALE-Sim v3.0.0 的 MemDomain/DATE2 extension。
- 粒度：systolic compute timing 来自 SCALE-Sim trace；统一 Bank 的 request issue/start/completion、端口、queue、Chunk load/use/release 是 cycle-indexed event model。可称“cycle-level bank-conflict-aware analytical/event model”；不宜笼统称完整 PIVOT “cycle-accurate”，因为控制路径、DRAM、NoC 和远端 GPU 未逐周期实现。
- GEMM：topology CSV 以 M/N/K 描述；四模型各 23 行，其中 7 行 Attention/Router、16 行（8 expert×FF1/FF2）MoE。
- Bank conflict：请求在 issue 时若所需 Bank port 未 ready 则 wait；`wait>0` 为该 Bank conflict，rate=`sum(per_bank_conflicts)/total_beats`（`unified_bank_domain.py:207-266`、`memdomain_runner.py:185-201`）。
- 容量：virtual object allocation 按 byte 占用、生命周期 release；30×2,048 B=61,440 B。
- 端口/queue：每 Bank 1 port；outstanding 达 32 时推进 arrival 至最早完成。
- DRAM：配置给 128 bit/cycle 与 20-cycle burst startup；不是 DDR command/timing model。仓库另有 Ramulator patch，但 DATE2 主矩阵未使用它。
- Prefetch：显式 Chunk streaming，compute/prefetch 共用 Bank service；timely/late 按 completion 与 use deadline。
- EP：DATE2 仅标量通信；旧 `simulator.py` 路径是一张 detailed GPU，其余 analytical black box，两者不可混写。

### 6.2 Accelerator Configuration

| 参数 | 默认值 | Sweep 值 | 单位 | 配置文件/变量 | 实际生效位置 |
|---|---:|---|---|---|---|
| Array rows/cols | 16/16 | 未找到 DATE2 sweep | PE | `architecture.json:tiling m,n` | workload conversion/compute trace；名称上是 tile，不应擅自写阵列频率 |
| Dataflow | 未找到统一 DATE2 字段 | — | — | — | 需要作者确认 |
| Clock | 未找到 simulator frequency | 旧 RTL 2 ns | ns | 本科 PDF §4.5 | 仅旧综合 |
| IA/W/OA/ACC | 8/8/8/32 | ACC sensitivity | bit | `architecture.json:precision` | request bytes/RMW |
| Bank 数 | 30 | Exp2 静态分区为 15 SP Banks 内的 253 个正整数 IA:W:OA 组合 | Banks | `physical_banks.count` | `ResourceBudget` |
| Bank entries×width | 128×128 | — | entries×bit | architecture JSON | capacity |
| 单 Bank/总 SRAM | 2,048 / 61,440 | — | byte | architecture/每个 suite JSON | mapping capacity |
| 单 Bank/总带宽 | 16 / 480 | — | byte/cycle | 总带宽由 JSON 480 除以 30 | `UnifiedBankDomain.per_bank_bandwidth` |
| Bank ports | 1 | 未找到当前 sweep 结果 | port/Bank | suite JSON | service model |
| Request buffer | 32 | 未找到当前 sweep 结果 | requests/Bank | suite JSON | outstanding throttle |
| Interleave | 16 | P10 旧链为 1,024 | byte | current suite JSON | `_beats()` |
| Mapping table entries | 无固定表项；随 live objects | — | records | `VirtualBankMappingTable.records` | 软件 dict abstraction |
| Mapping allocate/free latency | 0/0 | 未找到 | cycle/object | `mapping_overhead_per_object` | current Total 为 0 |
| Bank conflict penalty | 无单一常数 | — | cycle | port queue wait | 请求实际排队 |
| AccPipe | read 1 + add 1 + write 1 | — | cycle/transfer | architecture JSON | wmode=1 3×duration |
| Requant | 18 | — | cycle/16×16 tile | architecture JSON | compute request generation |
| Off-chip BW/startup | 128 / 20 | BW 64,128,256,512；latency 10,20,40,80（仅代码支持/规划） | bit/cycle；cycle | architecture JSON | 当前主矩阵字段 |
| DMA/prefetch queue depth | 未单列；复用 request buffer=32 | — | entries | 未找到独立字段 | 需要作者确认 |
| Chunk metadata 数 | 随 Chunk 数；无硬件上限 | — | entries | JSON `chunks` | 软件列表 |
| 1 tile / Chunk | C=1 时 2,048 | C={1,2,4,8} | byte；tiles | topology provenance/window configs | Chunk size（当前 MoDSE） |
| Window | Overall base=2，adaptive max=32 | {0,1,2,4,8,16,32,64} | Chunks | policy/window configs | decision issue point |
| Static IA:W:OA:ACC | 5:5:5:15；ACC 15 中 stripe 可用 12、碎片 3 | Exp2 IA+W+OA=15 的 253 组 | Banks | architecture JSON | static groups |
| GPU / EP | 1 | {1,2} | GPUs | robustness JSON | scalar comm |
| Top-k | 1 | {1,2} | experts/token | provenance | routed M/counts |

未找到并不可补齐：Clock frequency（simulator）、BankChannel 数、硬件 Mapping Table 固定项数、独立 DMA/Prefetch queue、当前 RTL pipeline latency。

### 6.3 Workloads

| Workload | 类型 | Tokens | Hidden | Experts | Top-k | Expert Shape（示例/规律） | Routing Source | GPU/EP | 数据文件 |
|---|---|---:|---:|---:|---:|---|---|---|---|
| HMoE | controlled heterogeneous proxy | 256 | 96 | 8 | 1 | E0 FF1 32×432×96；FF2 32×96×224；各专家不同 | topology counts `[32,48,50,24,34,28,21,19]` | 1；EP sweep 1/2 | `topologies/MoE/DATE2/models/HMoE.csv` |
| Mixtral | controlled homogeneous proxy | 256 | 96 | 8 | 1 | FF1 M×672×96；FF2 M×96×336 | 同上 | 1；EP sweep | `.../Mixtral.csv` |
| MoDSE | controlled heterogeneous proxy | 256 | 96 | 8 | 1 | E0 432，E1 48，E2 384，E3 96 等 | 同上 | 1；EP sweep | `.../MoDSE.csv` |
| Switchtrans | controlled homogeneous proxy | 256 | 96 | 8 | 1 | FF1 M×384×96；FF2 M×96×384 | 同上 | 1；EP sweep | `.../Switchtrans.csv` |

四者均为缩放/控制 workload：`paper_scale_performance_claim=false`，原始模型格式 FP32，但 simulator 为 INT8×INT8/INT32/INT8。每个 CSV 只有一个 block（7 non-MoE + 16 expert GEMM），不能据此声称完整公开模型全部层已仿真。Attention/Router 行被包含在 topology 与 compute cycles；专家为 Topology-count routing。Factorial 另测 Experts={4,8,16}、Tokens={32,128,256,512}、Top-k={1,2}、balanced/light/high、1/2 GPU。代码支持 seeded routing；有结果。

### 6.4 Baselines

| Baseline | Bank Organization | Allocation | Prefetch | Window/Chunk | Destination | Runtime Feedback | 配置/代码 |
|---|---|---|---|---|---|---|---|
| Static-NoPF | 固定 IA/W/OA/ACC groups | exhaustive cyclic contiguous static Weight group 中最低周期 | demand | W=0 | fixed group | 无 | runner `run_best_static_baseline*` |
| Static-NaivePF | 同静态组织 | 同类静态搜索 | fixed lookahead | suite W/C | fixed Weight banks | 无 | `NaivePrefetchPolicy` |
| Dynamic-NoPF | unified pool | dynamic placement，且静态 incumbent 在可行集 | demand | W=0 | mapping table | pressure/capacity | runner |
| Dynamic-NaivePF | unified/dynamic placement | 与 Static-NaivePF 完全相同 issue workload | fixed lookahead | suite W/C | deadline/pressure ranked pool；可 guard 到 static | 局部 pressure 仅用于 placement | runner:250-330 |
| MemDomain-Raw | unified | pressure/capacity-aware | Bank-aware | adaptive capacity-derived W；C 固定 | feasible group cost search | 64-cycle snapshot | runner/prefetch policy |
| MemDomain-Safe (Full) | 同 Raw | online guarded Full；必要时选择 implementable incumbent | 同 Raw/guard | 同上 | 同上 | 是；并含 final guard | runner:828-904 |
| Oracle | 不新增资源 | 七行中非 Oracle 真实候选的离线最小 | 候选继承 | 候选继承 | 候选继承 | 否，offline | experiment:153-177 |

“best conventional baseline”应按每 workload 在前四行取最小，而非固定名称；当前四模型均为 Static-NaivePF=Dynamic-NaivePF 并列。Full 不在线选择 Chunk size。公平性方面硬件预算相同，但 Static 的搜索空间是 cyclic contiguous group，Dynamic 是组合/guard 搜索，论文必须披露。

### 6.5 Metrics

| 指标 | 数学定义 | 分子 | 分母 | 粒度/CSV | 代码 |
|---|---|---|---|---|---|
| Total Cycles | 八个非负 component 之和 | — | — | workload row/`total_cycles` | `memdomain_experiment.py:109-123` |
| Memory Stall | `Total-Compute`（本报告）；各 component 可单列 | stall components | — | row | schema |
| Stall Ratio | `(Total-Compute)/Total` | exposed stall | Total | 二次计算；Exp1 有 `stall_ratio` | Exp1 generator |
| Normalized cycles | `T_scheme/T_Static-NoPF` | scheme | Static-NoPF | plot script | `analyze_date2.py:overall` |
| Reduction | `(T_base-T_scheme)/T_base`；正值为改善 | cycle saving | base | 本报告二次计算 | 明示公式 |
| Speedup | `T_base/T_scheme` | base | scheme | summary/plot | analysis script |
| Conflict rate | conflict_count/total_beats | wait>0 Bank runs | logical beats | row | runner:191-195 |
| Bank imbalance | stddev(per-bank busy)/mean busy (CV) | σ | μ | workload | runner:185-196 |
| Hotspot ratio | Banks with busy >1.5×mean / all Banks | hotspot Banks | 30 | row | runner:196-199 |
| Idle ratio | Banks with accesses=0 / all Banks | idle Banks | 30 | row | runner:199 |
| Effective parallelism | `min(B, sum(busy)/finish)` | busy cycles | finish | row | runner:200 |
| Coverage | prefetched chunks / all residency chunks（runner 当前实现） | prefetches | chunks | row | runner:618 |
| Accuracy | prefetches/prefetches，当前只要发出即 1 | prefetches | prefetches | row | runner:619；指标区分力有限 |
| Timely/Late | timely 或 late prefetch / all prefetches | classified count | prefetches | row | runner:585-622 |
| Unused | 当前 runner 固定 0 | 0 | prefetches | row | runner:622；不能据此证明真实无浪费 |
| Occupancy | Σ Chunk bytes×(release-load cycle) | byte-cycles | 无 | row | runner:623-626；单位 byte-cycle |
| Compute-transfer overlap | transfer intervals 与 compute intervals 并集交集长度 | overlap cycles | 无 | row | `_intersection_cycles` |
| Mapping count/failure | mapping table counters | count | 无 | row | virtual mapping stats |

未实际存在或未形成当前主 CSV 字段：Throughput、array utilization、IA/W/OA/ACC traffic breakdown（detail reports可重整但非 canonical row）、prefetch-induced-conflict 独立字段、DRAM traffic/stall 独立字段、local-vs-inter-GPU stall 完整拆分。多模型“平均”在不同脚本可能是 arithmetic mean；未找到 geometric mean 实现。论文应明确选择，不能混称 average speedup。

## 7. Overall Performance and Bank-Level Behavior

图片链：`run_date2_experiments.py --exp exp4` → `configs/MoE/DATE2/overall/*.json` → `outputs/DATE2/overall/*.csv` → `scripts/DATE2/analyze_date2.py:overall()` / exp4 notebook → `fig/DATE2/exp4_*.pdf`。当前图可能早于 CSV，需重画。

| Workload | Baseline | Total Cycles | Stall Cycles | Normalized | Reduction vs Static-NoPF | Speedup |
|---|---|---:|---:|---:|---:|---:|
| HMoE | Static-NoPF | 230257 | 84719 | 1.0000 | 0.00% | 1.000× |
| HMoE | Static-NaivePF | 174577 | 29039 | 0.7582 | 24.18% | 1.319× |
| HMoE | Dynamic-NoPF | 206973 | 61435 | 0.8989 | 10.11% | 1.112× |
| HMoE | Dynamic-NaivePF | 174577 | 29039 | 0.7582 | 24.18% | 1.319× |
| HMoE | MemDomain-Raw | 165732 | 20194 | 0.7198 | 28.02% | 1.389× |
| HMoE | MemDomain-Safe/Full | 166854 | 21316 | 0.7246 | 27.54% | 1.380× |
| HMoE | Oracle | 165732 | 20194 | 0.7198 | 28.02% | 1.389× |
| Mixtral | Static-NoPF | 208257 | 69697 | 1.0000 | 0.00% | 1.000× |
| Mixtral | Static-NaivePF | 159873 | 21313 | 0.7677 | 23.23% | 1.303× |
| Mixtral | Dynamic-NoPF | 190861 | 52301 | 0.9165 | 8.35% | 1.091× |
| Mixtral | Dynamic-NaivePF | 159873 | 21313 | 0.7677 | 23.23% | 1.303× |
| Mixtral | MemDomain-Raw | 159559 | 20999 | 0.7662 | 23.38% | 1.305× |
| Mixtral | MemDomain-Safe/Full | 159873 | 21313 | 0.7677 | 23.23% | 1.303× |
| Mixtral | Oracle | 159559 | 20999 | 0.7662 | 23.38% | 1.305× |
| MoDSE | Static-NoPF | 121758 | 33166 | 1.0000 | 0.00% | 1.000× |
| MoDSE | Static-NaivePF | 98718 | 10126 | 0.8108 | 18.92% | 1.233× |
| MoDSE | Dynamic-NoPF | 114302 | 25710 | 0.9388 | 6.12% | 1.065× |
| MoDSE | Dynamic-NaivePF | 98718 | 10126 | 0.8108 | 18.92% | 1.233× |
| MoDSE | MemDomain-Raw | 95552 | 6960 | 0.7848 | 21.52% | 1.274× |
| MoDSE | MemDomain-Safe/Full | 96674 | 8082 | 0.7940 | 20.60% | 1.259× |
| MoDSE | Oracle | 95552 | 6960 | 0.7848 | 21.52% | 1.274× |
| Switchtrans | Static-NoPF | 168372 | 52852 | 1.0000 | 0.00% | 1.000× |
| Switchtrans | Static-NaivePF | 131508 | 15988 | 0.7811 | 21.89% | 1.280× |
| Switchtrans | Dynamic-NoPF | 156084 | 40564 | 0.9270 | 7.30% | 1.079× |
| Switchtrans | Dynamic-NaivePF | 131508 | 15988 | 0.7811 | 21.89% | 1.280× |
| Switchtrans | MemDomain-Raw | 136500 | 20980 | 0.8107 | 18.93% | 1.233× |
| Switchtrans | MemDomain-Safe/Full | 131508 | 15988 | 0.7811 | 21.89% | 1.280× |
| Switchtrans | Oracle | 131508 | 15988 | 0.7811 | 21.89% | 1.280× |

Full 对 strongest conventional 的额外 cycle reduction：HMoE 4.42%，MoDSE 2.07%，Mixtral/Switchtrans 0%；算术平均 1.62%。Full 最大/最小对 Static-NoPF reduction 为 27.54%/20.60%，4/4 无回退。不能写“32/32 Overall”；Overall 只有 4 workload。Bank behavior 的绝对字段可由同一 CSV 读取，但 conflict rate 对当前行通常约 4%–5%，与旧 `analysis.json` 的约 99% 完全不同；必须随当前 CSV 重画。论文若声称 Full 一贯降低 conflict/hotspot，需逐行重整 `BANK_REPORT.csv`，不能仅凭 Total 推断。

## 8. Ablation Study

当前 canonical 行不是严格单因素消融链，尤其 Static→Dynamic 同时改变 resource organization/placement；Raw→Safe 加 guard，且 Safe 可能采用不同执行。不存在独立“Unified Only”测量。

| Ablation | 开启机制 | 关闭机制 | MoDSE Total | Stall | Bank Conflict | Prefetch metric |
|---|---|---|---:|---:|---:|---:|
| Static-NoPF | static domain | dynamic/prefetch | 121758 | 33166 | 12840 | 0 requests |
| Dynamic-NoPF | unified dynamic mapping | prefetch | 114302 | 25710 | 12744 | 0 requests |
| Static-NaivePF | fixed prefetch | dynamic mapping/adaptive | 98718 | 10126 | 12840 | timely=1.000 |
| Dynamic-NaivePF | dynamic placement + fixed prefetch | adaptive policy | 98718 | 10126 | 12840 | timely=1.000 |
| MemDomain-Raw | pressure-aware placement + adaptive-window rule | final guard | 95552 | 6960 | 12744 | timely=0.9681；late=0.0319 |
| MemDomain-Safe | Raw + online/final guard | — | 96674 | 8082 | 12744 | timely=0.9574；late=0.0426 |
| Oracle | offline best real row | runtime implementability | 95552 | 6960 | 12744 | 选择 Raw |

在 MoDSE 上 Dynamic NoPF 独立于 Static NoPF 的 reduction 为 6.12%；NaivePF 相对 Static-NoPF 为 18.92%；Raw 相对 Dynamic-NaivePF 为 3.21%。但 Dynamic-NaivePF 相对 Static-NaivePF 为 0，合同验证失败。没有独立 Bank-Aware Placement-only、Adaptive Window-only、Adaptive Chunk-online 或 Unified Path-only 行；不得人为补成完整链。旧 EP 四行小实验确有 dynamic prefetch 比 dynamic no-PF 慢 13 cycles（`documentation/P16_FINAL_EXPERIMENT_AUDIT.md`），但它属于旧模型，不能作为当前 DATE2 ablation 数字。

## 9. Window and Chunk Sensitivity

数据链：`configs/.../window_chunk/w{0,1,2,4,8,16,32,64}_c{1,2,4,8}.json` → `outputs/.../window_chunk/*.csv`；heatmap 脚本当前名为 `analyze_date2.py:window_chunk()`，但它导出旧命名 `exp5_window_chunk_heatmap.pdf`，现有 PDF 应重画。Workload=MoDSE；30 Banks；C 的单位是 tiles，C=1 为 2,048 B。

| W | C | Safe Total | Stall | Change vs Dynamic-NoPF | Late | Timely | Occupancy (byte-cycle) | Mapping |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1/2/4/8 | 114302 | 25710 | 0.00% | 0 | 0 | 0 | 1902/1182/826/650 |
| 1 | 1/2/4/8 | 98718 | 10126 | 13.63% | 0 | 1 | 22671360/45342720/90212352/177910784 | 1902/1182/826/650 |
| 2 | 1/2/4/8 | 98718 | 10126 | 13.63% | 0 | 1 | 45337856/90665984/180348928/355530752 | 1902/1182/826/650 |
| 4 | 1/2/4/8 | 98718 | 10126 | 13.63% | 0 | 1 | 90656256/181254144/360394752/709935104 | 1902/1182/826/650 |
| 8 | 1/2/4 | 98718 | 10126 | 13.63% | 0 | 1 | 181234688/362196992/719636480 | 1902/1182/826 |
| 8 | 8 | 109522 | 20930 | 4.18% | 0.0638 | 0.9362 | 881166336 | 650 |
| 16 | 1/2 | 98718 | 10126 | 13.63% | 0 | 1 | 362158080/723148800 | 1902/1182 |
| 16 | 4 | 111010 | 22418 | 2.88% | 0.1044 | 0.8956 | 933519360 | 826 |
| 16 | 8 | 114302 | 25710 | 0.00% | 0 | 0 | 0 | 650 |
| 32 | 1 | **91262** | **2670** | **20.16%** | 0 | 1 | 723070976 | 1902 |
| 32 | 2/4/8 | 114302 | 25710 | 0.00% | 0 | 0 | 0 | 1182/826/650 |
| 64 | 1/4/8 | 114302 | 25710 | 0.00% | 0 | 0 | 0 | 1902/826/650 |
| 64 | 2 | 110632 | 22040 | 3.21% | 0.0722 | 0.9278 | 1049628160 | 1182 |

最优 W32/C1；最大 Safe 回退为 0（多个点 guard 到 NoPF），late 最大 10.44%（W16/C4），occupancy 范围 0–1,049,628,160 byte-cycles。W 和 C 均非单调且存在强交互。所谓“稳定收益区”可保守定义为 W=1/2/4、全部 C，以及 W=8/16 的较小 C，均得到 13.63%；但这是当前 proxy 的观测，不应泛化。Safe 的无回退由 guard 造成；Raw heatmap 应另画以展示真实退化边界。

## 10. Routing and Expert Sensitivity

| 图片/子图 | 横轴 | 纵轴 | 固定参数 | Sweep | Workload | 原始数据 |
|---|---|---|---|---|---|---|
| Expert count | 4/8/16 | speedup/reduction | 其余模型参数 | experts | 四模型 | `outputs/DATE2/robustness_factorial/expert_count__*.csv` |
| Top-k | 1/2 | 同上 | — | top-k | 四模型 | `top_k__*.csv` |
| Token count | 32/128/256/512 | 同上 | — | tokens | 四模型 | `token_count__*.csv` |
| Routing severity | balanced/light/high | 同上 | — | skew class | 四模型 | `routing_severity__*.csv` |
| Routing seed | seed 40–44×light/high | 同上 | — | seed | 四模型 | `routing_seed__*.csv` |
| EP | 1/2 GPU | 同上/communication stall | scalar link params | GPU | 四模型 | `expert_parallel__*.csv` |

96 点 Safe reduction vs Static-NoPF=12.09%–47.94%（speedup 1.137–1.921×），96/96 无回退；最小 `token_count__MoDSE__512`，最大 `expert_count__HMoE__16`。对每点 strongest conventional 的额外 reduction 算术平均 1.68%，最大 22.21%（HMoE, 16 experts）。Safe 的 `selected_candidate` 96/96 均标为 Online-Guarded-Full。

Raw 在 11/96 点相对 Static-NoPF 回退；例如小 token 点出现数量级异常（最高约 30×），说明未经 guard 的在线 policy 并不鲁棒，也提示模型/策略边界值得排查。Static/NaivePF 的回退点需要按论文图筛选逐项呈现；不能以 Safe 结果替代。当前有 homogeneous（Mixtral/Switchtrans）与 heterogeneous（HMoE/MoDSE）、Top-1/2、1/2 GPU 数据；没有 intermediate dimension 独立 sweep、Bank count/capacity/bandwidth/queue-depth 当前 factorial 结果。代码/旧 DATE1 可能支持这些参数，但“代码支持，未找到已完成的当前 DATE2 实验结果”。

EP=2 仅增加 `20 + ceil(remote_bytes/128)` 的标量 communication stall；不能声称完整 expert-parallel all-to-all 鲁棒性。

## 11. Hardware Overhead and Synthesis

**当前 PIVOT：未找到 RTL 与综合报告。** `configs/MoE/DATE2/manifest.json` 明示 `simulator_only=true`、`rtl_dc_out_of_scope=true`。仓库只有通用 Verilog code examples 和 Python 侧 Buckyball contract adapter，不是 PIVOT RTL。

| Design | Area | Area overhead | Dynamic Power | Leakage | Total | WNS/TNS | Included Modules |
|---|---:|---:|---:|---:|---:|---|---|
| 旧 static baseline | 91,348.702 µm² logic | — | 61.614 mW | 6.568 µW | 61.621 mW | 未给出 | 旧固定 Bank 路径；PDF |
| 旧 MemDomain | 106,016.400 µm² logic | +16.06% | 69.539 mW | 7.471 µW | 69.548 mW | 未给出，是否 timing closure 未找到 | Unified path、dynamic mapping/scheduling、AccPipe（按 PDF） |
| 当前 PIVOT | 未找到 | 未找到 | 未找到 | 未找到 | 未找到 | 未找到 | Prefetch Queue/Chunk Metadata/pressure/multi-bank 等覆盖情况未知 |

旧结果条件：Synopsys Design Compiler（版本未给）、28 nm `scc28nhkcp_hdc35p140_rvt`、corner `ssg_v0p9_m40c`（0.9 V、−40°C 可由 corner 名读取，但 PDF 未另行解释）、2 ns；无 P&R/RC corner。旧 MemDomain combinational=94,518.522 µm²，sequential=11,497.878 µm²。另估 storage-related area=431,073.167 µm²、power=14.539 mW，含存储后 total=537,089.567 µm²/84.087 mW；主 overhead 表采用不含 SRAM macro 的 logic 口径。Cell count、top module、WNS、TNS 未找到。

固定警告：**旧版 MemDomain 综合结果，仅可作为参考，不能直接作为最终 PIVOT 硬件开销。** DATE 框架 PDF 明确说加入 Prefetch Queue、Chunk Metadata 等后需重新综合。

## 12. Figure-to-Data Mapping

| 论文小节 | 图片/子图 | 运行脚本 | 配置 | 原始数据 | 处理/绘图 | 输出图片 | 状态 |
|---|---|---|---|---|---|---|---|
| Motivation | layer bottleneck/flow/ACC | `--exp exp1` | architecture + MoDSE | `exp1/*.csv` | exp1 notebook/analyzer | `exp1_*.pdf` | 当前数据可用，图需核对时间 |
| Static characterization | best ratio | `--exp exp2` | 15-SP partition | `exp2/static_bank_sweep.csv` | exp2 notebook | `exp2_*.pdf` | 253 positive partitions/stage |
| NaivePF motivation | interference | `--exp exp3` | W×C | `exp3/naive_prefetch_interference.csv` | exp3 notebook | `exp3_*.pdf` | 当前 CSV 可读 |
| IV-B | Overall | `--exp exp4` | overall JSON | `overall/*.csv` | analyzer/exp4 notebook | `exp4_overall_performance.pdf` | 图/summary 陈旧，重画 |
| IV-B | Bank behavior | 同上 | 同上 | `exp4/*/BANK_REPORT.csv` | exp4 notebook | 多个 exp4 PDF | 需从 detail 重整 |
| IV-C | Ablation | 同上 | 同上 | overall rows | exp4 notebook | `exp4_cross_model_ablation.pdf` 等 | 非严格单因素 |
| IV-D | W×C heatmap | `--exp exp5` | joint/window JSON | 32 matrix CSV | analyzer/exp5 notebook | `exp5/exp6_window_chunk*.pdf` | 命名冲突，重画 |
| IV-E | factorial sensitivity | `--exp exp6` | 96 JSON | robustness_factorial CSV | exp6 notebook/export | `exp6_*.pdf` | 当前 CSV 可用 |
| IV-F | overhead table | 无 | 无 | 旧本科 PDF table 4.6–4.9 | 人工排版 | PDF 内表 | 旧版参考 |

图轴：Overall x=model、y=normalized Total cycles、legend=5 public baselines，无 error bar；W×C x=Chunk tiles、y=Window、color=cycles/stall/occupancy；Sensitivity x=sweep value、y=Safe speedup/reduction。所有现有图均无统计误差线；routing seeds 可用于误差/范围但当前脚本主要逐点画。`analyze_date2.py` 会过滤七行到五个 public baseline，并把 `MemDomain-Safe` 重命名 `MemDomain`；这是二次处理。旧 PDF/CSV 存在人工整理痕迹，发表前应由当前矩阵重新生成。

## 13. Simulator Abstractions and Limitations

| 简化项 | 模拟器做法 | 真实架构做法 | 绝对结果影响 | 相对比较影响 | Baseline 一致性 |
|---|---|---|---|---|---|
| Front/Mid/Backend handshake | 不逐级模拟 | RTL valid/ready/pipeline | 低估控制气泡 | 若方案控制复杂度不同会偏向 PIVOT | 否 |
| Mapping lookup | dict lookup；当前 exposed=0 | table lookup/arbitration | 低估延迟 | Dynamic 更受益 | Static 无此项 |
| Allocator search | 软件组合/排序 | 有限并行硬件 | 低估 decision cost | Dynamic/Full 更受益 | 否 |
| AccPipe | 固定 3×transfer atomic lock | pipeline/handshake | 改变绝对 stall | 若同 RMW 流量则部分一致 | 部分 |
| DRAM | bandwidth+startup | controller/timing/refresh | 绝对时间不准 | traffic pattern 不同时可改变排序 | 参数相同，行为不同 |
| NoC/EP | scalar latency+BW | packet/all-to-all/congestion | 低估尾延迟/争用 | EP 方案比较不充分 | 参数相同 |
| Remote GPUs | DATE2 无详细执行；旧 EP analytical blackbox | 多 GPU detailed | 无法捕捉远端 Bank conflict | routing/EP 结论有限 | 不适用 |
| Adaptive Window | 容量分级公式 | feedback controller | 控制开销缺失 | 可能高估 Full | 仅 Full |
| Adaptive Chunk | 离线 sweep；run 内固定 | runtime chunk controller | 未验证真实机制 | 不能支持在线 adaptive chunk claim | 不适用 |
| Prefetch metadata | Python objects，无固定上限 | SRAM/register queues | 无面积/拥塞成本 | Full 更受益 | 否 |
| Safe final guard | 可比较/回退到 incumbent | 需在线可观测判定 | 保证无回退 | 明显改善 robustness | 仅 Safe |
| Static search | cyclic contiguous groups | 固定设计可能一种/多种 | 给 static 较强基线 | 仍比 dynamic 组合空间窄 | 声明空间后公平 |

建议的保守表述： “We extend SCALE-Sim with a cycle-indexed, bank-conflict-aware event and analytical model.”；“The model abstracts control-path timing while preserving a common Bank capacity, port, queue, and bandwidth budget.”；“Window adaptation follows a capacity-based runtime rule, whereas Chunk-size sensitivity is evaluated by offline sweeps.”；“All candidates share the same physical resource budget, although their placement search spaces differ.”

不建议：“cycle-accurate implementation of the complete PIVOT RTL”；“online adaptive Window and Chunk selection”；“full All-to-All simulation”；“zero-cost mapping in hardware”。

## 14. Confirmed Conclusions

1. 论文候选结论：当前 Overall Full 对 Static-NoPF 降低 20.60%–27.54%。状态：已验证。数据：四个 hash-matched overall CSV。代码：明确 normalization 公式。限制：四个缩放 proxy；Safe/guarded。
2. 候选结论：Full Overall 4/4 无回退。状态：已验证。限制：guard 合同，不能归因于 Raw policy 本身。
3. 候选结论：W32/C1 在当前 MoDSE 32 点中最优，91,262 cycles、比 Dynamic-NoPF 低 20.16%。状态：已验证。限制：单一 workload/30 Banks。
4. 候选结论：Window/Chunk 非单调，late 最大 10.44%，occupancy 0–1,049,628,160 byte-cycles。状态：已验证。限制：Safe 数据；零值可能是 fallback。
5. 候选结论：Factorial Safe 在 96/96 点无回退，reduction 12.09%–47.94%。状态：已验证。限制：guarded Safe 与解析 EP。
6. 候选结论：总周期严格等于八个 additive components。状态：已验证。代码：schema constructor 强制。
7. 候选结论：所有矩阵行具有相同硬件预算且当前 164 个主 CSV hash 对应配置。状态：已验证。限制：未记录完整源 commit。

## 15. Conclusions Requiring Cautious Wording

1. 论文候选结论：PIVOT 使用 runtime feedback。状态：部分验证。数据/代码：64-cycle local BankSnapshot；限制：由已建模 compute service 得到，硬件 counter/control 未验证。推荐：“snapshot-based online placement abstraction”。不推荐：“implemented feedback controller”。
2. 候选结论：adaptive prefetch。状态：部分验证。Window 是容量规则；Chunk 是 offline sweep。推荐分别陈述；不推荐合并为 online Window/Chunk controller。
3. 候选结论：PIVOT 优于 best conventional。状态：部分验证。HMoE/MoDSE 严格优于，Mixtral/Switchtrans 持平。推荐“matches or improves”；不推荐“consistently outperforms”。
4. 候选结论：routing/EP robustness。状态：部分验证。96 点存在，但 EP 是固定 latency+BW。推荐“under the modeled EP communication cost”；不推荐“under realistic all-to-all congestion”。
5. 候选结论：Bank virtualization 降低 conflict。状态：部分验证。某些当前 rows conflict count 很接近，需用 detail report逐图支持；不要只用旧约 99% conflict-rate summary。
6. 候选结论：cycle-level simulation。状态：部分验证。推荐“cycle-indexed Bank/event model based on SCALE-Sim compute traces”；不推荐完整架构 cycle-accurate。

## 16. Unsupported or Missing Claims

1. 当前 PIVOT 面积/功耗为 +16.06%/+12.86%：未找到。数字仅对应旧 MemDomain。
2. 当前 PIVOT 满足 2 ns timing：未找到 WNS/TNS 或 timing report。
3. 在线 adaptive Chunk selection：未找到；只有离线 C sweep。
4. 完整 Frontend/Midend/Backend、Mapping lookup、NoC、DRAM timing 的逐周期模拟：未找到。
5. 完整公开 Mixtral/Switch/HMoE/MoDSE 全模型性能：不支持；当前是 23-row controlled/scaled proxy。
6. 全部 DATE2 validation passed：不支持；当前 contract validator 明确失败。
7. Dynamic-NaivePF 严格优于 matched Static-NaivePF：不支持；四 Overall 全部持平。
8. 独立 Unified Only、Bank-aware-only、Adaptive Chunk-only ablation：未找到。
9. 当前 Bank count/capacity/bandwidth/ports/queue sensitivity 结果：未找到。

## 17. Missing Information

- 当前 PIVOT RTL repository/commit、top module、Chisel version、test logs。
- 当前 synthesis Tcl、tool version、library files、area/power/timing reports、SRAM macro口径。
- Simulator clock/dataflow 与 DATE2 array 16×16 的最终论文命名。
- 独立 DMA queue、Prefetch queue、BankChannel、Mapping Table/Chunk Metadata entries。
- “Full PIVOT”究竟发表 Raw 还是 Safe；当前 plot 用 Safe。
- Safe online final guard 在硬件中如何获知 incumbent end-to-end outcome。
- 为什么当前合同要求 Dynamic-NaivePF strict improvement，而 implementation/data允许 equality。
- 为什么 `fig/DATE2/ANALYSIS.md` 称 Exp3–6 stale，但其后又存在 hash-matched CSV；需要作者确认哪一批已正式审核。
- 旧 summary/fig 是否应删除或仅重生成；当前存在强烈误引用风险。
- Workload 的公开来源与缩放映射需要从 catalog/论文引用手工核实，当前 controlled CSV 本身不能证明模型忠实度。

## 18. Minimal Additional Experiments

**必须**

1. 固定 validator 合同或算法预期后重跑/重新验证 Exp4–Exp6，并生成带 commit/source hashes 的 immutable manifest；不要为了“严格胜出”篡改模型。
2. 从当前 CSV 重画 IV-B–IV-E，清除/隔离旧 `summary_all.csv`、`analysis.json` 与旧 PDF 的引用。
3. 对当前 PIVOT RTL（含 Prefetch Queue、Chunk Metadata、pressure counters、multi-bank group logic）重新综合，保存 area/power/timing 原始报告。

**强烈建议**

4. 增加 Raw 与 Safe 并列 sensitivity，显示 guard 前真实回退。
5. 补 Unified-only、placement-only、adaptive-window-only、chunk-policy-only 的严格单因素 ablation。
6. 补 Bank count/capacity/bandwidth/port/queue sensitivity，至少覆盖论文默认点附近。
7. 用 detail `BANK_REPORT` 重整 conflict/hotspot/idle 图，并逐项验证定义。

**可选**

8. 接入 Ramulator/更完整 DRAM timing 与 packet/collective EP model，作为模型敏感性而非替换主实验。
9. 增加 full-scale（`dimension_divisor=1`）代表点，验证 proxy 趋势。
10. routing seed 以均值+误差线展示；明确算术或几何平均。

仅重新整理即可完成：Overall 当前 absolute/normalized cycles、四模型 ablation（但非完整机制链）、Window×Chunk Safe/Raw heatmap、96点 sensitivity、Bank detail 图。Hardware overhead 不能仅整理旧数据完成。

## 19. Questions for the Author

1. 论文中 “PIVOT/Full” 指 `MemDomain-Raw`、`MemDomain-Safe` 还是 Oracle？Safe guard 是否可硬件在线实现？
2. Dynamic-NaivePF 与 Static-NaivePF 持平是否为合法结果，还是表明 dynamic placement path 未生效？验证器为何要求 strict beat？
3. DATE2 正式数据批次是哪一批？是否批准废弃 `summary_all.csv`、`fig/DATE2/analysis.json` 和现有 exp4–exp6 PDF？
4. 16×16 `tiling` 是否就是论文 systolic array size？Dataflow 与 simulator clock 如何设置？
5. Chunk=1 的 2,048 B 是否对所有模型/Factorial 都成立，还是仅当前 MoDSE config？
6. 论文的 adaptive Chunk controller 在其他 RTL/仓库吗？本仓库只有离线 sweep。
7. 当前 PIVOT RTL、Buckyball commit、综合 top 与报告在哪里？
8. 旧综合 corner 名中的 0.9 V/−40°C 是否可直接作为正式 voltage/temperature，还是需由 liberty 报告确认？
9. “best conventional”应按 workload 动态取前四行最小，还是预先指定 Dynamic-NaivePF？
10. 是否接受将 simulator 表述为 bank-conflict-aware cycle-indexed analytical/event model，而不称完整 cycle-accurate PIVOT？

---

自检：19 个固定章节均非空；关键参数均带单位/来源；Overall 与 Window/Chunk 给出原始计算数；所有图均可追溯到配置/CSV/脚本；Simulator/RTL/旧综合已分离；offline Chunk sweep 未写成 online；不确定项均已标注。本报告未修改任何工程功能代码。
