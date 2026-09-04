# PIVOT Project Context Recovery

审计日期：2026-09-04（Asia/Shanghai）  
审计原则：以当前工作树、Git 对象、原始配置和 CSV 为准；未运行耗时实验，未修改生产代码、benchmark 或已有结果。  
置信度标记：**高**＝代码/配置/哈希或原始 CSV 直接证明；**中**＝实现直接证明但结果需要重验；**低**＝仅有估算、历史说明或不完整 provenance。

## 1. Repository Snapshot

- Repo root：`/home/wbh326/code/SCALE-Sim`。【高；`git rev-parse --show-toplevel`】
- 当前分支：`MoE_prefetch2`；HEAD：`0b7dace7aba26a69fdd93d7d8073bdb304b4e3a7`，与 `origin/MoE_prefetch2`、`origin/MoE_prefetch1` 和本地 `MoE_prefetch1` 同一提交。【高；Git refs】
- 最可信的**已提交** PIVOT/MemDomain 分支是 `MoE_prefetch2`：它包含 DATE2 完整提交历史；DATE3 是该分支上的未提交工作树，不是独立 Git 分支。【高；`git log --decorate`、`git status --short`】
- 相关本地分支：`MoE_prefetch`、`MoE_prefetch1`、`MoE_prefetch2`、`clean`、`clean1`、`clean1_prefetch`、`main`。远端还存在 SCALE-Sim 通用开发分支，详见 `git branch -a`。【高】
- 相关最近提交（新到旧）：`0b7dace Complete DATE2 sensitivity experiments and analysis`、`8ec536a Refine DATE2 mapping and prefetch experiments`、`c52e85b Complete DATE2 MemDomain architecture and experiment suite`、`a336e51 Guard fixed-window DATE2 sweeps with incumbent`、`4cc6a84 Replace Safe oracle selection with online guard`、`19215f0 Model hidden and exposed mapping latency`、`46d6653 Jointly optimize prefetch timing and bank placement`、`3248c50 Add lifetime-aware virtual bank placement`、`a181f20 Preserve prefetch work with timed capacity planning`、`abf2ea7 Fix MemDomain timing and online pressure`。【高；Git log】
- 工作树非干净：13 个 tracked 文件已修改，DATE3 的 runner、policy、EP、配置、脚本、测试、文档和图大多为 untracked；`outputs/` 被 `.gitignore:33` 忽略。【高；`git status --short`、`git diff --stat`、`.gitignore`】
- 当前生产 Python 文件可被 AST 解析（27 个 `scalesim/memory/*.py` 加入口文件），但 `.venv` 和 system Python 均缺 `numpy`/`pandas`，无法导入 simulator 或执行验证脚本。【高；只读 AST 检查及 import 检查；依赖声明见 `requirements.txt`、`setup.py`】

## 2. Project Goal

PIVOT 是面向 MoE 推理的统一 Bank 域和自适应 Weight 预取机制：在 routing 已知后，按真实 Expert/FFN 工作动态调整虚拟到物理 Bank 的映射，并联合选择 Chunk、Window 和 Bank group；目标是降低静态 IA/Weight/OA/ACC ownership 的碎片与冲突，同时控制 Bank 容量、单端口排队和共享 HBM 拥塞。【高；`configs/MoE/DATE3/architecture.json`，`scalesim/memory/pivot_ca_prefetch.py::CoverageAccuracyConstrainedPrefetchPolicy.choose`，`scalesim/memory/pivot_ca_runner.py::run_pivot_ca`】

DATE2 内部名称是 MemDomain；DATE3 的唯一论文公开名称是 PIVOT。`MemDomain-Raw`、`MemDomain-Safe` 和 `Oracle` 仅保留在内部诊断 matrix，不应当作为额外 proposed schemes 暴露。【高；`scripts/DATE3/experiment_contract.py::INTERNAL_TO_PUBLIC/ANALYSIS_ONLY`，`configs/MoE/DATE3/manifest.json`】

## 3. Architecture

当前硬件合同如下：【高；`scalesim/memory/buckyball_memdomain.py::BuckyballMemoryContract`，`configs/MoE/DATE2/architecture.json`】

| Resource | Current value |
|---|---:|
| Physical Banks | 30 |
| Bank width / entries | 128 bit / 128 |
| Capacity per Bank / total | 2,048 B / 61,440 B |
| Bandwidth per Bank / aggregate | 16 B/cycle / 480 B/cycle |
| Ports per Bank | 1 |
| Request queue depth | 32（payload） |
| Off-chip/HBM width | 128 bit/cycle = 16 B/cycle |
| HBM startup | 20 cycles/request |
| Array/tile | 16×16 |
| IA / Weight / OA | INT8 / INT8 / INT8 |
| ACC | INT32 |
| ACC stripe | 4 Banks |
| ACC overwrite / RMW | first K tile overwrite；later tile atomic read-add-write, 3× transfer duration |

Static-555 的完整分区实际是 `IA:W:OA:ACC = 5:5:5:15`；15 个 ACC Bank 中只有 12 个满足四 Bank stripe，3 个是固定边界碎片。统一模式允许四类对象从全部 30 Bank 中分配，但 live mapping 稳定且无 live-data migration。【高；`buckyball_memdomain.py::STATIC_ALLOCATION/legal_allocations`，`virtual_bank_mapping.py::VirtualBankMappingTable`】

## 4. Current Simulation Pipeline

PIVOT/DATE3 的真实主路径不是标准 `scalesim.scale_sim` CLI，而是专用 runner：【高】

```text
run_date3_experiments.py::main
  -> memdomain_runner.py::load_runner_config(JSON)
  -> date3_ep_model.py::EPContract.from_payload / localize_detailed_npu
     -> deterministic_routes_from_counts
  -> topology_workload.py::load_moe_topology (仅在生成/编译计划时读 CSV)
  -> topology_workload.py::generate_topology_runner_payload
     -> analytical ceil(M*N*K/256) stage timing
     -> buckyball_compiler.py::compile_gemm_bank_plan
     -> Weight chunks + IA/OA/ACC requests
  -> controls: date3_ep_system.py::run_date3_ep_paper_controls
     -> memdomain_runner.py::run_paper_control_executions
  -> PIVOT: pivot_ca_runner.py::run_pivot_ca_file/run_pivot_ca
     -> pivot_ca_prefetch.py::CoverageAccuracyConstrainedPrefetchPolicy.choose
     -> streaming_residency.py::StreamingResidencyEngine.run
        -> virtual_bank_mapping.py::VirtualBankMappingTable.allocate/resolve/release
        -> unified_bank_domain.py::UnifiedBankSession.submit
        -> serialized shared-HBM scheduler
  -> date3_ep_system.py::build_ep_system_timeline
     -> max(detailed NPU, analytical peers) + combine
  -> summary.csv / comparison.csv / detail CSV / metadata.json
```

重要差异：`configs/MoE/DATE2/architecture.json` 声称 compute timing source 是 “SCALE-Sim trace”，但当前 workload 生成器在 `generate_topology_runner_payload` 第 74–90 行直接使用 `ceil(M*N*K/(16*16))` 的解析 MAC 模型，并没有在 DATE3 主路径中调用标准 SCALE-Sim trace runner。因此准确表述应是 “SCALE-Sim 仓库内扩展的 cycle-indexed analytical/event model”，而非完整 SCALE-Sim trace-driven cycle-accurate 仿真。【高；代码与配置直接冲突】

## 5. MemDomain Implementation

- 请求统一类型为 `UnifiedMemoryRequest`，合法 tensor 为 `ia/weight/oa/accumulator`，kind 为 `read/write/prefetch`；所有请求走 `UnifiedBankDomain`/`UnifiedBankSession.submit`。【高；`unified_bank_domain.py`】
- `_beats` 按 interleave 和 allowed Bank group 分发字节；`_beat_duration` 使用总带宽除 Bank 数得到单 Bank 带宽，ACC `wmode=1` 将端口锁定为普通 transfer 的 3 倍。【高】
- 每 Bank 显式维护 port ready cycle、outstanding completions、queue depth、busy、conflict、wait；满队列时请求等待最早 completion。【高；`UnifiedBankDomain.simulate`、`UnifiedBankSession.submit`】
- `VirtualBankMappingTable` 维护 per-Bank capacity/occupied/peak、MappingRecord、allocation/release/resolve 计数；policy 支持 `round_robin`、`least_occupied`、`least_queue_pressure`、`conflict_aware`。【高；`virtual_bank_mapping.py::PLACEMENT_POLICIES`】
- mapping 在对象生命周期中不迁移；compute vBank 在最后一次 request completion 后释放，Weight chunk 在 consume/eviction/unused-release 时释放。【高；`streaming_residency.py::StreamingResidencyEngine.run`】
- 另有较早的 `buckyball_memdomain.py::PhysicalBankAllocator`，但当前 DATE3 event path 使用 `VirtualBankMappingTable`，不是该早期 allocator。【高；imports/call sites】

## 6. Dynamic Bank Mapping

- 静态路径由 `static_allocation_config` 冻结一组连续 IA/W/OA/ACC pools；Static-555 使用 `BankAllocation(5,5,5,15)`。【高；`memdomain_runner.py`】
- Static-Opt 由 `profiled_static_allocation -> compile_workload_static_plan` 对完整 workload 的所有 Expert GEMM 穷举一个固定四域分配，不读取运行时 prefetch outcome。【高】
- Dynamic 路径由 `compiled_dynamic_config` 使用每个 Expert/FFN 的 `compiler_bank_plans`，逐 stage 改变候选 pools；纯 Dynamic 进行确定性 rotation，PIVOT 使用 prefetch-coordinated origin。【中；实现明确，但这一部分是未提交修改】
- 最终 physical selection 由 `VirtualBankMappingTable.allocate` 根据 capacity 与 current pressure 排序；映射稳定至 release。【高】
- 当前实现没有 live migration。【高；`architecture.json::live_data_migration=false`】

## 7. PIVOT Prefetch

DATE3 PIVOT 使用 `CoverageAccuracyConstrainedPrefetchPolicy`，候选 `Chunk={1,2,4,8}`、`Window={1,2,4,8,16,32,64}`，并从低 pressure/capacity-feasible Bank groups 中保留最多 4 个候选。【高；`configs/MoE/DATE3/architecture.json::policy`，`pivot_ca_prefetch.py::_groups/choose`】

候选 score 联合考虑 predicted latency benefit、occupancy、pressure、conflict 和 mapping cost；约束 coverage/accuracy、residency ratio、timing margin，并使用 EMA feedback、cooldown/hysteresis 和有限 step adaptation。Chunk 在 DATE3 paper path 表示一个 HBM request 合并的 atomic Weight tiles 数，不只是 degree；Window 决定 Router 可见之后的 lead time。【高；`CoverageAccuracyPolicyConfig`、`choose`，`memdomain_runner.py::_fixed_issue_schedule`】

Routing 不是 future-expert prediction。`EPContract.routes` 由配置中的 token counts 在 routing 已知后确定性构造 Top-k assignments；多层 fixed/PIVOT issue schedule 受每层 Router visibility 限制，禁止 L 层触发 L+1 的请求。【高；`date3_ep_model.py::deterministic_routes_from_counts`，`_fixed_issue_schedule`】

及时、late、unused、evicted-before-use 的定义来自实际 load completion、first use、residency 和 release；coverage/accuracy 使用 unique useful timely bytes。【高；`streaming_residency.py`，`pivot_ca_prefetch.py::quality_from_lifetimes`，`documentation/DATE3/DATE3_DATA_CONTRACT.md`】

## 8. Guard / Safe Mechanism

- DATE2 `BankAwarePrefetchPolicy.decide(..., guard_incumbent=True)` 可在动态 group 不优时保留 fixed Bank incumbent；capacity/slack 不安全时 delay/cancel。【高；`prefetch_policy.py`】
- DATE3 主要 guard 在 `pivot_ca_runner.py::run_pivot_ca`：每个 epoch 在相同 completed prefix 上真实比较 adaptive proposal、frozen fixed-PF 和 coalesced NoPF 的 memory cost，提交三者最小者；日志写入 `online_incumbent_guard.csv`。【高】
- 这不是 Oracle：只比较当前已可见 prefix/epoch。Oracle/Ideal 不进入在线 policy。【高；`DATE3_DATA_CONTRACT.md`、guard call chain】
- legacy `select_safe_prefetch` 和 matrix-level `derive_selected_row` 仍存在。后者的 `Oracle` 是从已执行 candidates 事后选最小，明确为分析参考。【高；`prefetch_policy.py`、`memdomain_experiment.py`】

## 9. HBM Model

`offchip_load_cycles = startup + ceil(size_bytes/(bits_per_cycle/8))`，每个 scheme 使用相同参数。`StreamingResidencyEngine.run` 维护一个 `hbm_available_cycle`，所有 demand/prefetch Weight loads 共用一个非抢占串行 HBM channel，因此大/早 prefetch 会形成 queue wait；HMB completion 后才进入 on-chip mapping/Bank service。【高；`memdomain_runner.py::offchip_load_cycles`，`streaming_residency.py::run`】

模型真实覆盖 HBM bandwidth/startup contention、Bank capacity wait、Bank port conflict、prefetch-vs-compute interference、late/unused/eviction 分类；但它不是 DRAM command/bank/row-buffer 级模型，也没有完整 NoC/remote-NPU cycle simulation。Peer NPU 是解析模型，系统 critical path 为 `max(detailed_ready, peer_ready)+combine`。【高；`date3_ep_system.py::build_ep_system_timeline`】

## 10. Supported Schemes, Policies, Modes and Flags

公开 DATE3 schemes（同一 EP/system envelope）：【高；`scripts/DATE3/experiment_contract.py::PUBLIC_BASELINES`，`memdomain_runner.py::run_paper_control_executions`】

| Public scheme | Actual behavior |
|---|---|
| Static-555-NoPF | 固定 5:5:5:15，Weight demand load，无预取 |
| Static-Opt-NoPF | workload-wide offline static allocation，冻结全程，无预取 |
| Dynamic-NoPF | per-stage compiler/dynamic mapping；无预取；保留 matched static incumbent |
| Static-Opt-FixedPF | Static-Opt mapping + 固定 Window/Chunk schedule |
| Dynamic-FixedPF | Dynamic mapping + 与 static fixed-PF 匹配的 issue workload |
| PIVOT | online Chunk/Window/Bank-group 联合选择 + prefix incumbent guard |
| Ideal-NoPF | conflict-free NoPF lower bound，仅 reference，不可实现 |

内部 matrix enum 在 `memdomain_experiment.py::Baseline`：`Static-NoPF`、`Static-NaivePF`、`Dynamic-NoPF`、`Dynamic-NaivePF`、`MemDomain-Raw`、`MemDomain-Safe`、`Oracle`。内部 `Static-NoPF` 是历史 cyclic Weight search，不能直接重命名为 literal Static-555；因此 DATE3 另建 public control path。【高】

配置 `policy.prefetch_policy` 接受 `none`、`naive_fixed`、`bank_aware_raw`、`bank_aware_guarded`、`coverage_accuracy_constrained`；runner CLI 支持 `--suite`、`--exp`、`--variant`、`--force`、`--dry-run`、`--skip-details` 和 Exp6 filters。【高；`load_runner_config`、`run_date3_experiments.py::main`】

Static/Dynamic/FixedPF controls 共用 `run_raw_baseline_with_details -> StreamingResidencyEngine -> UnifiedBankDomain`；PIVOT 通过 `run_pivot_ca` 生成动作，但最终仍共用相同 residency、mapping、Bank、HBM 和 EP primitives。【高】

## 11. Four Workloads

源和派生文件层级如下：【高】

| Level | Location | Meaning |
|---|---|---|
| 384-d intermediate source | `topologies/MoE/{HMoE,Mixtral,MoDSE,Switchtrans}.csv` | 256 global tokens；不是原模型 full scale |
| DATE2 reduced (96-d) | `topologies/MoE/DATE2/models/*.csv` | N/K 统一除 4 并 16 对齐；M 不变 |
| DATE3 reduced (96-d) | `topologies/MoE/DATE3/models/*.csv` | 从 DATE2 复制；DATE3 配置的真正来源 |
| DATE3 experiment copies | `topologies/MoE/DATE3/{overall,end_to_end,...}` | suite provenance anchors |
| JSON request workloads | `configs/MoE/DATE2/**`、`configs/MoE/DATE3/**` | chunks、requests、compiler plans、routing/EP、hardware |
| Original/full-scale topology used by estimator | `outputs/DATE3/fullscale_estimation/topologies/*.csv` | estimator 生成的目标 M/N/K；不进入 event simulator |

DATE3 main workload 均为 8 Experts、Top-1、256 tokens，expert counts 为 `[32,48,50,24,34,28,21,19]`。HMoE/MoDSE 是异构 expert sizes；Mixtral/Switchtrans 是同构。【高；overall JSON `topology_provenance`】

DATE3 EP 默认 2 NPU，contiguous balanced ownership，Detailed NPU=0；routes 从 counts 确定性构造，每 Token 恰有 Top-k 个不同 Expert replicas。当前不是采集自真实模型运行的 token-by-token router trace，而是 controlled count trace 的确定性 realization。【高；`date3_ep_model.py`】

原模型规格由 full-scale estimator 固化为：【中；`outputs/DATE3/fullscale_estimation/fullscale_model_specs.csv` 和 estimator `TARGETS`，但属于 ignored/uncommitted estimation package】

| Model | M | d_model | Expert hidden sizes | Native top-k / evaluated main top-k |
|---|---:|---:|---|---|
| HMoE | 4096 | 1024 | 2304,2816,3328,3840,4352,4864,5376,5888 | 1 / 1 |
| Mixtral | 4096 | 4096 | 14336×8 | 2 / 1 |
| MoDSE | 4096 | 1536 | 6912,768,6144,1536,4608,3072,3840,3840 | 2 / 1 |
| Switchtrans | 4096 | 768 | 3072×8 | 1 / 1 |

因此历史描述中的 MoDSE 参数被当前 estimator 确认，但历史称 Top-k=1 是评估合同，不是 MoDSE native Top-k（estimator 标为 native Top-2）。【中】

## 12. Reduced-Scale Method

`scripts/prepare_date2_experiments.py::uniformly_scale_topology` 从 384-d intermediate CSV 出发，仅将 N/K 除以 4 并向上对齐 16；M（token/routed-token count）保持不变，Router N（expert count）不缩放。DATE3 再由 `scripts/DATE3/prepare_date3_experiments.py::date3_payload/attach_ep_contract` 复制 workload 并附 EP 合同。【高】

故 DATE3 的真实缩放 anchor 是 d_model=96、global M=256，而非 384-d 文件。所有 JSON 标记 `paper_scale_performance_claim=false` 和 `weight_scale_divisor=1`：Weight bytes 已按 reduced topology 直接生成，并没有在 event simulation 后乘一个 full-scale 系数。【高】

## 13. Full-Scale Estimation Method

真正存在的实现位于 ignored 目录：`outputs/DATE3/fullscale_estimation/scripts/fullscale_estimator.py`；审计、规格、验证、结果分别见同目录 `AUDIT_AND_SCALE_DECISION.md`、`fullscale_model_specs.csv`、`estimator_validation.csv`、`FULL_SCALE_ESTIMATION_REPORT.md`。【高；文件存在】

当前方法：【中/低，具体置信度如下】

1. 从 DATE3 的 M=256、d_model=96 anchor 恢复全局 M=4096；expert M 按相同 routing proportions 放大 16×为 `[512,768,800,384,544,448,336,304]`。【中】
2. N/K 同时恢复到表中 d_model/expert hidden；Mixtral Q=4096、K/V=1024，但 attention-score 仍为单代表 head 近似。【中】
3. IA=`M*K`、Weight=`N*K`、OA=`M*N`，ACC 按 16×16 output tiles ×1024 B 恢复；四类 traffic 都恢复，不是仅缩 Weight。【高；estimator `target_topology/fullscale_fig2a`】
4. Fig.2 使用与 DATE3 characterization 闭式 tile model 等价的解析重算；对 reduced MoDSE 23 stages×91 partitions=2093 点回放误差为 0。因此 Fig.2 full-scale 是重新按 M/N/K 计算，而非线性乘 cycles。【高；`validate_fast_characterization`、`estimator_validation.csv`】
5. Fig.1/4/5/6 先从原始 DATE3 CSV 重建当前图，再按恢复后的 tile、Chunk、Bank/HBM pressure 和 critical-path share 作趋势约束 central estimate；不是把所有 cycle 用同一个系数线性放大，也不是 full-scale event re-simulation。【中；实现存在；模型假设不可由 event simulator验证】
6. 同一 target topology、routing scale 和硬件用于所有 schemes，没有为 PIVOT 单独更改 scaling/hardware。【中；estimator source】
7. routing distribution 保持 reduced trace 比例；主实验仍 Top-1。它没有重放 native Mixtral/MoDSE Top-2 作为主结果。【中】
8. Expert Weight/activation/ACC/HBM 的 full-scale 压力进入估算特征；但除 Fig.2 外，HBM queue、Bank conflict、eviction 并未用 full-scale `StreamingResidencyEngine` 逐事件重仿真。【高；estimator/report 明示】
9. 最终 full-scale plotting tables 位于 `outputs/DATE3/fullscale_estimation/final_plotting_tables/`。除 Fig.2 外必须称 **calibrated estimates**，不能称 full-scale simulated measurements。【高】

这回答了 scaling 的关键限制：compute、traffic、HBM stall 和 conflict 在多处有非线性 tiling/capacity/queue 关系，不能简单线性 scaling；现有 estimator 对 Fig.2 重算了这些关系，但对 Fig.1/4/5/6 只做机制约束估算，未完成 full-scale event replay。【高】

## 14. Existing Final Results

### 14.1 当前实现哈希一致的最高可信 raw results

`outputs/DATE3/end_to_end/{model}/comparison.csv` 的 config hash 与当前配置一致，且四个 `metadata.json` 的 implementation hash 均等于当前 `868f24409032f03860b207c3de702c51e26d3d2613d15fd143aa655ede157920`，所需 PIVOT detail artifacts 齐全。【高】

下表直接来自这些 CSV；memory stall 是 `local_memory_stall_cycles`，speedup 以同模型 Static-555 为 1。`other=combine=320`，四组当前均无额外 exposed peer wait。【高】

| Model | Scheme | Compute | Memory stall | Total | Speedup |
|---|---|---:|---:|---:|---:|
| HMoE | Static-555-NoPF | 80,852 | 410,956 | 492,128 | 1.000× |
| HMoE | Static-Opt-NoPF | 80,852 | 289,675 | 370,847 | 1.327× |
| HMoE | Dynamic-NoPF | 80,852 | 280,514 | 361,686 | 1.361× |
| HMoE | Static-Opt-FixedPF | 80,852 | 218,106 | 299,278 | 1.644× |
| HMoE | Dynamic-FixedPF | 80,852 | 212,522 | 293,694 | 1.676× |
| HMoE | PIVOT | 80,852 | 194,086 | 275,258 | 1.788× |
| Mixtral | Static-555-NoPF | 98,192 | 485,407 | 583,919 | 1.000× |
| Mixtral | Static-Opt-NoPF | 98,192 | 338,892 | 437,404 | 1.335× |
| Mixtral | Dynamic-NoPF | 98,192 | 325,907 | 424,419 | 1.376× |
| Mixtral | Static-Opt-FixedPF | 98,192 | 242,911 | 341,423 | 1.710× |
| Mixtral | Dynamic-FixedPF | 98,192 | 230,182 | 328,694 | 1.776× |
| Mixtral | PIVOT | 98,192 | 219,791 | 318,303 | 1.834× |
| MoDSE | Static-555-NoPF | 58,592 | 245,380 | 304,292 | 1.000× |
| MoDSE | Static-Opt-NoPF | 58,592 | 172,374 | 231,286 | 1.316× |
| MoDSE | Dynamic-NoPF | 58,592 | 166,510 | 225,422 | 1.350× |
| MoDSE | Static-Opt-FixedPF | 58,592 | 142,739 | 201,651 | 1.509× |
| MoDSE | Dynamic-FixedPF | 58,592 | 134,812 | 193,724 | 1.571× |
| MoDSE | PIVOT | 58,592 | 113,246 | 172,158 | 1.768× |
| Switchtrans | Static-555-NoPF | 77,312 | 371,170 | 448,802 | 1.000× |
| Switchtrans | Static-Opt-NoPF | 77,312 | 260,390 | 338,022 | 1.328× |
| Switchtrans | Dynamic-NoPF | 77,312 | 250,242 | 327,874 | 1.369× |
| Switchtrans | Static-Opt-FixedPF | 77,312 | 192,246 | 269,878 | 1.663× |
| Switchtrans | Dynamic-FixedPF | 77,312 | 179,693 | 257,325 | 1.744× |
| Switchtrans | PIVOT | 77,312 | 167,162 | 244,794 | 1.833× |

这些是 reduced, 4-layer nonstationary DATE3 workloads，不是 original-scale full-model latency；配置的 `end_to_end_approximation` 仍忽略 embedding、normalization、softmax、residual 和 sampling。【高；end_to_end configs、`scripts/DATE3/build_experiment_compat.py::build_exp7`】

### 14.2 其他结果的可信度

- `outputs/DATE3/overall/*`：配置 hash 匹配，但保存实现 hash 为 `dc1d3a...`，与当前实现不一致，并缺 `baseline_metadata.json`；属于可追溯但 stale 的 raw results。【高】
- `outputs/DATE3/exp1`–`exp7`：paper-facing aggregate/compatibility tables；应追溯到相应 suite raw outputs，不能优先于 current-hash comparison CSV。【中】
- `outputs/DATE3/fullscale_estimation/final_plotting_tables/FIG5A_COST_BREAKDOWN_FINAL.csv`：full-scale mechanism estimates，不是 raw simulation。历史给出的 HMoE 前五行与该文件一致，但历史 PIVOT=183,629,056 与当前文件 PIVOT=150,633,728 冲突；当前 estimator/report 选择 72% reduction 的后者。不可将任一值写回 simulator。【高（冲突存在），中/低（估算值本身）】
- `outputs/DATE2/` 当前不存在，尽管提交中的 manifest/docs 指向它；因此无法从本机原始 DATE2 CSV 核验历史结果。【高】

## 15. Metric Definitions

DATE3 row 强制：

```text
total_cycles = compute_cycles
             + bank_stall_cycles
             + weight_load_stall_cycles
             + prefetch_miss_stall_cycles
             + prefetch_interference_stall_cycles
             + mapping_overhead_cycles
             + communication_stall_cycles
             + other_stall_cycles
```

【高；`memdomain_experiment.py::ExperimentRow.__post_init__`、`DATE3_DATA_CONTRACT.md`】

Public comparison 的 `local_memory_stall_cycles` 是前五种 local memory components 之和；`speedup_vs_static = static_total/scheme_total`；reduction 为 `100*(1-scheme_total/static_total)`。【高；`run_date3_experiments.py::_local_memory_stall/write_comparison`，`build_experiment_compat.py`】

仓库中没有统一名为 `stall coverage` 的 production metric。Full-scale Fig.5 effectiveness 使用 `(reference exposed stall - residual exposed stall)/reference exposed stall`，但 reference 随 scheme 改变：Static-FixedPF 对 Static-Opt，Dynamic-FixedPF/PIVOT 对 Dynamic-NoPF，不是全部对 Static-555。【高；`FIG5A_EFFECTIVENESS_FINAL.csv`】这与历史文字“全部相对 Static-Baseline exposed stall”的单一定义冲突。

## 16. Key File Index

- Entrypoints：`run_date2_experiments.py`、`run_date3_experiments.py`
- Preparation：`scripts/prepare_date2_experiments.py`、`scripts/DATE3/prepare_date3_experiments.py`
- Workload loader/generator：`scalesim/memory/topology_workload.py`
- Routing/EP：`scalesim/memory/date3_ep_model.py`、`date3_ep_system.py`
- Hardware contract/compiler：`buckyball_memdomain.py`、`buckyball_compiler.py`
- Canonical rows/schemes：`memdomain_experiment.py`
- Control runner：`memdomain_runner.py`
- Unified physical Bank service：`unified_bank_domain.py`
- Virtual mapping：`virtual_bank_mapping.py`
- Streaming residency/HBM：`streaming_residency.py`
- DATE2 prefetch policies：`prefetch_policy.py`
- DATE3 PIVOT policy/runner：`pivot_ca_prefetch.py`、`pivot_ca_runner.py`
- Public naming/contracts：`scripts/DATE3/experiment_contract.py`
- Result validation：`scripts/DATE2/validate_date2_contracts.py`、`scripts/DATE3/validate_date3_contracts.py`、`validate_paper_experiments.py`
- Full-scale estimator：`outputs/DATE3/fullscale_estimation/scripts/fullscale_estimator.py`
- Raw current DATE3 result root：`outputs/DATE3/`

## 17. Exact Reproduction Commands

先恢复环境；当前 `.venv` 缺依赖，下面的实验命令在依赖恢复前不能成功：【高】

```sh
.venv/bin/python -m pip install -r requirements.txt
```

生成配置会重写**派生** DATE2/DATE3 configs/topologies 并修复旧服务器绝对路径；执行前应先备份当前未提交工作树和 ignored outputs。审计本轮未执行这些命令。

```sh
.venv/bin/python scripts/prepare_date2_experiments.py
.venv/bin/python scripts/DATE3/prepare_date3_experiments.py
```

只读查看将运行/恢复哪些 DATE3 variants：

```sh
.venv/bin/python run_date3_experiments.py --exp exp4 --dry-run --skip-details
.venv/bin/python run_date3_experiments.py --exp exp5 --dry-run --skip-details
.venv/bin/python run_date3_experiments.py --exp exp6 --dry-run --skip-details
.venv/bin/python run_date3_experiments.py --exp exp7 --dry-run --skip-details
```

完整 DATE3 paper experiments（hash-resumable；不应使用 `--force`，除非明确要新目录/覆盖旧结果）：

```sh
.venv/bin/python run_date3_experiments.py --exp exp1
.venv/bin/python run_date3_experiments.py --exp exp2
.venv/bin/python run_date3_experiments.py --exp exp3 --skip-details
.venv/bin/python run_date3_experiments.py --exp exp4 --skip-details
.venv/bin/python run_date3_experiments.py --suite prefetch_calibration
.venv/bin/python run_date3_experiments.py --exp exp5 --skip-details
.venv/bin/python run_date3_experiments.py --exp exp6 --skip-details
.venv/bin/python run_date3_experiments.py --exp exp7 --skip-details
.venv/bin/python scripts/DATE3/validate_date3_contracts.py
.venv/bin/python scripts/DATE3/validate_paper_experiments.py
```

DATE2 已提交基线复现入口：

```sh
.venv/bin/python run_date2_experiments.py --suite all
.venv/bin/python scripts/DATE2/validate_date2_contracts.py
```

Full-scale estimator 的实现只写其专用目录：

```sh
.venv/bin/python outputs/DATE3/fullscale_estimation/scripts/fullscale_estimator.py
```

注意：当前 `source_path` 绝对路径失效，且依赖缺失；所以这些命令是代码规定的 exact entrypoints，不代表当前环境已具备立即复现条件。

## 18. Known Issues / Missing Pieces

1. **未提交 DATE3**：核心实现、配置、测试、文档、notebooks 和结果没有 Git provenance；服务器再次丢失会无法恢复。【高】
2. **结果分层不一致**：overall 的 implementation hash stale；end_to_end 是 current hash；不能混表。【高】
3. **缺 baseline metadata**：四个 end_to_end 目录有 current PIVOT metadata，但没有 `baseline_metadata.json`；当前 `current_matrix` 会判 matrix stale 并重跑 controls。【高】
4. **绝对路径失效**：DATE2 195 个 JSON 中 190 个、DATE3 257 个中 252 个 `topology_provenance.source_path` 指向不存在的 `/home/MikeNotFound/...`；会阻断 `profiled_static_allocation`/compiler service 等重跑路径。【高】
5. **Python 环境损坏**：`.venv`/system Python 缺 numpy/pandas，不能导入 simulator 或执行 validators。【高】
6. **DATE2 raw outputs 缺失**：`outputs/DATE2` 不存在；已提交代码和 docs 仍在，但原始结果无法本地复核。【高】
7. **Compute provenance 冲突**：architecture 写 SCALE-Sim trace，真实 generator 是解析 MAC timing。【高】
8. **Full-scale 非完整仿真**：除 Fig.2 外均为 calibrated estimates，Bank/HBM conflict 未 full-scale event replay；Mixtral 维度跨度最大，报告自评低置信度。【高】
9. **原生 Top-k 与主实验不同**：Mixtral/MoDSE estimator 标为 native Top-2，DATE3 main 是 Top-1 controlled workload。【高】
10. **End-to-end scope 漂移**：配置准备脚本注释称一个 block，但现存 end_to_end configs/aggregate 使用 `block_count=4`；full-scale 审计也记录此差异。【高】
11. **重复字典键**：`pivot_ca_runner.py::run_pivot_ca` 的 components literal 中 `other_stall_cycles` 出现两次；值相同，当前语义不变，但应在后续代码维护中清理。【高】
12. **HBM 是单通道抽象**：有真实串行 queue contention，但无 DRAM row/bank timing；论文措辞需限定。【高】

其中 1–5 会妨碍继续可信实验；必须先恢复依赖、修复/重新生成 portability-safe provenance，并把新实验输出放到新目录。不得直接覆盖现有 ignored results。

## 19. Historical Context vs Current Code Differences

| Historical claim | Current repository evidence | Resolution |
|---|---|---|
| HMoE PIVOT total 183,629,056 | 没有原始 CSV 命中该数；当前 full-scale estimate 是 150,633,728 | 历史值未确认，不采用 |
| Physical capacity about 61,440 B | contract/config 均为 61,440 B | 一致 |
| IA/W/OA/ACC=8/8/8/32 | contract/config 一致 | 一致 |
| Compute timing from SCALE-Sim trace | generator 使用 `ceil(MNK/256)` | 以代码为准，历史/architecture label 不准确 |
| PIVOT adaptive W/C/g | DATE3 policy 三者联合候选搜索 | 一致；DATE2 adaptive Window 更简单 |
| Safe 防回退 | DATE3 prefix 三选一 online guard | 一致且当前实现更明确 |
| Oracle 不泄漏在线 policy | Oracle 仅 matrix derivation；online guard 不调用 Oracle | 一致 |
| Four full models were simulated | current JSON 明示 reduced、`paper_scale_performance_claim=false` | 冲突；只有 reduced simulation + full-scale estimates |
| Full-scale uses uniform method | estimator 对 scheme 统一，但 Fig.2 重算、其他图估算 | 部分一致，需披露方法差异 |
| Stall coverage always relative Static baseline | final effectiveness references differ by scheme | 冲突；需统一论文定义或保持现文件定义 |
| Exp7 one complete block | existing configs/results use four blocks | 冲突；以现有 configs/results 为准 |

## 20. Confidence Summary and Next Insertion Point

| Conclusion | Confidence |
|---|---|
| Branch/commit/worktree snapshot | 高 |
| Hardware and precision contract | 高 |
| Unified Bank/mapping/residency/HBM behavior | 高 |
| Public scheme semantics and common resource envelope | 高 |
| DATE3 online guard has no Oracle call | 高 |
| Reduced workload dimensions/routing | 高 |
| Current-hash end_to_end raw results | 高（仅限 reduced 4-layer workload） |
| Stale overall results | 中，不应作为 final current results |
| Full-scale Fig.2 | 高（解析模型合同内） |
| Full-scale Fig.1/4/5/6 | 中到低，必须标 estimate |
| Full public model end-to-end claim | 未确认/低 |

后续增加 related-work baseline 的最佳插入位置不是 PIVOT policy 内部，而是同一 control execution path：

1. 在 `memdomain_experiment.py::Baseline` 增独立 internal flag/row；
2. 在 `memdomain_runner.py::run_raw_baseline_with_details` 或单独 sibling runner 实现其动作，同时复用 `StreamingResidencyEngine/UnifiedBankDomain`；
3. 在 `run_paper_control_executions` 注册公共 control；
4. 在 `run_date3_experiments.py::write_comparison` 和 `scripts/DATE3/experiment_contract.py::PUBLIC_BASELINES` 注册名称/合同；
5. 扩展 validators/tests，保持同一 hardware、workload、EP 和 hash provenance。

这一位置能让新 baseline 与现有 schemes 共享真正的 request/residency/HBM/EP 路径，并避免接触 PIVOT 的在线决策状态。【高；当前调用结构】
