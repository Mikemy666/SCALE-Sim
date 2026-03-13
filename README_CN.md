下面是你给出的 **SCALE-Sim README** 的完整中文翻译版（保持原结构，方便你对照原文阅读）。

---

# 脉动阵列加速器模拟器（SCALE-Sim）

SCALE-Sim 是一个用于模拟 **基于脉动阵列（systolic array）的深度学习加速器** 的工具。
它支持多种深度神经网络层，例如：

* 卷积层（Convolution）
* 全连接层（Fully Connected）
* 以及所有基于 **GEMM（矩阵乘法）** 的层（例如 Attention）。

---

# SCALE-Sim 各版本的功能

## v2 版本的功能

SCALE-Sim v2 具有以下特性：

1. 支持 **GEMM 和卷积（通过 im2col 实现）** 的仿真
2. **计算周期（compute cycles）使用解析模型**，并通过 RTL 仿真验证
3. 对 **输入矩阵、权重矩阵和输出矩阵** 建立独立的双缓冲（double-buffered）内存模型
4. **多精度仿真模式（Multi-Fidelity）**

   * 带宽计算模式（CALC）
   * stall 周期计算 / 用户指定带宽模式（USER）
5. 可以分别保存 **输入、权重、输出的 SRAM 与 DRAM 的周期级访问 trace**

SCALE-Sim v2 架构示意图：

(图：scalesim overview)

---

### SCALE-Sim v1

SCALE-Sim v1 是一个更早期的版本，由 ARM 参与开发：

[https://github.com/ARM-software/SCALE-Sim](https://github.com/ARM-software/SCALE-Sim)

该版本已经不再维护。

---

# SCALE-Sim v3 新增功能

相比 v2，SCALE-Sim v3 引入了许多高级特性：

1. **稀疏支持（Sparsity Support）**

   * 支持按层和按行的稀疏性

2. **Ramulator 集成**

   * 使用更精细的 DRAM 内存模型
   * 可评估 DRAM 性能

3. **Accelergy 集成**

   * 支持能耗和功耗估计

4. **内存布局支持（Layout Support）**

   * 支持更复杂的片上内存布局

5. **多核支持（Multi-core Support）**

   * 可以模拟多个 systolic array / tensor core

(图：scalesim v3 features)

---

# 30 秒快速上手

## 安装

SCALE-Sim 完全由 **Python** 编写，可以直接从源码安装。

安装命令：

```bash
pip3 install <scale-sim-v3目录路径>
```

如果你是开发者，需要修改 SCALE-Sim 代码，建议使用：

```bash
pip3 install -e <scale-sim-v3目录路径>
```

`-e` 参数会创建 **符号链接**，这样：

* 修改源码
* 环境中的 SCALE-Sim 会立即更新

不需要重新安装。

---

# 运行一次仿真

安装后可以使用下面的命令运行：

```bash
python3 -m scalesim.scale \
-c <config文件路径> \
-t <topology文件路径> \
-p <输出目录>
```

参数说明：

* `-c`：架构配置文件
* `-t`：网络拓扑文件
* `-p`：输出日志目录

---

# 从源码直接运行

如果你不想安装 package，可以直接运行源码：

```bash
PYTHONPATH=$PYTHONPATH:<scale_sim_repo_root> \
python3 <scale_sim_repo_root>/scalesim/scale.py \
-c <config文件路径> \
-t <topology文件路径>
```

如果第一次运行，需要先安装依赖：

```bash
pip3 install -r <scale_sim_repo_root>/requirements.txt
```

---

# 工具输入

SCALE-Sim 运行需要两个输入文件：

1️⃣ **配置文件（configuration file）**
2️⃣ **拓扑文件（topology file）**

---

# 配置文件（Configuration file）

配置文件用于指定：

* 模拟的硬件架构
* 仿真运行参数

示例：

(图：config-file-example)

配置文件包含三个部分：

### general

用于指定：

运行名称（run name）

这是用户自定义的。

---

### architecture_presets

用于描述 **脉动阵列硬件参数**，例如：

* systolic array 尺寸
* buffer 大小
* memory bandwidth 等

---

### run_preset

用于指定运行模式：

1️⃣ 使用用户给定带宽
2️⃣ 自动计算 **无 stall 执行所需的最优带宽**

详细文档：待补充（TBD）。

---

# 拓扑文件（Topology file）

拓扑文件是一个 **CSV 文件**。

它描述 **神经网络每一层的参数**。

通常以卷积层形式描述，例如：

(图：topo-file-example)

---

## GEMM 格式输入

SCALE-Sim 也支持直接输入 **GEMM 的 M/N/K 描述**：

(图：mnk topo example)

如果使用 GEMM 格式，需要加参数：

```bash
-i gemm
```

示例：

```bash
python3 scalesim/scale.py \
-c config.cfg \
-t gemm_topology.csv \
-i gemm
```

---

# 输出结果

例如运行 YOLO-Tiny 时，终端会输出：

(图：screen_out)

同时还会生成以下文件：

默认输出目录：

```
scalesim_outputs/
```

也可以用 `-p` 自定义输出路径。

---

## 生成的三个报告

### COMPUTE_REPORT.csv

每一层：

* 计算周期
* stall 周期
* 利用率

---

### BANDWIDTH_REPORT.csv

每一层：

* SRAM 平均带宽
* SRAM 最大带宽
* DRAM 平均带宽
* DRAM 最大带宽

---

### DETAILED_ACCESS_REPORT.csv

记录：

* SRAM 访问次数
* DRAM 访问次数
* 每种操作数的访问周期

---

## 访问 trace

SCALE-Sim 还会生成：

* SRAM access trace
* DRAM access trace

路径：

```
scalesim_outputs/<run_name>/
```

---

# 高级功能

## 多核支持（Multi-core）

SCALE-Sim v3 新增了 **多核仿真能力**。

SCALE-Sim v2 只能模拟 **单个 systolic array**。

v3 可以模拟：

* 多个 tensor core
* 多核 AI 加速器

详细说明：

```
multi-core/README.md
```

---

# 稀疏性支持

支持：

* layer-wise sparsity
* row-wise sparsity
* N:M 稀疏

支持稀疏格式：

* CSR
* CSC
* Blocked ELLPACK

并提供详细稀疏报告。

文档：

```
README_Sparsity.md
```

---

# Ramulator 内存模型

SCALE-Sim v3 集成了 **Ramulator DRAM 模型**。

可以评估：

* memory stall cycles
* bank conflicts
* 不同 DRAM 类型（DDR3 / DDR4）
* 不同 memory configuration

文档：

```
README_ramulator.md
```

---

# Accelergy 能耗模型

SCALE-Sim v3 集成 **Accelergy**。

用于估计：

* systolic array energy
* power consumption

同时支持：

* CACTI
* Aladdin

文档：

```
README_accelergy.md
```

---

# Layout（内存布局）

支持自定义 on-chip buffer 的布局。

功能：

### 自定义数据组织

可以分别指定：

* ifmap
* filter
* ofmap

的内存布局。

---

### Bank 冲突评估

模拟真实 memory access pattern。

---

### 多 bank 支持

可以配置：

* bank 数量
* bank 端口数量

---

### 布局定义参数

三个关键参数：

`intraline_factor`

每一行包含多少元素

`intraline_order`

一行内维度排列顺序

`interline_order`

行之间维度排列顺序

---

# 详细文档

SCALE-Sim v3 论文：

**SCALE-Sim v3: A modular cycle-accurate systolic accelerator simulator**

ISPASS 2025

作者：

* Ritik Raj
* Sarbartha Banerjee
* Nikhil Chandra
* Zishen Wan
* Jianming Tong
* Ananda Samajdhar
* Tushar Krishna

---

# 推荐阅读论文

1️⃣ SCALE-Sim ISPASS 2020

"A systematic methodology for characterizing scalability of DNN accelerators using SCALE-sim"

2️⃣ SCALE-Sim 2018

"SCALE-Sim: Systolic CNN Accelerator Simulator"

---

# 引用方式

如果你的论文使用了 SCALE-Sim，请引用：

```bibtex
@inproceedings{raj2025scale,
title={SCALE-Sim v3: A modular cycle-accurate systolic accelerator simulator for end-to-end system analysis},
author={Raj, Ritik and Banerjee, Sarbartha and Chandra, Nikhil and Wan, Zishen and Tong, Jianming and Samajdhar, Ananda and Krishna, Tushar},
booktitle={ISPASS},
year={2025}
}
```

---

# 贡献代码

欢迎贡献代码。

流程：

1. 先提交 issue 讨论修改
2. 更新 documentation
3. 提供 Jupyter notebook 教程
4. 添加测试用例
5. 更新版本号和 CHANGELOG
6. 需要两位开发者 review

---

# 开发团队

开发者：

Ritik Raj

顾问：

* Ananda Samajdar
* Tushar Krishna

---

## v3 贡献者

* Sarbartha Banerjee（Ramulator）
* Nikhil Chandra（Sparsity）
* Zishen Wan（Accelergy）
* Jianming Tong（Layout）

---

## v2 贡献者

* Jan Moritz Joseph
* Yuhao Zhu
* Paul Whatmough
* Vineet Nadella
* Sachit Kuhar

---

如果你愿意，我还可以帮你 **用一页图给你讲清楚 SCALE-Sim 整个架构（systolic array + memory + trace + stall cycle）**，这对你做 **MoE memory bottleneck 实验**会非常关键。
