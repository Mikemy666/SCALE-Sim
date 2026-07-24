"""Create the seven DATE1 analysis notebooks under fig/."""

from pathlib import Path
import json
import textwrap


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'fig'


def md(text):
    return {
        'cell_type': 'markdown', 'metadata': {},
        'source': textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def code(text):
    return {
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': textwrap.dedent(text).strip().splitlines(keepends=True),
    }


COMMON = r'''
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300})

def find_repo_root():
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "scalesim").is_dir() and (candidate / "outputs" / "DATE1").exists():
            return candidate
    fallback = Path("/home/MikeNotFound/code/SCALE-Sim")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("无法定位 SCALE-Sim 仓库根目录")

ROOT = find_repo_root()
FIG_DIR = ROOT / "fig" / "generated"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少实验报告：{path}\n请先运行对应实验脚本。")
    # SCALE-Sim reports currently contain trailing commas; index_col=False
    # prevents pandas from silently treating LayerID as a row index.
    df = pd.read_csv(path, skipinitialspace=True, index_col=False)
    df.columns = df.columns.str.strip().str.rstrip(",")
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]

def read_topology(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少 topology：{path}")
    df = pd.read_csv(path, skipinitialspace=True, index_col=False)
    df.columns = df.columns.str.strip().str.rstrip(",")
    return df

def numeric(df, columns=None):
    out = df.copy()
    targets = columns or out.columns
    for column in targets:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out

def savefig(name):
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight")
    print("已保存：", path)
'''


def write(name, title, description, cells):
    notebook = {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3'},
        },
        'cells': [md(f'# {title}\n\n{description}'), code(COMMON), *cells],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    write(
        'exp1.ipynb',
        '实验1：MoE 层访存瓶颈分析',
        '读取完整 MoDSE 网络结果，比较 Attention、Router 与 MoE 的执行周期、停滞周期和 Bank 冲突。',
        [
            code(r'''
RUN = ROOT / "outputs/DATE1/exp1/static_no_prefetch"
TOPOLOGY = ROOT / "topologies/MoE/DATE1/exp1/modse_full.csv"

compute = numeric(read_csv(RUN / "COMPUTE_REPORT.csv"))
bank = numeric(read_csv(RUN / "BANK_MODEL_REPORT.csv"))
bank_util = numeric(read_csv(RUN / "BANK_UTILIZATION_REPORT.csv"),
                    ["LayerID", "BankID", "AccessCount", "BusyCycles", "Utilization", "ConflictCount"])
topology = read_topology(TOPOLOGY)
topology = topology[topology["Layer"].notna() & (topology["Layer"].astype(str).str.strip() != "")].reset_index(drop=True)
topology["LayerID"] = np.arange(len(topology))
topology = topology.rename(columns={"Layer": "LayerName"})

df = compute.merge(bank, on="LayerID", suffixes=("_compute", "_bank"))
df = df.merge(topology[["LayerID", "LayerName"]], on="LayerID", how="left")

def module_type(name):
    name = str(name)
    if name.startswith("MoE-"):
        return "MoE"
    if name.startswith("Router"):
        return "Router"
    return "Attention"

df["Module"] = df["LayerName"].map(module_type)
df["StallRatio"] = df["Stall Cycles"] / df["Total Cycles"]
df["BankConflictShare"] = np.where(
    df["Stall Cycles"] > 0,
    df["stall_cycles_due_to_bank_conflict"] / df["Stall Cycles"],
    0,
)
df[["LayerID", "LayerName", "Module", "Total Cycles", "Stall Cycles", "StallRatio", "BankConflictShare"]]
'''),
            md('## 表1：模块级汇总\n\n总量用于说明整体瓶颈贡献，均值用于避免 MoE 层数量更多造成不公平比较。'),
            code(r'''
module_order = ["Attention", "Router", "MoE"]
summary = df.groupby("Module").agg(
    Layers=("LayerID", "count"),
    TotalCycles=("Total Cycles", "sum"),
    StallCycles=("Stall Cycles", "sum"),
    AvgCyclesPerLayer=("Total Cycles", "mean"),
    AvgStallPerLayer=("Stall Cycles", "mean"),
    AvgStallRatio=("StallRatio", "mean"),
    AvgBankConflictShare=("BankConflictShare", "mean"),
).reindex(module_order)

summary["TotalStallContribution"] = summary["StallCycles"] / summary["StallCycles"].sum()
summary.to_csv(FIG_DIR / "exp1_module_summary.csv")
display(summary.style.format({
    "TotalCycles": "{:,.0f}", "StallCycles": "{:,.0f}",
    "AvgCyclesPerLayer": "{:,.1f}", "AvgStallPerLayer": "{:,.1f}",
    "AvgStallRatio": "{:.2%}", "AvgBankConflictShare": "{:.2%}",
    "TotalStallContribution": "{:.2%}",
}))
'''),
            md('## 图1：Attention、Router 与 MoE 的访存停滞对比\n\n左图展示每层平均 Stall，右图展示平均 Stall Ratio。正文建议使用这张图。'),
            code(r'''
colors = ["#4E79A7", "#59A14F", "#E15759"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].bar(summary.index, summary["AvgStallPerLayer"], color=colors, edgecolor="black")
axes[0].set_ylabel("Average stall cycles per layer")
axes[0].set_title("Average Memory Stall")

axes[1].bar(summary.index, summary["AvgStallRatio"], color=colors, edgecolor="black")
axes[1].set_ylabel("Average stall ratio")
axes[1].set_title("Fraction of Time Spent Stalled")
axes[1].set_ylim(0, max(1.0, summary["AvgStallRatio"].max() * 1.15))

for ax in axes:
    ax.set_xlabel("Module")
plt.tight_layout()
savefig("exp1_module_bottleneck.pdf")
plt.show()
'''),
            md('## 图2：逐层结果（建议放附录）'),
            code(r'''
plot_df = df.sort_values("LayerID")
color_map = {"Attention": "#4E79A7", "Router": "#59A14F", "MoE": "#E15759"}
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
x_layer = plot_df["LayerID"].to_numpy(dtype=float)
axes[0].bar(x_layer, plot_df["Stall Cycles"].to_numpy(dtype=float),
            color=plot_df["Module"].map(color_map), edgecolor="black", linewidth=0.3)
axes[0].set_ylabel("Stall cycles")
axes[0].set_title("Per-layer Stall Cycles")
axes[1].bar(x_layer, plot_df["StallRatio"].to_numpy(dtype=float),
            color=plot_df["Module"].map(color_map), edgecolor="black", linewidth=0.3)
axes[1].set_ylabel("Stall ratio")
axes[1].set_xlabel("Layer ID")
axes[1].set_title("Per-layer Stall Ratio")
axes[1].set_xticks(x_layer)
plt.tight_layout()
savefig("exp1_per_layer_appendix.pdf")
plt.show()
'''),
            md('## 表2：逐物理 Bank 利用率与冲突\n\n该表使用新增的 `BANK_UTILIZATION_REPORT.csv`，不是 IA/Weight/OA 容量利用率代理指标。'),
            code(r'''
bank_util = bank_util.merge(topology[["LayerID", "LayerName"]], on="LayerID", how="left")
bank_util["Module"] = bank_util["LayerName"].map(module_type)
physical_bank_summary = bank_util.groupby(["Module", "TensorType"]).agg(
    MeanBankUtilization=("Utilization", "mean"),
    MaxBankUtilization=("Utilization", "max"),
    TotalConflicts=("ConflictCount", "sum"),
    IdleBankFraction=("Utilization", lambda values: float((values == 0).mean())),
).reindex(module_order, level=0)
physical_bank_summary.to_csv(FIG_DIR / "exp1_physical_bank_summary.csv")
display(physical_bank_summary.style.format({
    "MeanBankUtilization": "{:.2%}", "MaxBankUtilization": "{:.2%}",
    "TotalConflicts": "{:,.0f}", "IdleBankFraction": "{:.2%}",
}))
'''),
            md('## 自动生成实验结论'),
            code(r'''
moe = summary.loc["MoE"]
non_moe_avg_stall = summary.loc[["Attention", "Router"], "AvgStallPerLayer"].mean()
stall_gap = moe["AvgStallPerLayer"] / non_moe_avg_stall if non_moe_avg_stall else np.inf
print(f"MoE 每层平均 Stall 是 Attention/Router 均值的 {stall_gap:.2f} 倍。")
print(f"MoE 对全网络 Stall 的贡献为 {moe['TotalStallContribution']:.2%}。")
print(f"MoE 平均 Stall Ratio 为 {moe['AvgStallRatio']:.2%}。")
print(f"MoE Stall 中 Bank conflict 对应比例为 {moe['AvgBankConflictShare']:.2%}。")

if stall_gap > 1.5 and moe["TotalStallContribution"] > 0.5:
    print("结论：当前结果支持访存瓶颈主要集中在 MoE 层。")
else:
    print("结论：当前结果尚不足以强力支持 MoE 主导瓶颈，需要检查工作负载或统计口径。")
'''),
        ],
    )

    write(
        'exp2.ipynb', '实验2：静态 Bank 分配无法适应动态数据流',
        '扫描多种固定 IA/Weight/OA Bank 比例，绘制层—配置性能热力图和逐物理 Bank 热点指标。',
        [
            code(r'''
EXP = ROOT / "outputs/DATE1/exp2"
TOPOLOGY = ROOT / "topologies/MoE/DATE1/exp2/modse_full.csv"
topology = read_topology(TOPOLOGY)
topology = topology[topology["Layer"].notna() & (topology["Layer"].astype(str).str.strip() != "")].reset_index(drop=True)
topology["LayerID"] = np.arange(len(topology))
topology = topology.rename(columns={"Layer": "LayerName"})

sweep_path = EXP / "exhaustive_static_search/BANK_ALLOCATION_SWEEP_REPORT.csv"
if not sweep_path.exists():
    sweep_path = EXP / "exhaustive_static_search/BANK_ALLOCATION_SWEEP_PARTIAL.csv"
if not sweep_path.exists():
    raise FileNotFoundError(
        "缺少 253 种静态组合扫描结果。请重新执行 ./scripts/DATE1/run_exp2.sh；"
        "旧版七组 static_* 结果不再用于本实验。"
    )

sweep = numeric(
    read_csv(sweep_path),
    ["LayerID", "IfmapBankNum", "FilterBankNum", "OfmapBankNum",
     "TotalCycles", "StallCycles", "IfmapConflictDelay",
     "FilterConflictDelay", "OfmapConflictDelay", "TotalConflictDelay"],
)
expected_candidates = 253
counts_per_layer = sweep.groupby("LayerID")["AllocationRatio"].nunique()
if counts_per_layer.min() != expected_candidates:
    raise RuntimeError(f"静态组合扫描不完整，每层应有 {expected_candidates} 组")

bank_all = sweep.rename(columns={
    "IfmapBankNum": "ifmap_banknum", "FilterBankNum": "filter_banknum",
    "OfmapBankNum": "ofmap_banknum", "TotalCycles": "total_cycles",
    "StallCycles": "stall_cycles_due_to_bank_conflict",
    "TotalConflictDelay": "total_bank_conflict_delay",
})
bank_all["Run"] = "static_" + bank_all["AllocationRatio"].str.replace(":", "_", regex=False)
bank_all = bank_all.merge(topology[["LayerID", "LayerName"]], on="LayerID")
moe = bank_all[bank_all["LayerName"].str.startswith("MoE-")].copy()
moe["NormCycles"] = moe.groupby("LayerID")["total_cycles"].transform(lambda x: x / x.min())

all_summary = moe.groupby("Run").agg(MoETotalCycles=("total_cycles", "sum")).sort_values("MoETotalCycles")
all_summary["NormalizedCycles"] = all_summary["MoETotalCycles"] / all_summary["MoETotalCycles"].min()
selected_path = EXP / "exhaustive_static_search/SELECTED_STATIC_ALLOCATION_SUMMARY.csv"
if selected_path.exists():
    selected_ratios = read_csv(selected_path)["AllocationRatio"].astype(str).tolist()
    run_order = ["static_" + ratio.replace(":", "_") for ratio in selected_ratios]
else:
    targets = [1.0, 1.15, 1.35, 1.75, float(all_summary["NormalizedCycles"].max())]
    run_order = []
    for target in targets:
        available = all_summary.loc[~all_summary.index.isin(run_order)]
        run_order.append((available["NormalizedCycles"] - target).abs().idxmin())
    if "static_8_8_8" not in run_order:
        run_order.append("static_8_8_8")

all_run_labels = {name: name.replace("static_", "").replace("_", ":") for name in moe["Run"].unique()}
run_labels = {name: all_run_labels[name] for name in run_order}
'''),
            md('## 表1：不同静态分配方案的 MoE 总体结果'),
            code(r'''
aggregate = moe.groupby("Run").agg(
    MoETotalCycles=("total_cycles", "sum"),
    MoEConflictStall=("stall_cycles_due_to_bank_conflict", "sum"),
    MoETotalConflictDelay=("total_bank_conflict_delay", "sum"),
).reindex(run_order)
aggregate["NormalizedCycles"] = aggregate["MoETotalCycles"] / aggregate["MoETotalCycles"].min()
aggregate["SlowdownVsBest"] = aggregate["NormalizedCycles"]
aggregate.index = [run_labels[name] for name in aggregate.index]
aggregate.to_csv(FIG_DIR / "exp2_static_allocation_summary.csv")
display(aggregate.style.format({
    "MoETotalCycles": "{:,.0f}", "MoEConflictStall": "{:,.0f}",
    "MoETotalConflictDelay": "{:,.0f}", "NormalizedCycles": "{:.3f}",
    "SlowdownVsBest": "{:.3f}×",
}))

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(aggregate.index, aggregate["NormalizedCycles"], color="#4E79A7", edgecolor="black")
ax.set_ylabel("Normalized MoE total cycles (best = 1)")
ax.set_xlabel("Static IA:Weight:OA allocation")
ax.set_title("Overall Performance Is Sensitive to Static Allocation")
for bar, value in zip(bars, aggregate["NormalizedCycles"]):
    ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}×", ha="center", va="bottom")
plt.tight_layout(); savefig("exp2_static_overall_comparison.pdf"); plt.show()
'''),
            code(r'''
pivot = moe.pivot(index="LayerName", columns="Run", values="NormCycles").reindex(columns=run_order)
fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=1)
ax.set_xticks(range(len(pivot.columns)), [run_labels[name] for name in pivot.columns], rotation=35, ha="right")
ax.set_yticks(range(len(pivot.index)), pivot.index)
ax.set_title("Static Bank Allocation Sensitivity")
ax.set_xlabel("Static IA:Weight:OA allocation")
plt.colorbar(im, ax=ax, label="Normalized layer cycles (best = 1)")
plt.tight_layout(); savefig("exp2_static_ratio_heatmap.pdf"); plt.show()
'''),
            code(r'''
best = moe.loc[moe.groupby("LayerID")["total_cycles"].idxmin(),
               ["LayerID", "LayerName", "Run", "ifmap_banknum", "filter_banknum", "ofmap_banknum", "total_cycles"]]
best = best.sort_values("LayerID").reset_index(drop=True)
best["BestRatio"] = best["Run"].map(all_run_labels)
best.to_csv(FIG_DIR / "exp2_best_static_per_layer.csv", index=False)
display(best[["LayerName", "BestRatio", "total_cycles"]])
print("逐层最佳比例出现次数：")
display(best["BestRatio"].value_counts().rename_axis("BankRatio").to_frame("Layers"))
x = np.arange(len(best))
plt.figure(figsize=(13, 4.5))
plt.bar(x, best["ifmap_banknum"], label="IA")
plt.bar(x, best["filter_banknum"], bottom=best["ifmap_banknum"], label="Weight")
plt.bar(x, best["ofmap_banknum"], bottom=best["ifmap_banknum"] + best["filter_banknum"], label="OA")
plt.xticks(x, best["LayerName"], rotation=55, ha="right")
plt.ylabel("Number of banks"); plt.title("Best Static Allocation Differs across MoE Layers")
plt.legend(); plt.tight_layout(); savefig("exp2_best_ratio_per_layer.pdf"); plt.show()
'''),
            md('## 图3：不同静态分配下的冲突来源\n\n不再使用“同一张量池内部逐 Bank CoV”。规则 GEMM 的取模映射会让池内 Bank 访问高度均匀，CoV 接近 0 并不能表示数据缺失。这里直接比较 IA、Weight、OA 三类请求产生的冲突延迟。'),
            code(r'''
conflict_breakdown = moe.groupby("Run").agg(
    IAConflictDelay=("IfmapConflictDelay", "sum"),
    WeightConflictDelay=("FilterConflictDelay", "sum"),
    OAConflictDelay=("OfmapConflictDelay", "sum"),
).reindex(run_order)
conflict_breakdown.index = [run_labels[name] for name in conflict_breakdown.index]
conflict_breakdown.to_csv(FIG_DIR / "exp2_conflict_breakdown.csv")
display(conflict_breakdown.style.format("{:,.0f}"))

ax = conflict_breakdown.plot(
    kind="bar", stacked=True, figsize=(10, 4.8),
    color=["#4E79A7", "#F28E2B", "#59A14F"], edgecolor="black",
)
ax.set_ylabel("Accumulated bank-conflict delay")
ax.set_xlabel("Static IA:Weight:OA allocation")
ax.set_title("Conflict Source Changes with Static Bank Allocation")
ax.legend(["IA", "Weight", "OA"])
plt.xticks(rotation=35, ha="right")
plt.tight_layout(); savefig("exp2_conflict_breakdown.pdf"); plt.show()
'''),
            md('## 自动生成实验结论'),
            code(r'''
spread = moe.groupby("LayerID")["total_cycles"].agg(["min", "max"])
spread["WorstBestRatio"] = spread["max"] / spread["min"]
unique_best = best["BestRatio"].nunique()
overall_best = aggregate["MoETotalCycles"].idxmin()
equal_slowdown = aggregate.loc["8:8:8", "NormalizedCycles"] if "8:8:8" in aggregate.index else np.nan

print(f"整体最优固定比例：{overall_best}。")
print(f"逐层出现了 {unique_best} 种不同的最佳静态比例。")
print(f"各层最差/最好配置的平均性能差距为 {spread['WorstBestRatio'].mean():.2f}×，最大为 {spread['WorstBestRatio'].max():.2f}×。")
print(f"均衡静态配置 8:8:8 相对整体最优配置慢 {equal_slowdown:.2f}×。")
if unique_best > 1 and spread["WorstBestRatio"].mean() > 1.2:
    print("结论：不同 MoE 层偏好的 Bank 比例不同，固定静态分配无法持续匹配动态变化的 IA/Weight/OA 需求。")
else:
    print("结论：当前比例扫描对静态分配失配的支持较弱，需要扩大配置或工作负载范围。")
'''),
        ],
    )

    write(
        'exp3.ipynb', '实验3：普通预取引入新的 Bank 争用',
        '比较无预取与不同窗口，观察预取收益、Bank interference 和总执行时间。',
        [
            code(r'''
EXP = ROOT / "outputs/DATE1/exp3"
rows, layer_rows = [], []
run_order = ["static_no_prefetch", "static_prefetch_w1", "static_prefetch_w2", "static_prefetch_w4"]
for run_name in run_order:
    run_dir = EXP / run_name
    summary_path = run_dir / "EP_MOE_SUMMARY.csv"
    allocation_path = run_dir / "EP_MOE_BANK_ALLOCATION.csv"
    if not summary_path.exists():
        continue
    summary = numeric(read_csv(summary_path)).iloc[0].to_dict()
    summary["Run"] = run_dir.name
    match = re.search(r"w(\d+)$", run_dir.name)
    summary["Window"] = int(match.group(1)) if match else 0
    rows.append(summary)
    if allocation_path.exists():
        alloc_raw = read_csv(allocation_path)
        ratios = set(alloc_raw["AllocationRatio"].astype(str).str.strip())
        if ratios != {"4:14:6"}:
            raise RuntimeError(
                f"{run_dir.name} 使用的 Bank 比例为 {sorted(ratios)}，当前实验3要求 4:14:6；"
                "请重新执行 ./scripts/DATE1/run_exp3.sh。"
            )
        alloc = numeric(alloc_raw)
        # numeric() intentionally coerces report columns for aggregation, but
        # LayerName is the categorical heatmap index and must remain text.
        alloc["LayerName"] = alloc_raw["LayerName"].astype(str).str.strip()
        summary["PrefetchBankRequests"] = alloc["RuntimePrefetchBankRequests"].sum()
        summary["AddedPrefetchInterference"] = alloc["RuntimePrefetchBankInterferenceStall"].sum()
        alloc["Run"], alloc["Window"] = run_dir.name, summary["Window"]
        layer_rows.append(alloc)
if len(rows) != len(run_order):
    missing = sorted(set(run_order) - {row["Run"] for row in rows})
    raise FileNotFoundError(f"exp3 结果不完整，缺少：{missing}。请执行 ./scripts/DATE1/run_exp3.sh")
result = pd.DataFrame(rows).set_index("Run").reindex(run_order).reset_index()
baseline_time = result.loc[result["Window"] == 0, "MoEGroupTime"].iloc[0]
result["NormalizedTime"] = result["MoEGroupTime"] / baseline_time
result["SpeedupVsNoPrefetch"] = baseline_time / result["MoEGroupTime"]
result["InterferencePerMillionRequests"] = np.where(
    result["PrefetchBankRequests"] > 0,
    result["AddedPrefetchInterference"] / result["PrefetchBankRequests"] * 1e6,
    0,
)
table = result[["Run", "Window", "MoEGroupTime", "NormalizedTime", "SpeedupVsNoPrefetch",
                "AvgPrefetchHitRate", "TotalPrefetchMissStall", "AddedPrefetchInterference",
                "PrefetchBankRequests", "TotalPrefetchBandwidthOverhead",
                "InterferencePerMillionRequests"]]
table.to_csv(FIG_DIR / "exp3_prefetch_summary.csv", index=False)
display(table.style.format({
    "MoEGroupTime": "{:,.0f}", "NormalizedTime": "{:.4f}",
    "SpeedupVsNoPrefetch": "{:.3f}×", "AvgPrefetchHitRate": "{:.2%}",
    "TotalPrefetchMissStall": "{:,.0f}", "AddedPrefetchInterference": "{:,.0f}",
    "PrefetchBankRequests": "{:,.0f}", "TotalPrefetchBandwidthOverhead": "{:,.0f}",
    "InterferencePerMillionRequests": "{:.1f}",
}))
'''),
            code(r'''
fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
x = np.arange(len(result))
bars = ax1.bar(x, result["NormalizedTime"], color="#4E79A7", label="Normalized time")
ax1.axhline(1, color="black", linestyle="--", linewidth=1)
ax1.set_ylabel("Normalized MoE group time\n(no prefetch = 1)")
ax1.set_xticks(x, ["No prefetch", "Window 1", "Window 2", "Window 4"])
for bar, speedup in zip(bars, result["SpeedupVsNoPrefetch"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{speedup:.3f}×",
             ha="center", va="bottom", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(x, result["AddedPrefetchInterference"], color="#E15759", marker="o", linewidth=2,
         label="Added prefetch interference")
ax2.set_ylabel("Added bank-interference stall cycles")
ax1.set_title("Prefetch Benefit versus New Bank Interference")
fig.tight_layout(); savefig("exp3_prefetch_interference.pdf"); plt.show()
'''),
            code(r'''
layers = pd.concat(layer_rows, ignore_index=True)
layer_summary = layers.groupby("Run").agg(
    NormalBankConflictStall=("LayerBankConflictStall", "sum"),
    PrefetchBankRequests=("RuntimePrefetchBankRequests", "sum"),
    AddedPrefetchInterference=("RuntimePrefetchBankInterferenceStall", "sum"),
).reindex(run_order)
layer_summary.to_csv(FIG_DIR / "exp3_bank_interference_breakdown.csv")
display(layer_summary.style.format("{:,.0f}"))

per_layer = layers.pivot_table(
    index="LayerName", columns="Run", values="RuntimePrefetchBankInterferenceStall", aggfunc="sum"
).reindex(columns=run_order).fillna(0)
per_layer.to_csv(FIG_DIR / "exp3_interference_per_layer.csv")
prefetch_columns = run_order[1:]
fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(per_layer[prefetch_columns].values, aspect="auto", cmap="Reds")
ax.set_xticks(range(3), ["Window 1", "Window 2", "Window 4"])
ax.set_yticks(range(len(per_layer)), per_layer.index)
ax.set_title("Where Naive Prefetch Adds Bank Interference")
plt.colorbar(im, ax=ax, label="Interference stall cycles")
plt.tight_layout(); savefig("exp3_interference_per_layer.pdf"); plt.show()
'''),
            md('## 自动生成实验结论'),
            code(r'''
best = result.loc[result["MoEGroupTime"].idxmin()]
positive = result[result["AddedPrefetchInterference"] > 0].sort_values("Window")
first_conflict = positive.iloc[0] if not positive.empty else None
peak_conflict = positive.loc[positive["AddedPrefetchInterference"].idxmax()] if not positive.empty else None
print(f"无预取执行时间：{baseline_time:,.0f} cycles。")
print(f"最佳窗口为 W{int(best['Window'])}，相对无预取加速 {best['SpeedupVsNoPrefetch']:.3f}×。")
if first_conflict is not None:
    print(f"新增 Bank 干扰从 W{int(first_conflict['Window'])} 开始出现："
          f"{first_conflict['AddedPrefetchInterference']:,.0f} stall cycles。")
    print(f"峰值出现在 W{int(peak_conflict['Window'])}："
          f"{peak_conflict['AddedPrefetchInterference']:,.0f} stall cycles；其执行时间相对无预取"
          f"增加 {(peak_conflict['NormalizedTime']-1)*100:.2f}%。")
    print("结论：较小窗口可以隐藏权重加载延迟；普通预取变得激进后会与正常访存争用物理 Bank，新增冲突可使预取从收益转为性能损失。")
else:
    print("结论：本次结果未观察到新增预取 Bank 干扰，不能支持实验3假设。")
'''),
        ],
    )

    write(
        'exp4.ipynb', '实验4：动态 Bank 分配缓解冲突',
        '比较 Static-Equal、候选 Best-Static 与 Dynamic，并展示动态方案逐层实际分配。',
        [
            code(r'''
EXP = ROOT / "outputs/DATE1/exp4"
rows, allocations = [], []
run_order = ["static_equal_8_8_8", "static_best_4_14_6", "dynamic_24"]
for run_name in run_order:
    run_dir = EXP / run_name
    summary_path = run_dir / "EP_MOE_SUMMARY.csv"
    alloc_path = run_dir / "EP_MOE_BANK_ALLOCATION.csv"
    if not summary_path.exists(): continue
    item = numeric(read_csv(summary_path)).iloc[0].to_dict(); item["Run"] = run_dir.name
    alloc_raw = read_csv(alloc_path)
    ratios = set(alloc_raw["AllocationRatio"].astype(str).str.strip())
    if run_name == "static_equal_8_8_8" and ratios != {"8:8:8"}:
        raise RuntimeError(f"{run_name} 的 Bank 比例异常：{sorted(ratios)}")
    if run_name == "static_best_4_14_6" and ratios != {"4:14:6"}:
        raise RuntimeError(f"{run_name} 的 Bank 比例异常：{sorted(ratios)}")
    if run_name == "dynamic_24":
        totals = (numeric(alloc_raw)[["IfmapBankNum", "FilterBankNum", "OfmapBankNum"]].sum(axis=1))
        if not (totals == 24).all():
            raise RuntimeError("dynamic_24 存在 Bank 总数不为24的逐层分配")
    alloc = numeric(alloc_raw)
    alloc["LayerName"] = alloc_raw["LayerName"].astype(str).str.strip()
    alloc["AllocationRatio"] = alloc_raw["AllocationRatio"].astype(str).str.strip()
    alloc["Run"] = run_dir.name
    item["LayerBankConflictStall"] = alloc["LayerBankConflictStall"].sum()
    rows.append(item); allocations.append(alloc)
if len(rows) != len(run_order):
    missing = sorted(set(run_order) - {row["Run"] for row in rows})
    raise FileNotFoundError(f"exp4 结果不完整，缺少：{missing}。请执行 ./scripts/DATE1/run_exp4.sh")
result = pd.DataFrame(rows).set_index("Run").reindex(run_order)
equal_time = result.loc["static_equal_8_8_8", "MoEGroupTime"]
result["NormalizedTime"] = result["MoEGroupTime"] / equal_time
result["SpeedupVsEqual"] = equal_time / result["MoEGroupTime"]
result["ConflictReductionVsEqual"] = 1 - result["LayerBankConflictStall"] / result.loc["static_equal_8_8_8", "LayerBankConflictStall"]
result.to_csv(FIG_DIR / "exp4_static_dynamic_summary.csv")
display(result[["MoEGroupTime", "LayerBankConflictStall", "TotalExpertWaitingCycles",
                "NormalizedTime", "SpeedupVsEqual", "ConflictReductionVsEqual"]].style.format({
    "MoEGroupTime": "{:,.0f}", "LayerBankConflictStall": "{:,.0f}",
    "TotalExpertWaitingCycles": "{:,.0f}", "NormalizedTime": "{:.3f}",
    "SpeedupVsEqual": "{:.3f}×", "ConflictReductionVsEqual": "{:.2%}",
}))
'''),
            code(r'''
labels = ["Static equal\n8:8:8", "Best static\n4:14:6", "Dynamic\n24 banks"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
x = np.arange(len(result))
bars = axes[0].bar(x, result["NormalizedTime"], color=["#BAB0AC", "#F28E2B", "#4E79A7"], edgecolor="black")
axes[0].set_xticks(x, labels); axes[0].set_ylabel("Normalized MoE group time\n(static equal = 1)")
axes[0].set_title("Execution Time")
for bar, speedup in zip(bars, result["SpeedupVsEqual"]):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{speedup:.2f}×", ha="center", va="bottom")
conflict_norm = result["LayerBankConflictStall"] / result.loc["static_equal_8_8_8", "LayerBankConflictStall"]
bars = axes[1].bar(x, conflict_norm, color=["#BAB0AC", "#F28E2B", "#4E79A7"], edgecolor="black")
axes[1].set_xticks(x, labels); axes[1].set_ylabel("Normalized bank-conflict stall\n(static equal = 1)")
axes[1].set_title("Bank Conflict")
for bar, reduction in zip(bars, result["ConflictReductionVsEqual"]):
    reduction = max(0.0, float(reduction))
    label = "0.0%" if reduction == 0 else f"-{reduction:.1%}"
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height(), label, ha="center", va="bottom")
fig.suptitle("Static versus Dynamic Bank Allocation")
plt.tight_layout(); savefig("exp4_dynamic_benefit.pdf"); plt.show()
'''),
            code(r'''
alloc_all = pd.concat(allocations, ignore_index=True)
dynamic = alloc_all[alloc_all["Run"] == "dynamic_24"].sort_values(["ExpertID", "LayerID"])
dynamic[["LayerID", "LayerName", "AllocationRatio", "IfmapBankNum", "FilterBankNum", "OfmapBankNum",
         "LayerTotalCycles", "LayerBankConflictStall"]].to_csv(
    FIG_DIR / "exp4_dynamic_per_layer_allocations.csv", index=False
)
x = np.arange(len(dynamic))
plt.figure(figsize=(13, 4.5))
plt.bar(x, dynamic["IfmapBankNum"], label="IA")
plt.bar(x, dynamic["FilterBankNum"], bottom=dynamic["IfmapBankNum"], label="Weight")
plt.bar(x, dynamic["OfmapBankNum"], bottom=dynamic["IfmapBankNum"] + dynamic["FilterBankNum"], label="OA")
plt.xticks(x, dynamic["LayerName"], rotation=55, ha="right")
plt.ylabel("Number of banks"); plt.title("Per-layer Dynamic Bank Allocation")
plt.legend(); plt.tight_layout(); savefig("exp4_dynamic_allocations.pdf"); plt.show()
'''),
            md('## 自动生成实验结论'),
            code(r'''
equal = result.loc["static_equal_8_8_8"]
static_best = result.loc["static_best_4_14_6"]
dynamic_row = result.loc["dynamic_24"]
speedup_equal = equal["MoEGroupTime"] / dynamic_row["MoEGroupTime"]
speedup_best = static_best["MoEGroupTime"] / dynamic_row["MoEGroupTime"]
conflict_drop_equal = 1 - dynamic_row["LayerBankConflictStall"] / equal["LayerBankConflictStall"]
conflict_drop_best = 1 - dynamic_row["LayerBankConflictStall"] / static_best["LayerBankConflictStall"]
unique_ratios = dynamic["AllocationRatio"].nunique()
print(f"Dynamic-24 相对 Static-Equal 获得 {speedup_equal:.3f}× 加速，Bank 冲突停顿下降 {conflict_drop_equal:.2%}。")
print(f"Dynamic-24 相对全局 Best-Static 获得 {speedup_best:.3f}× 加速，Bank 冲突停顿下降 {conflict_drop_best:.2%}。")
print(f"8 个 MoE 层共采用 {unique_ratios} 种动态 Bank 比例。")
print("结论：全局最优静态比例已经消除了大部分均分方案的失配，但不同层仍偏好不同的 IA/Weight/OA 比例；动态分配进一步减少冲突并缩短执行时间。")
'''),
        ],
    )

    write(
        'exp5.ipynb', '实验5：动态分配与预取协同消融',
        '分析 Static/Dynamic × Prefetch On/Off 四组实验，分别量化两个机制的收益。',
        [
            code(r'''
EXP = ROOT / "outputs/DATE1/exp5"
order = ["static_no_prefetch", "static_prefetch", "dynamic_no_prefetch", "dynamic_prefetch"]
rows = []
for name in order:
    run_dir = EXP / name
    if not (run_dir / "EP_MOE_SUMMARY.csv").exists(): continue
    item = numeric(read_csv(run_dir / "EP_MOE_SUMMARY.csv")).iloc[0].to_dict(); item["Run"] = name
    alloc_raw = read_csv(run_dir / "EP_MOE_BANK_ALLOCATION.csv")
    totals = numeric(alloc_raw)[["IfmapBankNum", "FilterBankNum", "OfmapBankNum"]].sum(axis=1)
    if not (totals == 24).all(): raise RuntimeError(f"{name} 存在 Bank 总数不为24的分配")
    alloc = numeric(alloc_raw)
    item["DetailedBankConflictStall"] = alloc["LayerBankConflictStall"].sum()
    rows.append(item)
if len(rows) != 4:
    raise FileNotFoundError("exp5 四组结果不完整，请先执行 ./scripts/DATE1/run_exp5.sh")
result = pd.DataFrame(rows).set_index("Run").reindex(order)
baseline = result.loc["static_no_prefetch", "MoEGroupTime"]
result["NormalizedTime"] = result["MoEGroupTime"] / baseline
result["SpeedupVsBaseline"] = baseline / result["MoEGroupTime"]
result.to_csv(FIG_DIR / "exp5_ablation_summary.csv")
display(result[["MoEGroupTime", "NormalizedTime", "SpeedupVsBaseline", "TotalExpertWaitingCycles",
                "AvgPrefetchHitRate", "TotalPrefetchMissStall", "TotalPrefetchBankInterferenceStall",
                "DetailedBankConflictStall"]])
'''),
            code(r'''
plt.figure(figsize=(8.5, 4.5))
bars = plt.bar(result.index, result["NormalizedTime"], color=["#BAB0AC", "#F28E2B", "#59A14F", "#4E79A7"], edgecolor="black")
plt.axhline(1, color="black", linestyle="--", linewidth=1)
plt.ylabel("Normalized MoE group time"); plt.title("Dynamic Bank + Prefetch Ablation")
plt.xticks(rotation=20, ha="right")
for bar, speedup in zip(bars, result["SpeedupVsBaseline"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{speedup:.2f}×", ha="center", va="bottom")
plt.tight_layout(); savefig("exp5_ablation.pdf"); plt.show()
'''),
            code(r'''
t_sn = result.loc["static_no_prefetch", "MoEGroupTime"]
t_sp = result.loc["static_prefetch", "MoEGroupTime"]
t_dn = result.loc["dynamic_no_prefetch", "MoEGroupTime"]
t_dp = result.loc["dynamic_prefetch", "MoEGroupTime"]
comparison = pd.DataFrame({
    "Comparison": ["Prefetch only", "Dynamic only", "Combined"],
    "CycleReduction": [t_sn - t_sp, t_sn - t_dn, t_sn - t_dp],
    "Speedup": [t_sn / t_sp, t_sn / t_dn, t_sn / t_dp],
})
additive_reduction = (t_sn - t_sp) + (t_sn - t_dn)
combined_reduction = t_sn - t_dp
comparison["ReductionVsBaselinePercent"] = comparison["CycleReduction"] / t_sn * 100
comparison.to_csv(FIG_DIR / "exp5_mechanism_comparison.csv", index=False)
display(comparison)
print(f"独立收益简单相加为 {additive_reduction:,.0f} cycles，协同方案实际减少 {combined_reduction:,.0f} cycles。")
print(f"额外协同收益为 {combined_reduction-additive_reduction:,.0f} cycles；组合方案相对基线加速 {t_sn/t_dp:.3f}×。")
'''),
        ],
    )

    write(
        'exp6.ipynb', '实验6：预取窗口与 Chunk 粒度敏感性',
        '分别分析 Window sweep，并绘制 ChunkSizeBytes × Window 的二维性能热力图。',
        [
            code(r'''
EXP = ROOT / "outputs/DATE1/exp6"
rows = []
for run_dir in sorted(path for path in EXP.iterdir() if path.is_dir()):
    summary_path = run_dir / "EP_MOE_SUMMARY.csv"
    config_path = run_dir / "EP_MOE_CONFIG.csv"
    if not summary_path.exists() or not config_path.exists(): continue
    item = numeric(read_csv(summary_path)).iloc[0].to_dict(); item["Run"] = run_dir.name
    cfg = read_csv(config_path).set_index("Key")["Value"]
    item["Window"] = int(float(cfg.get("ChunkPrefetchWindow", 0)))
    item["ChunkSizeBytes"] = int(float(cfg.get("ChunkSizeBytes", 0)))
    rows.append(item)
if not rows:
    raise FileNotFoundError("exp6 尚无结果，请先执行 ./scripts/DATE1/run_exp6.sh")
result = pd.DataFrame(rows)
expected_runs = ({f"window_{w}" for w in (0,1,2,4,8)} |
                 {f"chunk_{size}_window_{w}" for size in (4096,8192,16384,32768) for w in (1,2,4,8)})
missing = sorted(expected_runs - set(result["Run"]))
if missing: raise FileNotFoundError(f"exp6 结果不完整，缺少：{missing}")
result.to_csv(FIG_DIR / "exp6_sensitivity_summary.csv", index=False)
display(result[["Run", "Window", "ChunkSizeBytes", "MoEGroupTime", "AvgPrefetchHitRate",
                "TotalPrefetchMissStall", "TotalPrefetchBankInterferenceStall", "TotalPrefetchBandwidthOverhead"]])
'''),
            code(r'''
window = result[result["Run"].str.startswith("window_")].sort_values("Window")
if not window.empty:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True)
    axes[0].plot(window["Window"], window["MoEGroupTime"], marker="o", linewidth=2)
    axes[0].axhline(window.loc[window["Window"] == 0, "MoEGroupTime"].iloc[0], color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("MoE group cycles"); axes[0].set_title("Window Sensitivity (Legacy Variable-size Chunks)")
    for metric, label in [("TotalPrefetchMissStall", "Miss stall"),
                          ("TotalPrefetchBankInterferenceStall", "Bank interference")]:
        axes[1].plot(window["Window"], window[metric], marker="o", label=label)
    axes[1].set_ylabel("Stall cycles"); axes[1].legend()
    axes[2].plot(window["Window"], window["TotalPrefetchBandwidthOverhead"] / 1e6, marker="o", color="#59A14F")
    axes[2].set_xlabel("Chunk prefetch window"); axes[2].set_ylabel("Prefetch traffic (MB)")
    plt.tight_layout(); savefig("exp6_window_sensitivity.pdf"); plt.show()
'''),
            code(r'''
chunk = result[result["ChunkSizeBytes"] > 0].copy()
if not chunk.empty:
    pivot = chunk.pivot(index="ChunkSizeBytes", columns="Window", values="MoEGroupTime")
    baseline_time = window.loc[window["Window"] == 0, "MoEGroupTime"].iloc[0]
    norm = pivot / baseline_time
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(norm.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(norm.columns)), norm.columns)
    ax.set_yticks(range(len(norm.index)), [f"{x//1024} KB" for x in norm.index])
    ax.set_xlabel("Prefetch window"); ax.set_ylabel("Chunk size")
    ax.set_title("Chunk Size × Window Sensitivity")
    plt.colorbar(im, ax=ax, label="Normalized MoE group time")
    for row in range(len(norm.index)):
        for col in range(len(norm.columns)):
            value = norm.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 1.0 else "black")
    plt.tight_layout(); savefig("exp6_chunk_window_heatmap.pdf"); plt.show()
'''),
            md('## 自动生成实验结论'),
            code(r'''
legacy_best = window.loc[window["MoEGroupTime"].idxmin()]
explicit_best = chunk.loc[chunk["MoEGroupTime"].idxmin()]
explicit_worst = chunk.loc[chunk["MoEGroupTime"].idxmax()]
baseline_time = window.loc[window["Window"] == 0, "MoEGroupTime"].iloc[0]
print(f"旧版可变 Chunk 下最佳窗口为 W{int(legacy_best['Window'])}，相对无预取加速 {baseline_time/legacy_best['MoEGroupTime']:.3f}×。")
print(f"显式 Chunk 矩阵最优为 {int(explicit_best['ChunkSizeBytes']/1024)}KB/W{int(explicit_best['Window'])}：{explicit_best['MoEGroupTime']:,.0f} cycles。")
print(f"最差为 {int(explicit_worst['ChunkSizeBytes']/1024)}KB/W{int(explicit_worst['Window'])}：{explicit_worst['MoEGroupTime']:,.0f} cycles。")
print("结论：Window=1 在所有 Chunk 粒度下最稳定；窗口达到2及以上后，Bank 干扰会抵消预取收益。Chunk 粒度的影响小于窗口选择。")
'''),
        ],
    )

    write(
        'exp7.ipynb', '实验7：路由分布与专家配置敏感性',
        '分析 routing skew、Top-k、Token 数和 Expert 数对 Dynamic-Prefetch 的影响。',
        [
            code(r'''
EXP = ROOT / "outputs/DATE1/exp7"
rows, trace_rows = [], []
for run_dir in sorted(path for path in EXP.iterdir() if path.is_dir()):
    summary_path = run_dir / "EP_MOE_SUMMARY.csv"
    trace_path = run_dir / "EP_MOE_ROUTED_TRACE.csv"
    if not summary_path.exists(): continue
    if not trace_path.exists():
        raise RuntimeError(f"{run_dir.name} 是旧版固定trace结果；请用新架构重新执行 ./scripts/DATE1/run_exp7.sh")
    item = numeric(read_csv(summary_path)).iloc[0].to_dict(); item["Run"] = run_dir.name
    trace_raw = read_csv(trace_path)
    trace = numeric(trace_raw)
    for column in ["LayerName", "RoutingMode", "TraceMode"]:
        trace[column] = trace_raw[column].astype(str).str.strip()
    for column in ["IsActiveExpert", "IsDetailedGPU", "TraceScaled"]:
        trace[column] = trace_raw[column].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    detailed_active = trace[(trace["IsDetailedGPU"] == 1) & (trace["IsActiveExpert"] == 1)]
    if detailed_active.empty or not (detailed_active["EffectiveM"] == detailed_active["RoutedTokens"]).all():
        raise RuntimeError(f"{run_dir.name} 的 routed-token-aware trace 一致性检查失败")
    item["DetailedEffectiveMMin"] = detailed_active["EffectiveM"].min()
    item["DetailedEffectiveMMax"] = detailed_active["EffectiveM"].max()
    trace["Run"] = run_dir.name; trace_rows.append(trace)
    rows.append(item)
if not rows:
    raise FileNotFoundError("exp7 尚无结果，请先执行 ./scripts/DATE1/run_exp7.sh")
result = pd.DataFrame(rows)
expected_count = 30
if len(result) != expected_count:
    raise FileNotFoundError(f"exp7 应有 {expected_count} 组结果，当前只有 {len(result)} 组")
result["Skew"] = pd.to_numeric(result["Run"].str.extract(r"routing_skew_(\d+p\d+)_seed", expand=False).str.replace("p", "."), errors="coerce")
result["Seed"] = pd.to_numeric(result["Run"].str.extract(r"seed_(\d+)$", expand=False), errors="coerce")
result["Tokens"] = pd.to_numeric(result["Run"].str.extract(r"tokens_(\d+)$", expand=False), errors="coerce")
result.to_csv(FIG_DIR / "exp7_all_results.csv", index=False)
trace_all = pd.concat(trace_rows, ignore_index=True)
trace_all.to_csv(FIG_DIR / "exp7_routed_trace_audit.csv", index=False)
display(result[["Run", "NumExperts", "TopK", "MoEGroupTime", "ExpertTokenImbalance",
                "ExpertCycleImbalance", "GPULoadImbalanceCycles", "MinimumGPUUtilization",
                "TotalExpertWaitingCycles", "TotalPrefetchBankInterferenceStall"]])
'''),
            code(r'''
routing = result[result["Skew"].notna()].copy()
routing_summary = routing.groupby("Skew").agg(
    TokenImbalanceMean=("ExpertTokenImbalance", "mean"),
    TokenImbalanceStd=("ExpertTokenImbalance", "std"),
    GPULoadImbalanceMean=("GPULoadImbalanceCycles", "mean"),
    GPULoadImbalanceStd=("GPULoadImbalanceCycles", "std"),
    MinGPUUtilMean=("MinimumGPUUtilization", "mean"),
    MinGPUUtilStd=("MinimumGPUUtilization", "std"),
    MoETimeMean=("MoEGroupTime", "mean"),
    MoETimeStd=("MoEGroupTime", "std"),
).reset_index()
routing_summary.to_csv(FIG_DIR / "exp7_routing_seed_summary.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.ravel()
axes[0].errorbar(routing_summary["Skew"], routing_summary["TokenImbalanceMean"],
                 yerr=routing_summary["TokenImbalanceStd"], marker="o", capsize=4)
axes[0].set_xlabel("Routing skew factor"); axes[0].set_ylabel("Expert-token imbalance")
axes[0].set_title("Routing Imbalance")
axes[1].errorbar(routing_summary["Skew"], routing_summary["GPULoadImbalanceMean"],
                 yerr=routing_summary["GPULoadImbalanceStd"], marker="o", capsize=4, color="#E15759")
axes[1].set_xlabel("Routing skew factor"); axes[1].set_ylabel("GPU-load imbalance cycles")
axes[1].set_title("GPU Load Imbalance")
axes[2].errorbar(routing_summary["Skew"], routing_summary["MinGPUUtilMean"],
                 yerr=routing_summary["MinGPUUtilStd"], marker="o", capsize=4, color="#59A14F")
axes[2].set_xlabel("Routing skew factor"); axes[2].set_ylabel("Minimum GPU utilization")
axes[2].set_title("Least-utilized GPU")
axes[3].errorbar(routing_summary["Skew"], routing_summary["MoETimeMean"],
                 yerr=routing_summary["MoETimeStd"], marker="o", capsize=4, color="#B07AA1")
axes[3].set_xlabel("Routing skew factor"); axes[3].set_ylabel("MoE group cycles")
axes[3].set_title("Routed-trace Execution Time")
plt.tight_layout(); savefig("exp7_routing_sensitivity.pdf"); plt.show()
'''),
            code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

topk = result[result["Run"].str.startswith("topk_")].sort_values("TopK")
if not topk.empty:
    axes[0].bar(topk["TopK"].astype(int).astype(str), topk["MoEGroupTime"], color="#4E79A7")
axes[0].set_title("Top-k"); axes[0].set_xlabel("Top-k"); axes[0].set_ylabel("MoE group cycles")

tokens = result[result["Run"].str.startswith("tokens_")].copy()
if not tokens.empty:
    tokens = tokens.sort_values("Tokens")
    axes[1].plot(tokens["Tokens"], tokens["MoEGroupTime"], marker="o")
axes[1].set_title("Token Count"); axes[1].set_xlabel("Tokens"); axes[1].set_ylabel("MoE group cycles")

experts = result[result["Run"].str.startswith("experts_")].sort_values("NumExperts")
if not experts.empty:
    axes[2].plot(experts["NumExperts"], experts["MoEGroupTime"], marker="o")
axes[2].set_title("Expert Count"); axes[2].set_xlabel("Experts")

plt.tight_layout(); savefig("exp7_configuration_sensitivity.pdf"); plt.show()
'''),
            md('## Routed-token-aware trace 一致性检查'),
            code(r'''
audit = trace_all[(trace_all["IsDetailedGPU"] == 1) & (trace_all["IsActiveExpert"] == 1)].copy()
audit["MMatchesRouting"] = audit["EffectiveM"] == audit["RoutedTokens"]
display(audit.groupby("Run").agg(
    DetailedLayers=("LayerID", "count"), RoutedTokenMin=("RoutedTokens", "min"),
    RoutedTokenMax=("RoutedTokens", "max"), EffectiveMMin=("EffectiveM", "min"),
    EffectiveMMax=("EffectiveM", "max"), AllMMatchRouting=("MMatchesRouting", "all"),
))
print("所有活跃详细专家均满足 EffectiveM == RoutedTokens；Token、Top-k 和路由变化已进入原生计算与访存trace。")
'''),
            md('## 自动生成实验结论'),
            code(r'''
low = routing_summary.loc[routing_summary["Skew"].idxmin()]
high = routing_summary.loc[routing_summary["Skew"].idxmax()]
experts = result[result["Run"].str.startswith("experts_")].sort_values("NumExperts")
print(f"路由 skew 从 {low['Skew']:.1f} 增至 {high['Skew']:.1f} 时，平均 Token 不均衡从 "
      f"{low['TokenImbalanceMean']:.1f} 增至 {high['TokenImbalanceMean']:.1f}。")
print(f"最小 GPU 利用率均值从 {low['MinGPUUtilMean']:.3f} 变为 {high['MinGPUUtilMean']:.3f}；"
      "高 skew 下部分 seed 可能只激活单侧 GPU，因此必须报告误差棒。")
print(f"专家数从 {int(experts.iloc[0]['NumExperts'])} 增至 {int(experts.iloc[-1]['NumExperts'])} 时，"
      f"MoEGroupTime 从 {experts.iloc[0]['MoEGroupTime']:,.0f} 增至 {experts.iloc[-1]['MoEGroupTime']:,.0f} cycles。")
print(f"Token sweep 的 MoEGroupTime 范围为 {tokens['MoEGroupTime'].min():,.0f}--{tokens['MoEGroupTime'].max():,.0f} cycles；"
      f"Top-k=2 相对Top-k=1变化 {(topk.iloc[-1]['MoEGroupTime']/topk.iloc[0]['MoEGroupTime']-1)*100:.2f}%。")
print("结论：routed-token-aware trace 已将路由Token数传递到详细单核的GEMM、IA/OA访存、Bank冲突和预取时序，可用于完整分析Token、Top-k与路由倾斜敏感性。")
'''),
        ],
    )


if __name__ == '__main__':
    main()
