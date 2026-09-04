"""Retarget and strengthen the public Exp1/Exp2 notebooks for DATE3."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG = "date3_contract_analysis"


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def markdown(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [TAG]},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


EXP1_CODE = r"""
contract = df.groupby('category').agg(
    layers=('layer', 'count'),
    compute_cycles=('compute_cycles', 'sum'),
    memory_stall_cycles=('memory_stall_cycles', 'sum'),
    mean_stall_ratio=('stall_ratio', 'mean'),
    mean_memory_to_compute=('memory_to_compute_ratio', 'mean'),
).reindex(['Non-MoE', 'MoE Expert'])
contract['stall_share'] = contract.memory_stall_cycles / contract.memory_stall_cycles.sum()
moe_rows = df[df.category.eq('MoE Expert')]
unique_ratios = len(moe_rows[['ideal_ia_banks', 'ideal_weight_banks', 'ideal_oa_banks']].drop_duplicates())
moe_acc_stall_share = moe_rows.accumulator_stall_cycles.sum() / moe_rows.memory_stall_cycles.sum()
moe_weight_stall_share = moe_rows.weight_stall_cycles.sum() / moe_rows.memory_stall_cycles.sum()
checks = pd.Series({
    'MoE contributes >50% aggregate exposed stall': contract.loc['MoE Expert', 'stall_share'] > 0.5,
    'MoE normalized stall ratio exceeds Non-MoE': contract.loc['MoE Expert', 'mean_stall_ratio'] > contract.loc['Non-MoE', 'mean_stall_ratio'],
    'Every MoE expert stage is memory-bound': (moe_rows.memory_to_compute_ratio > 1).all(),
    'MoE IA:Weight:OA demand is stage-dependent': unique_ratios > 1,
})
display(contract.round(4))
display(checks.rename('pass').to_frame())
print(f"MoE aggregate stall share: {contract.loc['MoE Expert', 'stall_share']:.1%}")
print(f"MoE / Non-MoE mean stall ratio: {contract.loc['MoE Expert', 'mean_stall_ratio']:.1%} / {contract.loc['Non-MoE', 'mean_stall_ratio']:.1%}")
print(f"Memory-bound MoE stages: {(moe_rows.memory_to_compute_ratio > 1).sum()}/{len(moe_rows)}")
print(f"Distinct MoE ideal IA:Weight:OA ratios: {unique_ratios}")
print(f"MoE ACC overwrite/RMW share of exposed stall: {moe_acc_stall_share:.1%}")
print(f"MoE Weight-only share of exposed stall: {moe_weight_stall_share:.1%}")
assert checks.all(), checks[~checks]
"""

EXP1_MD = r"""
## DATE3 实验 1 论文判断

- **访存瓶颈成立**：所有 MoE FF1/FF2 阶段的 exposed memory stall 均超过其 compute cycles；MoE 的平均 stall ratio 和 memory/compute ratio 也高于非 MoE。
- **MoE 是主要优化对象，但不是唯一瓶颈**：MoE 汇总后贡献超过一半的总 stall；与此同时 Attention/Router 也有明显访存停顿，不能声称“每个 MoE 层的绝对 stall 都高于每个 Attention 层”。
- **固定资源边界失配成立**：专家阶段出现多种不同的理想 IA:Weight:OA Bank 比例，并且 FF1/FF2 的方向会改变，支持统一资源池和动态映射的动机。
- **主要停顿来源是 ACC 路径**：当前结果中的绝大多数 exposed stall 来自 INT32 ACC overwrite/RMW；这直接支持统一 SP/ACC 资源路径，但不能由 Exp1 单独推出 Weight Prefetch 会获得同等幅度收益。Weight Prefetch 的必要性和协同收益应分别由 Exp3、Exp5 验证。
- **论文可用表述**：在当前受控完整 Block 中，MoE 专家阶段表现出更强的归一化访存受限特征，并贡献主要的汇总访存停顿，因此具有显著片上访存优化空间。

该实验是动机与瓶颈定位实验，不负责证明最终 MemDomain/PIVOT 的端到端加速比；性能收益应由 Exp4、Exp5 和 Exp7 给出。
"""

EXP2_CODE = r"""
global_table = sweep.groupby(['ia_banks', 'weight_banks', 'oa_banks'], as_index=False).agg(
    total_cycles=('total_cycles', 'sum'),
    memory_stall_cycles=('memory_stall_cycles', 'sum'),
)
global_best_row = global_table.loc[global_table.total_cycles.idxmin()]
fixed_555_row = global_table.query('ia_banks == 5 and weight_banks == 5 and oa_banks == 5').iloc[0]
stage_best_total = best.total_cycles.sum()
joined = sweep.merge(best[['layer', 'total_cycles']], on='layer', suffixes=('', '_stage_best'))
joined['slowdown_vs_stage_best'] = joined.total_cycles / joined.total_cycles_stage_best
mask = (
    joined.ia_banks.eq(int(global_best_row.ia_banks))
    & joined.weight_banks.eq(int(global_best_row.weight_banks))
    & joined.oa_banks.eq(int(global_best_row.oa_banks))
)
global_per_stage = joined[mask]
metrics = pd.Series({
    'stage_count': best.layer.nunique(),
    'partition_count_per_stage': int(sweep.groupby('layer').size().iloc[0]),
    'distinct_stage-optimal_ratios': len(best[['ia_banks', 'weight_banks', 'oa_banks']].drop_duplicates()),
    'global_best_ia': int(global_best_row.ia_banks),
    'global_best_weight': int(global_best_row.weight_banks),
    'global_best_oa': int(global_best_row.oa_banks),
    'global_best_optimal_stage_count': int(np.isclose(global_per_stage.slowdown_vs_stage_best, 1.0).sum()),
    'global_best_gap_to_stagewise_lower_bound_pct': 100 * (global_best_row.total_cycles / stage_best_total - 1),
    'fixed_555_gap_to_stagewise_lower_bound_pct': 100 * (fixed_555_row.total_cycles / stage_best_total - 1),
    'global_best_worst_stage_slowdown_pct': 100 * (global_per_stage.slowdown_vs_stage_best.max() - 1),
})
display(metrics.rename('value').to_frame().round(4))
print(
    f"Global fixed optimum = {int(global_best_row.ia_banks)}:{int(global_best_row.weight_banks)}:{int(global_best_row.oa_banks)}; "
    f"optimal for {int(metrics['global_best_optimal_stage_count'])}/{int(metrics['stage_count'])} stages."
)
assert metrics['partition_count_per_stage'] == 91
assert metrics['distinct_stage-optimal_ratios'] > 1
assert metrics['global_best_gap_to_stagewise_lower_bound_pct'] > 0
assert metrics['global_best_worst_stage_slowdown_pct'] > 0
"""

EXP2_MD = r"""
## DATE3 实验 2 论文判断

- **静态最优不具备全阶段普适性**：23 个阶段出现多种最优 IA:Weight:OA 比例；一个全局固定最优比例只覆盖部分阶段，其余阶段存在不同程度的退化。
- **动态搜索空间的理论动机成立**：逐阶段选择静态最优形成严格更低的下界，说明包含所有静态候选的动态映射仍有优化空间。
- **收益强度需要诚实描述**：当前受控 MoDSE Block 中，全局固定最优与逐阶段下界的汇总差距较温和，但最坏阶段退化更明显。因此 Exp2 支撑“持续匹配困难”和“局部热点/闲置共存”，不单独支撑“大幅端到端加速”。
- **理论契约**：最终动态方案必须把对应静态分配保留为在线 incumbent；否则动态结果劣于静态只能说明实现没有满足设计空间包含关系。

最终可观收益需要结合 Exp4 的动态映射实测、Exp5 的预取协同，以及 Exp7 的完整 Block 近似结果共同论证。
"""


def update(name: str, analysis_code: str, conclusion: str) -> None:
    path = ROOT / "fig" / f"{name}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    retained = []
    for cell in notebook["cells"]:
        if TAG in cell.get("metadata", {}).get("tags", []):
            continue
        source = "".join(cell.get("source", []))
        source = source.replace("DATE2", "DATE3")
        source = source.replace("outputs/DATE3", "outputs/DATE3")
        source = source.replace("fig/DATE3", "fig/DATE3")
        source = source.replace(
            "## 2. Banked INT32 ACC工作集",
            "## 2. Banked INT32 ACC 层级累计足迹与停顿",
        )
        source = source.replace(
            "这里观察ACC工作集与暴露停顿",
            "这里观察层内所有输出Tile的累计ACC足迹与暴露停顿",
        )
        source = source.replace(
            "ax.set_xlabel('Banked INT32 ACC working-set bytes')",
            "ax.set_xlabel('Layer-aggregate INT32 ACC footprint (bytes)')",
        )
        source = source.replace(
            "ax.set_title('ACC working set and exposed stall')",
            "ax.set_title('Layer-aggregate ACC footprint and exposed stall')",
        )
        cell["source"] = [line + "\n" for line in source.rstrip("\n").splitlines()]
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        retained.append(cell)
    # Replace the old static final judgment; the preceding plotting cells are
    # retained and all now consume DATE3 paths.
    if retained and retained[-1]["cell_type"] == "markdown":
        retained.pop()
    retained.extend((code(analysis_code), markdown(conclusion)))
    notebook["cells"] = retained
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(path)


def main() -> None:
    update("exp1", EXP1_CODE, EXP1_MD)
    update("exp2", EXP2_CODE, EXP2_MD)


if __name__ == "__main__":
    main()
