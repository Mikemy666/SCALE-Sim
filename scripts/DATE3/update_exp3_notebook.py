"""Retarget Exp3 to DATE3 and add reproducible paper-contract analysis."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG = "date3_exp3_analysis"


def cell(kind: str, source: str):
    value = {
        "cell_type": kind,
        "metadata": {"tags": [TAG]},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


COMPONENT_MD = """
## 3.1 退化周期的直接分量归因

下图只展示发生退化的配置，并使用原始 baseline matrix 的周期分量差值。
正值表示预取增加了该路径的暴露周期，负值表示预取消除了部分等待。
"""

COMPONENT_CODE = r"""
component_fields = [
    'bank_stall_cycles', 'weight_load_stall_cycles',
    'prefetch_miss_stall_cycles', 'prefetch_interference_stall_cycles',
    'mapping_overhead_cycles', 'communication_stall_cycles',
    'other_stall_cycles',
]
component_rows = []
for window in sorted(pf.window.unique()):
    for chunk in sorted(pf.chunk_tiles.unique()):
        source = ROOT / 'outputs/DATE3/window_chunk' / f'w{window}_c{chunk}' / 'baseline_matrix.csv'
        matrix = pd.read_csv(source).set_index('baseline')
        static = matrix.loc['Static-NoPF']
        naive = matrix.loc['Static-NaivePF']
        row = {'window': window, 'chunk_tiles': chunk}
        for field in component_fields:
            row[field] = naive[field] - static[field]
        row['component_delta_sum'] = sum(row[field] for field in component_fields)
        row['total_cycle_delta'] = naive.total_cycles - static.total_cycles
        component_rows.append(row)
components = pd.DataFrame(component_rows)
assert np.allclose(components.component_delta_sum, components.total_cycle_delta)
regressing = components[components.total_cycle_delta.gt(0)].copy()
regressing['config'] = 'W' + regressing.window.astype(str) + '/C' + regressing.chunk_tiles.astype(str)
plot_columns = [
    'weight_load_stall_cycles', 'prefetch_interference_stall_cycles',
    'bank_stall_cycles', 'prefetch_miss_stall_cycles', 'mapping_overhead_cycles',
    'communication_stall_cycles', 'other_stall_cycles',
]
component_labels = {
    'weight_load_stall_cycles': 'Weight-service stall',
    'prefetch_interference_stall_cycles': 'Explicit prefetch interference',
    'bank_stall_cycles': 'Bank stall',
    'prefetch_miss_stall_cycles': 'Prefetch-miss stall',
    'mapping_overhead_cycles': 'Mapping overhead',
    'communication_stall_cycles': 'Exposed EP wait',
    'other_stall_cycles': 'Combine/other',
}
fig, ax = plt.subplots(figsize=(11, 4.8))
regressing.set_index('config')[plot_columns].rename(columns=component_labels).plot(
    kind='bar', stacked=True, ax=ax
)
ax.axhline(0, color='black', lw=.8)
ax.set_ylabel('Cycle delta vs Static-NoPF')
ax.set_xlabel('Regressing fixed-prefetch configuration')
ax.set_title('Where the naive-prefetch slowdown is exposed')
ax.legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.savefig(FIG/'exp3_component_attribution.pdf', bbox_inches='tight'); plt.show()
display(regressing[['window', 'chunk_tiles', 'total_cycle_delta', *plot_columns]])
"""

VERDICT_CODE = r"""
active = pf[pf.window.gt(0)].copy()
best = active.loc[active.cycle_change_pct.idxmin()]
worst = active.loc[active.cycle_change_pct.idxmax()]
short = active[active.window.isin([1, 2, 4])]
large = active[active.window.isin([16, 32, 64])]
quality_identity = np.allclose(
    active.coverage_pf, active.accuracy_pf, equal_nan=True
)
same_volume = active.prefetched_unique_bytes_pf.nunique() == 1
explicit_interference_count = int(
    active.interference_stall_delta.ne(0).sum()
)
metrics = pd.Series({
    'active_configurations': len(active),
    'faster_configurations': int(active.cycle_change_pct.lt(0).sum()),
    'slower_configurations': int(active.cycle_change_pct.gt(0).sum()),
    'best_cycle_change_pct': best.cycle_change_pct,
    'best_window': int(best.window),
    'best_chunk': int(best.chunk_tiles),
    'worst_cycle_change_pct': worst.cycle_change_pct,
    'worst_window': int(worst.window),
    'worst_chunk': int(worst.chunk_tiles),
    'late_ratio_correlation': active[['cycle_change_pct', 'late_prefetch_ratio_pf']].corr().iloc[0, 1],
    'occupancy_correlation': active[['cycle_change_pct', 'occupancy_mbyte_cycles']].corr().iloc[0, 1],
    'conflict_correlation': active[['cycle_change_pct', 'conflict_change_pct']].corr().iloc[0, 1],
    'explicit_interference_nonzero_configs': explicit_interference_count,
    'all_active_configs_prefetch_same_bytes': same_volume,
    'coverage_equals_accuracy_here': quality_identity,
})
display(metrics.rename('value').to_frame().round(4))
checks = pd.Series({
    'Both speedup and slowdown are observed': active.cycle_change_pct.lt(0).any() and active.cycle_change_pct.gt(0).any(),
    'Large fixed settings can become late': large.late_prefetch_ratio_pf.gt(0).any(),
    'Late ratio tracks slowdown': metrics['late_ratio_correlation'] > 0.8,
    'Occupancy tracks slowdown': metrics['occupancy_correlation'] > 0.5,
    'Some slowdown has explicit interference': explicit_interference_count > 0,
    'All prefetched data can still be harmful': ((active.unused_prefetch_ratio_pf == 0) & (active.cycle_change_pct > 0)).any(),
})
display(checks.rename('pass').to_frame())
assert checks.all(), checks[~checks]
"""

VERDICT_MD = r"""
## DATE3 实验 3 论文判断

- **核心 C3 结论以本次输出为准**：只有上一单元同时报告至少一个加速点和至少一个减速点，才能写成“固定 Naive Prefetch 不是单调有益”。配置数量、最佳/最差点和百分比必须引用上方动态生成的 `metrics`，不能沿用旧结果。
- **退化链条必须由分量图支撑**：结合 late ratio、驻留 byte-cycles、Bank conflict 和显式 interference 分量说明原因；不把相关性写成因果，也不预设一定是大 Window 或大 Chunk 最差。
- **不能写成简单单调规律**：结果不是“Window/Chunk 越大一定越差”。不同组合表现明显非单调，恰恰说明 Window、Chunk 和 Bank placement 必须联合在线选择。
- **覆盖边界由布尔检查决定**：不能预先声称某个 Window 区间全部及时，也不能预先声称 Coverage 与 Accuracy 相等；分别读取 `checks`、`same_volume` 和 `quality_identity`。
- **控制开销边界**：若 `mapping_overhead_cycles` 始终为零，本实验不能证明小 Chunk 的控制开销，只能讨论传输及时性、驻留和 Bank 干扰。

因此 Exp3 足以完成论文中的过渡论证：**只采用固定预取会出现性能和资源占用风险，需要后续的 Bank-aware、Coverage/Accuracy-constrained Chunk/Window 协同优化。**
"""

ROOT_CAUSE_CODE = r"""
q=pf[pf.window.gt(0)].copy()
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
scatter = None
for ax,x,label,title in [
    (axes[0],'late_prefetch_ratio_pf','Late-prefetch ratio','(a) Timeliness'),
    (axes[1],'occupancy_mbyte_cycles','Occupancy (MByte-cycles)','(b) Residency pressure'),
    (axes[2],'conflict_change_pct','Bank-conflict increase (%)','(c) Bank conflict')]:
    scatter=ax.scatter(q[x],q.cycle_change_pct,c=q.chunk_tiles,cmap='viridis',s=60)
    ax.axhline(0,color='black',lw=.8);ax.set_xlabel(label);ax.set_ylabel('Cycle change (%)')
    ax.set_title(title);ax.grid(axis='y',alpha=.2)
colorbar=fig.colorbar(scatter,ax=axes.ravel().tolist(),shrink=.86,pad=.02)
colorbar.set_label('Chunk size (tiles)')
cor=q[['cycle_change_pct','timely_prefetch_ratio_pf','late_prefetch_ratio_pf',
       'occupancy_mbyte_cycles','conflict_change_pct']].corr()
fig.subplots_adjust(wspace=.28,right=.91)
plt.savefig(FIG/'exp3_root_cause_correlation.pdf',bbox_inches='tight');plt.show()
display(cor.round(3))
"""


def main() -> None:
    path = ROOT / "fig/exp3.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    retained = []
    for original in notebook["cells"]:
        if TAG in original.get("metadata", {}).get("tags", []):
            continue
        source = "".join(original.get("source", []))
        source = source.replace("DATE2", "DATE3")
        source = source.replace(
            "## 2. Window过大或Chunk过大：迟到与驻留压力同时增加",
            "## 2. 不匹配的 Window×Chunk：迟到与驻留压力增加",
        )
        source = source.replace(
            "'prefetch_occupancy_byte_cycles','compute_transfer_overlap_cycles'}",
            "'prefetch_occupancy_byte_cycles','compute_transfer_overlap_cycles',\n"
            "          'coverage','accuracy','prefetched_unique_bytes',\n"
            "          'coverage_valid','accuracy_valid'}",
        )
        source = source.replace(
            "pf['conflict_change_pct']=(pf.bank_conflict_count_pf/pf.bank_conflict_count_nop-1)*100",
            "conflict_den=pf.bank_conflict_count_nop.to_numpy()\n"
            "conflict_num=(pf.bank_conflict_count_pf-pf.bank_conflict_count_nop).to_numpy()\n"
            "pf['conflict_change_pct']=np.where(conflict_den>0,conflict_num/conflict_den*100,\n"
            "                                         np.where(conflict_num==0,0.0,np.nan))",
        )
        source = source.replace(
            "limit=float(np.nanmax(np.abs(perf.to_numpy())))\n"
            "im=ax.imshow(perf,aspect='auto',cmap='RdYlGn_r',\n"
            "             norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))",
            "vmin=float(np.nanmin(perf.to_numpy())); vmax=float(np.nanmax(perf.to_numpy()))\n"
            "im=ax.imshow(perf,aspect='auto',cmap='RdYlGn_r',\n"
            "             norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))",
        )
        if (original["cell_type"] == "code"
                and "Late-prefetch ratio" in source
                and "fig,axes=plt.subplots(1,3" in source):
            source = ROOT_CAUSE_CODE.strip()
        original["source"] = [
            line + "\n" for line in source.rstrip("\n").splitlines()
        ]
        if original["cell_type"] == "code":
            original["execution_count"] = None
            original["outputs"] = []
        retained.append(original)
    if retained and retained[-1]["cell_type"] == "markdown":
        retained.pop()
    # Put direct cycle attribution after the correlation section.
    insertion = next(
        index for index, value in enumerate(retained)
        if value["cell_type"] == "markdown"
        and "所有预取最终都被使用" in "".join(value["source"])
    )
    retained[insertion:insertion] = [
        cell("markdown", COMPONENT_MD), cell("code", COMPONENT_CODE)
    ]
    retained.extend((cell("code", VERDICT_CODE), cell("markdown", VERDICT_MD)))
    notebook["cells"] = retained
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
