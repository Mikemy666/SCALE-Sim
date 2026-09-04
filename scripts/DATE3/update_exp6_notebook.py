"""Build the DATE3 Exp6 one-variable-at-a-time sensitivity notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "fig/exp6.ipynb"


def cell(kind: str, source: str) -> dict:
    value = {
        "cell_type": kind,
        "metadata": {"tags": ["date3_exp6_analysis"]},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


HEADER = r"""
# DATE3 实验6：四模型单变量敏感性与失败边界

每次只改变一个变量，并在 HMoE、Mixtral、MoDSE、Switchtrans 四个模型上比较：
`Static-555-NoPF`、`Static-Opt-NoPF`、`Dynamic-NoPF`、
`Static-Opt-FixedPF`、`Dynamic-FixedPF` 和唯一正式方案 `PIVOT（MemDomain）`。

判定只在**同一模型、同一变量值**内部进行。跨变量值的绝对周期变化代表工作量、
路由或 EP 通信量变化，不要求单调，也不能相互比较得出方案优劣。
"""

LOAD = r"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'fig' else Path.cwd().resolve()
OUT = ROOT / 'outputs/DATE3'
FIG = ROOT / 'fig/DATE3'
FIG.mkdir(parents=True, exist_ok=True)
MODELS = ['HMoE', 'Mixtral', 'MoDSE', 'Switchtrans']
SCHEMES = ['Static-555-NoPF', 'Static-Opt-NoPF', 'Dynamic-NoPF',
           'Static-Opt-FixedPF', 'Dynamic-FixedPF', 'PIVOT']
CONTROLS = SCHEMES[:-1]
MATERIAL_TOL = 1.0  # |gain| < 1% is displayed as a near tie, not hidden.
COLORS = {
    'Static-555-NoPF': '#9E9E9E', 'Static-Opt-NoPF': '#F28E2B',
    'Dynamic-NoPF': '#4E79A7', 'Static-Opt-FixedPF': '#EDC948',
    'Dynamic-FixedPF': '#76B7B2', 'PIVOT': '#59A14F',
}
LABELS = {
    'Static-555-NoPF': 'Static-555', 'Static-Opt-NoPF': 'Static-Opt',
    'Dynamic-NoPF': 'Dynamic-NoPF', 'Static-Opt-FixedPF': 'Static-Opt+FixedPF',
    'Dynamic-FixedPF': 'Dynamic-FixedPF', 'PIVOT': 'PIVOT',
}
VALUE_ORDER = {
    'expert_count': ['4', '8', '16'],
    'token_count': ['32', '128', '256', '512'],
    'top_k': ['1', '2'],
    'expert_parallel': ['1', '2'],
    'routing_severity': ['balanced', 'light', 'high'],
    'routing_seed': [
        *[f'light_seed{x}' for x in range(40, 45)],
        *[f'high_seed{x}' for x in range(40, 45)],
    ],
}

source = OUT / 'exp6/robustness_comparison.csv'
assert source.exists(), source
d = pd.read_csv(source)
d['value'] = d.value.astype(str)
assert len(d) == 96 * len(SCHEMES)
assert set(d.policy_name) == set(SCHEMES)
assert set(d.model) == set(MODELS)
assert set(d.variable) == set(VALUE_ORDER)
assert not d.duplicated(['variable', 'value', 'model', 'policy_name']).any()
for variable, order in VALUE_ORDER.items():
    assert set(d.loc[d.variable.eq(variable), 'value']) == set(order), variable

wide = d.pivot(
    index=['variable', 'value', 'model'], columns='policy_name', values='total_cycles'
).reindex(columns=SCHEMES)
best_control = wide[CONTROLS].min(axis=1)
best_control_name = wide[CONTROLS].idxmin(axis=1)
static_tuning = (1 - wide['Static-Opt-NoPF']/wide['Static-555-NoPF']) * 100
dynamic_reduction = (1 - wide['Dynamic-NoPF']/wide['Static-Opt-NoPF']) * 100
fixed_pf_mapping_reduction = (
    1 - wide['Dynamic-FixedPF']/wide['Static-Opt-FixedPF']
) * 100
pivot_vs_static = (1 - wide.PIVOT/wide['Static-555-NoPF']) * 100
pivot_vs_best = (1 - wide.PIVOT/best_control) * 100
assert (static_tuning >= -1e-9).all(), 'Static-Opt regresses below Static-555'
assert (dynamic_reduction >= -1e-9).all(), 'Dynamic-NoPF regresses below Static-Opt'
assert (fixed_pf_mapping_reduction >= -1e-9).all(), 'Dynamic-FixedPF regresses below Static-Opt-FixedPF'
print(f'Loaded {len(d)} rows = 96 configurations × {len(SCHEMES)} schemes.')
"""

PLOT_HELPER = r"""
def variable_table(variable):
    q = wide.xs(variable, level='variable').reset_index()
    q['Dynamic reduction vs Static (%)'] = (
        1 - q['Dynamic-NoPF']/q['Static-Opt-NoPF']
    ) * 100
    q['PIVOT reduction vs Static-555 (%)'] = (1 - q.PIVOT/q['Static-555-NoPF']) * 100
    indexed = pd.MultiIndex.from_frame(q[['value', 'model']])
    q['Best control'] = best_control_name.xs(variable, level='variable').reindex(indexed).to_numpy()
    q['PIVOT gain vs best control (%)'] = pivot_vs_best.xs(
        variable, level='variable'
    ).reindex(indexed).to_numpy()
    order = {value: index for index, value in enumerate(VALUE_ORDER[variable])}
    q['_order'] = q.value.map(order)
    return q.sort_values(['model', '_order']).drop(columns='_order')


def plot_variable(variable, label, filename):
    table = variable_table(variable)
    values = VALUE_ORDER[variable]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.0), sharey=False)
    width = .13
    for ax, model in zip(axes, MODELS):
        current = table[table.model.eq(model)].set_index('value').reindex(values)
        x = np.arange(len(values))
        for index, scheme in enumerate(SCHEMES):
            normalized = current[scheme] / current['Static-555-NoPF']
            colors = COLORS[scheme]
            if scheme == 'PIVOT':
                colors = [
                    '#59A14F' if gain >= -MATERIAL_TOL else '#E15759'
                    for gain in current['PIVOT gain vs best control (%)']
                ]
            bars = ax.bar(
                x + (index-(len(SCHEMES)-1)/2)*width, normalized, width,
                color=colors, label=LABELS[scheme],
            )
            if scheme == 'PIVOT':
                for bar, gain in zip(bars, current['PIVOT gain vs best control (%)']):
                    if gain < -MATERIAL_TOL:
                        bar.set_hatch('//')
                    ax.text(
                        bar.get_x()+bar.get_width()/2, bar.get_height()*1.025,
                        f'{gain:+.1f}%', ha='center', va='bottom',
                        rotation=90 if len(values) > 4 else 0,
                        fontsize=7, color='#B22222' if gain < -MATERIAL_TOL else '#27632A',
                    )
        ax.axhline(1, color='black', lw=.8)
        ax.set_xticks(x, values, rotation=32 if len(values) > 4 else 0, ha='right' if len(values) > 4 else 'center')
        ax.set_title(model); ax.set_xlabel(label)
        ax.set_ylabel('Cycles / Static-555 (lower is better)')
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(.5, .925),
        ncol=6, frameon=False,
    )
    fig.suptitle(f'Exp6 sensitivity: {label}', y=.995)
    fig.subplots_adjust(top=.76, bottom=.18, wspace=.28)
    target = FIG/filename
    plt.savefig(target, bbox_inches='tight')
    plt.savefig(target.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.show(); plt.close(fig)
    display(table.round(3))
    return table
"""

VARIABLE_CELLS = (
    ("expert_count", "Expert count", "exp6_expert_count.pdf"),
    ("token_count", "Token count", "exp6_token_count.pdf"),
    ("top_k", "Top-k", "exp6_top_k.pdf"),
    ("expert_parallel", "Expert-parallel NPU count", "exp6_expert_parallel.pdf"),
    ("routing_severity", "Routing severity", "exp6_routing_severity.pdf"),
    ("routing_seed", "Routing distribution and seed", "exp6_routing_seed.pdf"),
)

DIAG_MD = r"""
## 6.7 契约统计与失败原因

绿色 PIVOT 柱表示相对五个可实现对照的逐配置最优者至多相差 1%；红色斜线柱表示
PIVOT 发生超过 1% 的实质退化。柱顶仍标出未经截断的精确收益，不隐藏小幅负值。
失败表保留最佳对照名称，后续按短序列、固定预取局部优势或EP关键路径分别解释。
"""

DIAG_CODE = r"""
rows = []
for variable in VALUE_ORDER:
    gain = pivot_vs_best.xs(variable, level='variable')
    static_gain = pivot_vs_static.xs(variable, level='variable')
    dynamic_gain = dynamic_reduction.xs(variable, level='variable')
    rows.append({
        'variable': variable, 'configurations': len(gain),
        'PIVOT wins': int((gain > 1e-9).sum()),
        'PIVOT ties': int(gain.abs().le(1e-9).sum()),
        'PIVOT regressions': int((gain < -1e-9).sum()),
        'PIVOT material regressions (<-1%)': int((gain < -MATERIAL_TOL).sum()),
        'incremental gain min (%)': gain.min(),
        'incremental gain mean (%)': gain.mean(),
        'incremental gain max (%)': gain.max(),
        'PIVOT vs Static min (%)': static_gain.min(),
        'PIVOT vs Static mean (%)': static_gain.mean(),
        'Dynamic regressions': int((dynamic_gain < -1e-9).sum()),
        'Fixed-PF mapping regressions': int((
            fixed_pf_mapping_reduction.xs(variable, level='variable') < -1e-9
        ).sum()),
    })
contract = pd.DataFrame(rows).set_index('variable')
display(contract.round(3))

failure = pd.DataFrame({
    'best_control': best_control_name,
    'best_control_cycles': best_control,
    'PIVOT_cycles': wide.PIVOT,
    'PIVOT_gain_vs_best_control_pct': pivot_vs_best,
    'PIVOT_gain_vs_Static_pct': pivot_vs_static,
})[pivot_vs_best < -1e-9].sort_values('PIVOT_gain_vs_best_control_pct')
display(failure.round(3))

material_failure = failure[failure.PIVOT_gain_vs_best_control_pct < -MATERIAL_TOL]
print(f'Material failures (>1% slowdown): {len(material_failure)}/{len(wide)} configurations.')
"""

MECHANISM_CODE = r"""
# Attribute material failures using like-for-like PIVOT versus Dynamic-FixedPF
# counters. Ratios above one mean PIVOT creates more work/pressure; coverage is
# reported as a percentage-point difference because a ratio is unstable near zero.
metric_names = [
    'total_cycles', 'local_memory_stall_cycles', 'prefetch_requests',
    'late_bytes', 'hbm_queue_wait_cycles', 'hbm_max_queue_depth', 'coverage',
]
pair = d[d.policy_name.isin(['Dynamic-FixedPF', 'PIVOT'])].pivot(
    index=['variable', 'value', 'model'], columns='policy_name', values=metric_names
)
mechanism = pd.DataFrame(index=pair.index)
for metric in metric_names[:-1]:
    mechanism[f'{metric}: PIVOT / Dynamic-FixedPF'] = (
        pair[(metric, 'PIVOT')] / pair[(metric, 'Dynamic-FixedPF')].clip(lower=1e-12)
    )
mechanism['coverage change (percentage points)'] = 100 * (
    pair[('coverage', 'PIVOT')] - pair[('coverage', 'Dynamic-FixedPF')]
)
material_mechanism = mechanism.reindex(material_failure.index)
display(material_mechanism.groupby(level='variable').median().round(3))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
failure_variables = list(material_mechanism.index.get_level_values('variable').unique())
x = np.arange(len(failure_variables))
diagnostic_columns = [
    'total_cycles: PIVOT / Dynamic-FixedPF',
    'prefetch_requests: PIVOT / Dynamic-FixedPF',
    'hbm_queue_wait_cycles: PIVOT / Dynamic-FixedPF',
]
titles = ['Cycle inflation', 'Request amplification', 'HBM queue-wait amplification']
for ax, column, title in zip(axes, diagnostic_columns, titles):
    values = material_mechanism[column].groupby(level='variable').median().reindex(failure_variables)
    bars = ax.bar(x, values, color='#E15759')
    ax.bar_label(bars, labels=[f'{v:.2f}x' for v in values], padding=3, fontsize=8)
    ax.axhline(1, color='black', lw=.8)
    if column.startswith('hbm_queue_wait_cycles'):
        ax.set_yscale('log')
        ax.set_ylabel('Ratio (log scale)')
    ax.set_xticks(x, failure_variables, rotation=25, ha='right')
    ax.set_title(title); ax.grid(axis='y', alpha=.2)
fig.suptitle('Why PIVOT loses in material-failure configurations')
plt.tight_layout()
plt.savefig(FIG/'exp6_material_failure_mechanism.pdf', bbox_inches='tight')
plt.savefig(FIG/'exp6_material_failure_mechanism.png', dpi=220, bbox_inches='tight')
plt.show(); plt.close(fig)
"""

ROOT_CAUSE_CODE = r"""
# Load PIVOT-only runtime diagnostics without treating the internal PIVOT-CA
# summary identifier as another public scheme.
runtime_rows = []
for row in d[d.policy_name.eq('PIVOT')].itertuples(index=False):
    with (OUT/'robustness_factorial'/row.variant/'summary.csv').open(
        newline='', encoding='utf-8'
    ) as stream:
        summary = next(csv.DictReader(stream))
    runtime_rows.append({
        'variable': row.variable, 'value': str(row.value), 'model': row.model,
        'fallback_rate': float(summary['fallback_rate']),
        'guard_rate': float(summary['online_incumbent_guard_rate']),
        'coverage': float(summary['coverage']), 'accuracy': float(summary['accuracy']),
        'late_ratio': float(summary['late_bytes'])/max(float(summary['prefetched_bytes']), 1),
        'selected_chunk_mean': float(summary['selected_chunk_mean']),
        'selected_window_mean': float(summary['selected_window_mean']),
        'bank_stall_cycles': int(summary['bank_stall_cycles']),
        'weight_load_stall_cycles': int(summary['weight_load_stall_cycles']),
        'prefetch_interference_stall_cycles': int(summary['prefetch_interference_stall_cycles']),
        'peak_occupied_bytes': int(summary['peak_occupied_bytes']),
    })
runtime = pd.DataFrame(runtime_rows).set_index(['variable', 'value', 'model'])

token32 = wide.xs(('token_count', '32'), level=('variable', 'value')).reindex(MODELS)
token_runtime = runtime.xs(('token_count', '32'), level=('variable', 'value')).reindex(MODELS)
token_aggregate = d[
    d.variable.eq('token_count') & d.value.eq('32')
].set_index(['model', 'policy_name'])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
x = np.arange(len(MODELS)); width = .13
for index, scheme in enumerate(SCHEMES):
    color = '#E15759' if scheme == 'PIVOT' else COLORS[scheme]
    axes[0].bar(
        x+(index-(len(SCHEMES)-1)/2)*width,
        token32[scheme]/token32['Static-555-NoPF'], width,
        color=color, label=LABELS[scheme],
    )
axes[0].axhline(1, color='black', lw=.8); axes[0].set_yscale('log')
axes[0].set_xticks(x, MODELS); axes[0].set_ylabel('Cycles / Static-555 (log)')
axes[0].set_title('(a) Token=32 short-sequence case'); axes[0].legend(fontsize=6)

quality = token_runtime[['coverage', 'accuracy', 'fallback_rate']] * 100
quality.plot(kind='bar', ax=axes[1], color=['#4E79A7', '#76B7B2', '#E15759'])
axes[1].set_ylabel('Ratio (%)'); axes[1].set_xlabel('')
axes[1].set_title('(b) Quality failure and fallback'); axes[1].legend(fontsize=7)

static_stall = token_aggregate.xs('Static-555-NoPF', level='policy_name').reindex(MODELS).local_memory_stall_cycles
pivot_stall = token_aggregate.xs('PIVOT', level='policy_name').reindex(MODELS).local_memory_stall_cycles
inflation = pivot_stall/static_stall
bars = axes[2].bar(MODELS, inflation, color='#E15759')
axes[2].bar_label(bars, labels=[f'{v:.1f}x' for v in inflation], padding=3)
axes[2].axhline(1, color='black', lw=.8)
axes[2].set_ylabel('PIVOT local stall / Static local stall')
axes[2].set_title('(c) Local-stall inflation')
for ax in axes: ax.grid(axis='y', alpha=.2)
plt.tight_layout(); plt.savefig(FIG/'exp6_failure_diagnosis.pdf', bbox_inches='tight'); plt.show(); plt.close(fig)

token_diagnosis = token_runtime.join(pd.DataFrame({
    'PIVOT_cycles': token32.PIVOT,
    'Static_555_cycles': token32['Static-555-NoPF'],
    'stall_inflation_x': inflation,
}))
display(token_diagnosis.round(3))

non_token32_failure = failure.drop(
    index=[idx for idx in failure.index if idx[0] == 'token_count' and str(idx[1]) == '32']
)
display(non_token32_failure.round(3))
"""

VERDICT = r"""
## DATE3 实验6判断

- 每个变量独立分析四个模型和六个方案，不跨变量连接趋势。
- 三条必须成立的公平契约是：Static-Opt不劣于Static-555、PIVOT-Map不劣于Static-Opt、
  相同固定预取下PIVOT-Map+PF不劣于Static-Opt+PF。
- 固定预取相对NoPF允许变好或变差；这不是契约失败，而是Exp3/Exp5所验证的参数敏感性。
- PIVOT相对五个对照的win/tie/regress必须使用重新运行后生成的统计，不再保留旧四组数据结论。
- 小于1%的差距作为近似持平显示，但表格保留精确值；超过1%的红色斜线配置属于实质失败边界。
- Token=32、Token=128、Expert=16、EP=1及高不平衡路由用于分析短提前量、请求放大和HBM排队。
  这些配置可以作为“不支持/性能较差配置”的原因分析，但不能被描述为PIVOT普遍最优的证据。
"""


CELLS = [cell("markdown", HEADER), cell("code", LOAD), cell("code", PLOT_HELPER)]
for variable, label, filename in VARIABLE_CELLS:
    CELLS.append(cell("code", f"{variable} = plot_variable({variable!r}, {label!r}, {filename!r})"))
CELLS.extend([
    cell("markdown", DIAG_MD), cell("code", DIAG_CODE),
    cell("code", MECHANISM_CODE), cell("code", ROOT_CAUSE_CODE), cell("markdown", VERDICT),
])


def main() -> None:
    old = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook = {
        "cells": CELLS,
        "metadata": old.get("metadata", {}),
        "nbformat": old.get("nbformat", 4),
        "nbformat_minor": old.get("nbformat_minor", 5),
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
