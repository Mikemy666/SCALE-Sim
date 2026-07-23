"""Create the standalone DATE2 exp4 cross-model analysis notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


cells = [
md("""
# DATE2 实验 4：跨模型整体性能

本实验在两个同构专家模型（Mixtral、Switchtrans）和两个异构专家模型（HMoE、MoDSE）上比较七种方案。主数据来自 `outputs/DATE2/overall/*.csv`；`outputs/DATE2/exp4/*/MEASURED_SELECTIONS.csv` 用于核对实际测量候选。

分析目标：

1. 比较端到端周期和相对 Static-NoPF 的加速；
2. 检查 Raw Bank-aware 在不同专家结构上的适用性；
3. 检查 Safe 的实际选择及其与 Oracle 的差距；
4. 明确当前数据能够和不能够支持的论文结论。
"""),
code("""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scalesim.memory.memdomain_experiment import workload_digest

roots=[Path.cwd().resolve(),Path.cwd().resolve().parent]
ROOT=next(p for p in roots if (p/'outputs/DATE2/overall').exists())
OUT=ROOT/'outputs/DATE2'; FIG=ROOT/'fig/DATE2'; FIG.mkdir(parents=True,exist_ok=True)
ORDER=['Static-NoPF','Static-NaivePF','Dynamic-NoPF','Dynamic-NaivePF',
       'MemDomain-Raw','MemDomain-Safe','Oracle']
MODEL_ORDER=['HMoE','MoDSE','Mixtral','Switchtrans']
MODEL_CLASS={'HMoE':'Heterogeneous','MoDSE':'Heterogeneous',
             'Mixtral':'Homogeneous','Switchtrans':'Homogeneous'}
frames=[]
for path in sorted((OUT/'overall').glob('*.csv')):
    data=pd.read_csv(path)
    model=path.stem
    config=json.loads((ROOT/f'configs/MoE/DATE2/overall/{model}.json').read_text())
    assert set(data.workload_hash)=={workload_digest(config)}, f'Stale result: {path}'
    assert set(data.baseline)==set(ORDER) and len(data)==len(ORDER)
    measured=pd.read_csv(OUT/f'exp4/{model}/MEASURED_SELECTIONS.csv')
    check=data[data.baseline.isin(measured.baseline)].set_index('baseline').total_cycles
    assert check.equals(measured.set_index('baseline').total_cycles), f'Detail mismatch: {model}'
    data['model']=model; data['model_class']=MODEL_CLASS[model]
    base=float(data.loc[data.baseline=='Static-NoPF','total_cycles'].iloc[0])
    data['normalized_cycles']=data.total_cycles/base
    data['speedup_vs_static_nopf']=base/data.total_cycles
    frames.append(data)
d=pd.concat(frames,ignore_index=True)
d.to_csv(FIG/'exp4_overall_data.csv',index=False)
print('Validated current workload hashes and detailed selections for four models')
"""),
md("""
## 1. 七种方案的端到端性能
"""),
code("""
p=d.pivot(index='model',columns='baseline',values='normalized_cycles').reindex(MODEL_ORDER)[ORDER]
colors=['#9e9e9e','#c7c7c7','#4e79a7','#76b7b2','#e15759','#59a14f','#b07aa1']
ax=p.plot(kind='bar',figsize=(13,5),color=colors,width=.86)
ax.axhline(1,color='black',lw=.8); ax.set_ylabel('Normalized total cycles (lower is better)')
ax.set_xlabel(''); ax.legend(ncol=4,fontsize=8); ax.tick_params(axis='x',rotation=0)
plt.tight_layout(); plt.savefig(FIG/'exp4_overall_performance.pdf',bbox_inches='tight'); plt.show()
summary=d.pivot(index='model',columns='baseline',values='speedup_vs_static_nopf').reindex(MODEL_ORDER)[ORDER]
display(summary.round(3))
"""),
md("""
Static-NaivePF 在四个模型上均获得约 1.76–1.97× 加速。MemDomain-Raw 仅在 MoDSE 上达到 1.772×，并略优于 Static-NaivePF；在其余三个模型上 Raw 明显退化。这表明当前 Raw 策略对模型结构敏感，不能据此宣称跨模型普遍最优。
"""),
md("""
## 2. Raw 相对动态朴素预取的收益与代价
"""),
code("""
raw=d[d.baseline=='MemDomain-Raw'].set_index('model')
naive=d[d.baseline=='Dynamic-NaivePF'].set_index('model')
raw_compare=pd.DataFrame(index=MODEL_ORDER)
raw_compare['cycle_change_percent']=(raw.total_cycles/naive.total_cycles-1)*100
raw_compare['interference_change_percent']=(raw.prefetch_interference_stall_cycles/
    naive.prefetch_interference_stall_cycles-1)*100
raw_compare['bank_conflict_change_percent']=(raw.bank_conflict_count/
    naive.bank_conflict_count-1)*100
raw_compare['coverage_delta']=raw.prefetch_coverage-naive.prefetch_coverage
raw_compare.to_csv(FIG/'exp4_raw_vs_dynamic_naive.csv')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,col,title in zip(axes,raw_compare.columns[:3],
    ['Total cycles','Prefetch interference','Bank conflicts']):
    values=raw_compare[col]
    ax.bar(values.index,values,color=np.where(values<=0,'#59a14f','#e15759'))
    ax.axhline(0,color='black',lw=.8); ax.set_ylabel('Change vs Dynamic-NaivePF (%)')
    ax.set_title(title); ax.tick_params(axis='x',rotation=20)
plt.tight_layout(); plt.savefig(FIG/'exp4_raw_tradeoff.pdf',bbox_inches='tight'); plt.show()
display(raw_compare.round(2))
"""),
md("""
MoDSE 是当前唯一同时获得端到端收益、干扰下降和冲突下降的模型：Raw 相对 Dynamic-NaivePF 周期减少约 2.65%，预取干扰减少约 23.1%。其他模型中 Raw 降低了预取覆盖率并产生额外 demand-load stall，收益不足以覆盖调度代价。
"""),
md("""
## 3. Safe 选择、Oracle 上界与候选集合缺口
"""),
code("""
safe=d[d.baseline=='MemDomain-Safe'].set_index('model').loc[MODEL_ORDER]
oracle=d[d.baseline=='Oracle'].set_index('model').loc[MODEL_ORDER]
selection=pd.DataFrame({
    'model_class':[MODEL_CLASS[m] for m in MODEL_ORDER],
    'safe_selected':safe.selected_candidate,
    'oracle_selected':oracle.selected_candidate,
    'safe_speedup':safe.speedup_vs_static_nopf,
    'oracle_speedup':oracle.speedup_vs_static_nopf,
    'safe_gap_vs_oracle_percent':(safe.total_cycles/oracle.total_cycles-1)*100,
    'fallback_used':safe.fallback_used
},index=MODEL_ORDER)
selection.to_csv(FIG/'exp4_safe_selection.csv')
fig,ax=plt.subplots(figsize=(9,4.3))
x=np.arange(len(selection)); width=.35
ax.bar(x-width/2,selection.safe_speedup,width,label='MemDomain-Safe',color='#59a14f')
ax.bar(x+width/2,selection.oracle_speedup,width,label='Oracle',color='#b07aa1')
ax.set_xticks(x,selection.index); ax.set_ylabel('Speedup vs Static-NoPF')
ax.set_title('Safe selection versus measured Oracle'); ax.legend()
plt.tight_layout(); plt.savefig(FIG/'exp4_safe_oracle_gap.pdf',bbox_inches='tight'); plt.show()
display(selection.round(3))
"""),
md("""
当前 Safe 候选来源是 `Dynamic-NoPF | MemDomain-Raw | Static-NoPF`，没有包含已经实际测量且在三个模型上最优的 `Static-NaivePF`。因此：

- MoDSE：Safe 正确选择 MemDomain-Raw，并与 Oracle 一致；
- HMoE、Mixtral、Switchtrans：Safe 回退到 Static-NoPF，虽然保证不比 NoPF 差，但错过了 Static-NaivePF 的显著收益；
- 这不是“Safe 机制失败”，而是 Safe 候选集合不完整。修正前不能用实验4宣称完整方案跨模型最优。
"""),
md("""
## 4. 同构与异构专家模型汇总
"""),
code("""
class_summary=(d[d.baseline.isin(['Static-NaivePF','Dynamic-NaivePF','MemDomain-Raw','MemDomain-Safe','Oracle'])]
    .groupby(['model_class','baseline']).speedup_vs_static_nopf.mean().unstack()
    .reindex(['Homogeneous','Heterogeneous']))
class_summary.to_csv(FIG/'exp4_model_class_summary.csv')
ax=class_summary[ORDER[1:2]+ORDER[3:]].plot(kind='bar',figsize=(10,4.3),color=colors[1:2]+colors[3:])
ax.axhline(1,color='black',lw=.8); ax.set_ylabel('Mean per-model speedup')
ax.set_xlabel(''); ax.tick_params(axis='x',rotation=0); ax.legend(ncol=3,fontsize=8)
plt.tight_layout(); plt.savefig(FIG/'exp4_model_class_comparison.pdf',bbox_inches='tight'); plt.show()
display(class_summary.round(3))
"""),
md("""
## 实验 4 结论

- 实验覆盖两个同构和两个异构专家模型，且四组输出均通过当前 workload hash 校验。
- 预取本身在四个模型上都有稳定的大幅收益，说明跨模型的主要优化机会成立。
- 当前 Raw Bank-aware 的优势只在 MoDSE 上成立，不能声称全域最优。
- 当前 Safe 在 MoDSE 上达到 Oracle，但其候选集合遗漏 Static-NaivePF，导致另外三个模型只回退到 NoPF。
- 论文主结果在修正 Safe 候选集合并重新运行实验4前，只能表述为“机制在 MoDSE 上有效、预取跨模型有效”，不能表述为“MemDomain-Safe 跨模型优于所有基线”。
""")
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
(ROOT / "fig/exp4.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(ROOT / "fig/exp4.ipynb")
