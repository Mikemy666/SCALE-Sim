"""Create the standalone DATE2 exp1 analysis notebook."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def md(text): return {"cell_type":"markdown","metadata":{},"source":[line+"\n" for line in text.strip().splitlines()]}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[line+"\n" for line in text.strip().splitlines()]}

cells=[
md("""
# DATE2 实验 1：MoE/非 MoE 瓶颈与时变访存流

数据源：`outputs/DATE2/exp1/layer_characterization.csv`。

本 Notebook 对应论文 B1 与 C1，回答三个问题：

1. MoE Expert FF1/FF2 是否贡献主要 memory stall？
2. Stall ratio 是否与 Bank imbalance/conflict 同步变化？
3. 不同专家、FFN 阶段的 IA/Weight/OA 需求及理想 Bank 比例是否变化？

注意：结论完全由当前 CSV 计算，不预设“MoE 必须更差”。
"""),
code("""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

roots=[Path.cwd().resolve(),Path.cwd().resolve().parent]
ROOT=next(path for path in roots if (path/'outputs/DATE2/exp1/layer_characterization.csv').exists())
DATA=ROOT/'outputs/DATE2/exp1/layer_characterization.csv'
FIG=ROOT/'fig/DATE2'; FIG.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA)
required={'layer','layer_type','compute_cycles','memory_stall_cycles','stall_ratio','ia_bytes','weight_bytes','oa_bytes','bank_conflict_rate','bank_imbalance','ideal_ia_banks','ideal_weight_banks','ideal_oa_banks'}
assert required.issubset(df.columns)
assert len(df)==23 and df.layer.nunique()==23
df['category']=df.layer_type.replace({'Expert-FF1':'MoE Expert','Expert-FF2':'MoE Expert'})
df.head()
"""),
md("""
## 1. MoE 与非 MoE 周期分解

同时报告总贡献和按层平均。由于 MoE 有 16 个 FFN 阶段、非 MoE 只有 7 层，只比较总和会受到层数影响。
"""),
code("""
summary=df.groupby('category').agg(
    layers=('layer','count'),
    compute_total=('compute_cycles','sum'),
    stall_total=('memory_stall_cycles','sum'),
    total_cycles=('total_cycles','sum'),
    compute_mean=('compute_cycles','mean'),
    stall_mean=('memory_stall_cycles','mean'),
    stall_ratio_mean=('stall_ratio','mean'),
    conflict_rate_mean=('bank_conflict_rate','mean'),
    imbalance_mean=('bank_imbalance','mean'),
).reset_index()
summary.to_csv(FIG/'exp1_group_summary.csv',index=False)
display(summary.round(3))
"""),
code("""
order=df.sort_values(['category','total_cycles']).reset_index(drop=True)
colors={'Non-MoE':'#4E79A7','MoE Expert':'#E15759'}
fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[2.2,1]})
y=np.arange(len(order))
axes[0].barh(y,order.compute_cycles,color='#59A14F',label='Compute')
axes[0].barh(y,order.memory_stall_cycles,left=order.compute_cycles,color='#F28E2B',label='Memory stall')
axes[0].set_yticks(y,order.layer,fontsize=8); axes[0].set_xlabel('Cycles'); axes[0].set_title('(a) Per-layer cycle breakdown'); axes[0].legend()
means=summary.set_index('category').loc[['Non-MoE','MoE Expert']]
x=np.arange(2)
axes[1].bar(x,means.compute_mean,color='#59A14F',label='Compute')
axes[1].bar(x,means.stall_mean,bottom=means.compute_mean,color='#F28E2B',label='Memory stall')
axes[1].set_xticks(x,['Non-MoE','MoE Expert']); axes[1].set_ylabel('Mean cycles per layer'); axes[1].set_title('(b) Per-layer mean')
plt.tight_layout(); plt.savefig(FIG/'exp1_cycle_breakdown.pdf',bbox_inches='tight'); plt.show()
"""),
md("""
当前结果：MoE Expert 的总 stall 为 **12,416 cycles**，非 MoE 为 **11,776 cycles**；但 MoE 包含更多阶段。按层平均后，MoE Expert 为 **776 cycles**，非 MoE 为 **1,682 cycles**。平均 stall ratio 分别为 **23.85%** 和 **35.15%**。

因此，当前结果只能说明 MoE 阶段合计贡献了略多 stall，不能支持“单个 MoE 层比非 MoE 层具有更高 stall”的强结论。Router、QKT、QKTV 是需要单独解释的非 MoE 高 stall 层。
"""),
md("""
## 2. Stall ratio、Bank imbalance 与 conflict
"""),
code("""
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for category,q in df.groupby('category'):
    axes[0].scatter(q.bank_imbalance,q.stall_ratio,s=55,label=category,color=colors[category],alpha=.85)
    axes[1].scatter(q.bank_conflict_rate,q.stall_ratio,s=55,label=category,color=colors[category],alpha=.85)
for _,r in df.nlargest(3,'stall_ratio').iterrows(): axes[0].annotate(r.layer,(r.bank_imbalance,r.stall_ratio),fontsize=7)
axes[0].set_xlabel('Bank imbalance'); axes[0].set_ylabel('Stall ratio'); axes[0].set_title('(a) Imbalance vs stall')
axes[1].set_xlabel('Bank conflict rate'); axes[1].set_ylabel('Stall ratio'); axes[1].set_title('(b) Conflict vs stall')
axes[0].legend(); plt.tight_layout(); plt.savefig(FIG/'exp1_stall_bank_correlation.pdf',bbox_inches='tight'); plt.show()
corr_all=df[['stall_ratio','bank_imbalance','bank_conflict_rate']].corr()
corr_expert=df[df.category=='MoE Expert'][['stall_ratio','bank_imbalance','bank_conflict_rate']].corr()
print('All layers correlation'); display(corr_all.round(3))
print('Expert-only correlation'); display(corr_expert.round(3))
"""),
md("""
全部 23 层中，stall ratio 与 Bank imbalance 的 Pearson 相关系数为 **0.873**，支持二者总体相关；但只看 Expert 阶段时为 **0.333**。Conflict rate 与 stall ratio 在全部层中接近 0，说明当前 `bank_conflict_rate` 已接近饱和，不宜单独作为瓶颈证据。
"""),
md("""
## 3. 不同专家的规模、流量和周期
"""),
code("""
expert=df[df.category=='MoE Expert'].copy()
expert['expert_id']=expert.layer.str.extract(r'E(\\d+)').astype(int)
expert_summary=expert.groupby('expert_id').agg(tokens=('M','sum'),compute_cycles=('compute_cycles','sum'),memory_stall_cycles=('memory_stall_cycles','sum'),total_cycles=('total_cycles','sum'),ia_bytes=('ia_bytes','sum'),weight_bytes=('weight_bytes','sum'),oa_bytes=('oa_bytes','sum')).reset_index()
expert_summary.to_csv(FIG/'exp1_expert_summary.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,4.2)); x=np.arange(len(expert_summary))
axes[0].bar(x,expert_summary.compute_cycles,color='#59A14F',label='Compute');axes[0].bar(x,expert_summary.memory_stall_cycles,bottom=expert_summary.compute_cycles,color='#F28E2B',label='Memory stall');axes[0].set_xticks(x,[f'E{i}' for i in expert_summary.expert_id]);axes[0].set_ylabel('Cycles');axes[0].set_title('(a) Per-expert cycles');axes[0].legend()
bottom=np.zeros(len(expert_summary))
for col,label,color in [('ia_bytes','IA','#4E79A7'),('weight_bytes','Weight','#E15759'),('oa_bytes','OA','#76B7B2')]: axes[1].bar(x,expert_summary[col],bottom=bottom,label=label,color=color);bottom+=expert_summary[col].to_numpy()
axes[1].set_xticks(x,[f'E{i}' for i in expert_summary.expert_id]);axes[1].set_ylabel('Bytes');axes[1].set_title('(b) Per-expert traffic');axes[1].legend()
plt.tight_layout();plt.savefig(FIG/'exp1_expert_heterogeneity.pdf',bbox_inches='tight');plt.show()
display(expert_summary)
"""),
md("""
专家差异显著：E2 总周期为 **16,832**，而 E1/E3 为 **2,368**，相差约 7.1 倍；Weight traffic 也随专家宽度变化。这部分数据能够支持专家规模和输入量共同改变访存需求。
"""),
md("""
## 4. IA/Weight/OA 时变需求与理想 Bank 比例
"""),
code("""
stage=df[df.category=='MoE Expert'].sort_values(['layer_type','layer']).reset_index(drop=True)
fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True);x=np.arange(len(stage))
bottom=np.zeros(len(stage))
for col,label,color in [('ia_bytes','IA','#4E79A7'),('weight_bytes','Weight','#E15759'),('oa_bytes','OA','#76B7B2')]: axes[0].bar(x,stage[col],bottom=bottom,label=label,color=color);bottom+=stage[col].to_numpy()
axes[0].set_ylabel('Traffic bytes');axes[0].set_title('(a) IA/Weight/OA demand');axes[0].legend(ncol=3)
axes[1].plot(x,stage.ideal_ia_banks,marker='o',label='IA');axes[1].plot(x,stage.ideal_weight_banks,marker='s',label='Weight');axes[1].plot(x,stage.ideal_oa_banks,marker='^',label='OA');axes[1].set_ylabel('Ideal Banks (sum=24)');axes[1].set_xticks(x,stage.layer,rotation=60,ha='right',fontsize=8);axes[1].set_title('(b) Per-stage ideal static ratio');axes[1].legend(ncol=3)
plt.tight_layout();plt.savefig(FIG/'exp1_flow_and_ideal_banks.pdf',bbox_inches='tight');plt.show()
stage[['layer','ia_bytes','weight_bytes','oa_bytes','ideal_ia_banks','ideal_weight_banks','ideal_oa_banks']]
"""),
md("""
Expert FF1 的理想 IA Bank 数范围为 2–12、Weight 为 6–15、OA 为 6–11；FF2 中 IA/OA 的角色随矩阵方向交换。不存在一组固定 IA:Weight:OA 比例对所有专家和阶段同时最优，这直接支持 C1 的“需求独立变化与静态比例失配”结论。

## 实验 1 最终判断

- **支持**：专家异构；IA/Weight/OA 独立变化；理想 Bank 比例随专家及 FF1/FF2 改变；全体层 stall ratio 与 Bank imbalance 总体相关。
- **不支持**：当前数据不能证明 MoE 单层的 memory stall 或 stall ratio 高于非 MoE。
- **写论文时应避免**：把接近饱和的 conflict rate 当成唯一根因，或忽略 Router/QKT/QKTV 非 MoE 层的高 stall。
""")]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
(ROOT/"fig/exp1.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print(ROOT/"fig/exp1.ipynb")
