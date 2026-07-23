"""Create the standalone DATE2 exp2 static-Bank analysis notebook."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def md(t):return {"cell_type":"markdown","metadata":{},"source":[x+"\n" for x in t.strip().splitlines()]}
def code(t):return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[x+"\n" for x in t.strip().splitlines()]}
cells=[
md("""
# DATE2 实验 2：静态 Bank 所有权失配

数据源：

- `outputs/DATE2/exp2/static_bank_sweep.csv`
- `outputs/DATE2/exp2/per_stage_best.csv`

固定总 Bank 数为 24，并要求 IA、Weight、OA 均至少获得 1 个 Bank，因此每个阶段共有 $C(23,2)=253$ 种静态分区。本实验比较全局固定比例与逐阶段最优比例。
"""),
code("""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
roots=[Path.cwd().resolve(),Path.cwd().resolve().parent]
ROOT=next(p for p in roots if (p/'outputs/DATE2/exp2/static_bank_sweep.csv').exists())
OUT=ROOT/'outputs/DATE2/exp2';FIG=ROOT/'fig/DATE2';FIG.mkdir(parents=True,exist_ok=True)
sweep=pd.read_csv(OUT/'static_bank_sweep.csv');best=pd.read_csv(OUT/'per_stage_best.csv')
assert len(sweep)==23*253 and sweep.layer.nunique()==23
assert sweep.groupby('layer').size().eq(253).all()
assert len(sweep[['ia_banks','weight_banks','oa_banks']].drop_duplicates())==253
assert (sweep.ia_banks+sweep.weight_banks+sweep.oa_banks).eq(24).all()
calculated=sweep.sort_values(['layer','total_cycles','ia_banks','weight_banks']).groupby('layer',as_index=False).first()
check=calculated[['layer','total_cycles','ia_banks','weight_banks','oa_banks']].merge(best[['layer','total_cycles','ia_banks','weight_banks','oa_banks']],on='layer',suffixes=('_calc','_file'))
for c in ('total_cycles','ia_banks','weight_banks','oa_banks'):assert (check[c+'_calc']==check[c+'_file']).all()
print('Validated:',len(sweep),'sweep rows and',len(best),'per-stage optima')
"""),
md("""
## 1. 每个阶段的最优 Bank 比例
"""),
code("""
ordered=best.sort_values(['layer_type','layer']).reset_index(drop=True);x=np.arange(len(ordered))
fig,ax=plt.subplots(figsize=(14,5));bottom=np.zeros(len(ordered))
for col,label,color in [('ia_banks','IA','#4E79A7'),('weight_banks','Weight','#E15759'),('oa_banks','OA','#76B7B2')]:
    ax.bar(x,ordered[col],bottom=bottom,label=label,color=color);bottom+=ordered[col].to_numpy()
ax.set_xticks(x,ordered.layer,rotation=60,ha='right',fontsize=8);ax.set_ylabel('Banks (sum=24)');ax.set_title('Per-stage best static IA:Weight:OA allocation');ax.legend(ncol=3)
plt.tight_layout();plt.savefig(FIG/'exp2_per_stage_best_ratio.pdf',bbox_inches='tight');plt.show()
print('Unique best ratios:',len(best[['ia_banks','weight_banks','oa_banks']].drop_duplicates()))
display(best.groupby('layer_type')[['ia_banks','weight_banks','oa_banks']].agg(['min','max']))
"""),
md("""
23 个阶段出现 **19 种**最优比例。Expert-FF1 的 IA/Weight/OA 最优范围分别为 2–12、6–15、6–11；Expert-FF2 为 6–10、6–15、2–12；非 MoE 层则达到 IA 3–22、Weight 1–2、OA 1–20。不存在能代表全部阶段的固定比例。
"""),
md("""
## 2. 全局固定比例搜索与二维热图
"""),
code("""
global_result=sweep.groupby(['ia_banks','weight_banks','oa_banks'],as_index=False).agg(total_cycles=('total_cycles','sum'),memory_stall_cycles=('memory_stall_cycles','sum'),bank_conflicts=('bank_conflict_count','sum'),hotspot_ratio=('hotspot_bank_ratio','mean'),idle_ratio=('idle_bank_ratio','mean'))
global_result['normalized_cycles']=global_result.total_cycles/global_result.total_cycles.min()
global_result.to_csv(FIG/'exp2_global_static_summary.csv',index=False)
global_best=global_result.nsmallest(1,'total_cycles').iloc[0]
pivot=global_result.pivot(index='ia_banks',columns='weight_banks',values='normalized_cycles')
fig,ax=plt.subplots(figsize=(8,6));values=pivot.to_numpy();vmax=min(2,float(np.nanpercentile(values,90)));masked=np.ma.masked_invalid(values);im=ax.imshow(masked,origin='lower',aspect='auto',cmap='viridis',vmin=1,vmax=vmax)
ax.set_xticks(range(len(pivot.columns)),pivot.columns);ax.set_yticks(range(len(pivot.index)),pivot.index);ax.set_xlabel('Weight Banks');ax.set_ylabel('IA Banks');ax.set_title('Global fixed allocation: normalized total cycles\\n(OA Banks = 24 - IA - Weight)');fig.colorbar(im,ax=ax,label='Normalized cycles')
best_x=list(pivot.columns).index(int(global_best.weight_banks));best_y=list(pivot.index).index(int(global_best.ia_banks));ax.scatter(best_x,best_y,marker='*',s=180,color='red',edgecolor='white')
plt.tight_layout();plt.savefig(FIG/'exp2_static_ratio_heatmap.pdf',bbox_inches='tight');plt.show()
global_result.nsmallest(10,'total_cycles')
"""),
md("""
全工作负载最优固定比例为 **8:8:8**，总周期 **112,144**。这与 DATE1 曾使用的 4:14:6 不同，说明旧实验的固定最优比例不能直接移植到新架构。
"""),
md("""
## 3. 代表性固定比例与逐阶段 Oracle
"""),
code("""
ratios=[(8,8,8),(7,10,7),(4,14,6),(12,8,4),(2,11,11)]
representative=pd.DataFrame([global_result[(global_result.ia_banks==a)&(global_result.weight_banks==w)&(global_result.oa_banks==o)].iloc[0] for a,w,o in ratios])
representative['ratio']=[f'{a}:{w}:{o}' for a,w,o in ratios]
stage_oracle=int(best.total_cycles.sum());fixed_best=int(global_best.total_cycles)
comparison=pd.concat([representative[['ratio','total_cycles','memory_stall_cycles']],pd.DataFrame([{'ratio':'Per-stage best','total_cycles':stage_oracle,'memory_stall_cycles':int(best.memory_stall_cycles.sum())}])],ignore_index=True)
comparison['normalized_to_stage_best']=comparison.total_cycles/stage_oracle
comparison.to_csv(FIG/'exp2_representative_comparison.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar(comparison.ratio,comparison.total_cycles,color=['#4E79A7']*5+['#59A14F']);axes[0].tick_params(axis='x',rotation=35);axes[0].set_ylabel('Total cycles');axes[0].set_title('(a) End-to-end comparison')
axes[1].bar(comparison.ratio,comparison.memory_stall_cycles,color=['#F28E2B']*5+['#59A14F']);axes[1].tick_params(axis='x',rotation=35);axes[1].set_ylabel('Memory stall cycles');axes[1].set_title('(b) Memory stall')
plt.tight_layout();plt.savefig(FIG/'exp2_static_vs_stage_best.pdf',bbox_inches='tight');plt.show();display(comparison)
print(f'Best fixed vs per-stage gap: {(fixed_best/stage_oracle-1)*100:.2f}%')
"""),
md("""
逐阶段分别选择最优比例得到的上界为 **102,736 cycles**，比全局最优固定 8:8:8 再低 **9.16%**。4:14:6 为 124,016 cycles，比 8:8:8 慢 10.59%。逐阶段最优是切换潜力的上界，不代表零开销动态实现结果。
"""),
md("""
## 4. 固定比例对不同阶段的失配
"""),
code("""
rows=[]
for a,w,o in [(8,8,8),(4,14,6),(2,11,11)]:
    q=sweep[(sweep.ia_banks==a)&(sweep.weight_banks==w)&(sweep.oa_banks==o)].merge(best[['layer','total_cycles']],on='layer',suffixes=('','_best'));q['slowdown']=q.total_cycles/q.total_cycles_best;q['ratio']=f'{a}:{w}:{o}';rows.append(q)
penalty=pd.concat(rows);penalty.to_csv(FIG/'exp2_per_stage_penalty.csv',index=False)
fig,ax=plt.subplots(figsize=(14,5));p=penalty.pivot(index='layer',columns='ratio',values='slowdown').loc[ordered.layer];p.plot(kind='bar',ax=ax,width=.85);ax.axhline(1,color='black',lw=.8);ax.set_ylabel('Slowdown vs per-stage best');ax.set_xlabel('');ax.tick_params(axis='x',rotation=60,labelsize=8);ax.legend(title='Fixed ratio')
plt.tight_layout();plt.savefig(FIG/'exp2_per_stage_slowdown.pdf',bbox_inches='tight');plt.show()
display(p.describe().loc[['mean','max']].round(3))
"""),
md("""
固定 8:8:8 相对逐阶段最优的几何平均 slowdown 为 **1.151×**，最差 Router 为 **2.25×**；4:14:6 的几何平均为 **1.248×**，Router 最差达到 **4.25×**。对 Weight-heavy Expert 有利的比例可能严重伤害 Router 或 Attention 阶段。
"""),
md("""
## 5. Hotspot、Idle Bank 与队列压力
"""),
code("""
both=(sweep.hotspot_bank_ratio>0)&(sweep.idle_bank_ratio>0)
fig,axes=plt.subplots(1,2,figsize=(11,4.2));axes[0].scatter(sweep.idle_bank_ratio,sweep.hotspot_bank_ratio,c=sweep.memory_stall_cycles,s=10,cmap='magma',alpha=.55);axes[0].set_xlabel('Idle Bank ratio');axes[0].set_ylabel('Hotspot Bank ratio');axes[0].set_title('(a) Hotspot and idle coexist')
axes[1].scatter(sweep.max_queue_depth,sweep.memory_stall_cycles,s=10,alpha=.45,color='#4E79A7');axes[1].set_xlabel('Max queue depth');axes[1].set_ylabel('Memory stall cycles');axes[1].set_title('(b) Queue pressure vs stall')
plt.tight_layout();plt.savefig(FIG/'exp2_bank_pathology.pdf',bbox_inches='tight');plt.show()
print(f'Hotspot nonzero: {(sweep.hotspot_bank_ratio>0).mean():.2%}')
print(f'Idle nonzero: {(sweep.idle_bank_ratio>0).mean():.2%}')
print(f'Both: {both.mean():.2%}')
print('Stall/queue correlation:',sweep.memory_stall_cycles.corr(sweep.max_queue_depth))
"""),
md("""
在全部阶段×静态比例点中，88.83% 出现 hotspot，10.98% 出现 idle Bank，10.57% 同时出现二者。Memory stall 与最大队列深度的相关系数为 **0.605**。这支持“问题不仅是总 Bank 数不足，而是所有权失配导致部分 Bank 排队、部分 Bank 空闲”。

## 实验 2 最终判断

- **可以证明**：各阶段最优 Bank 比例显著不同；最优全局固定比例仍比逐阶段上界慢 9.16%；固定比例会对部分阶段造成 2–4 倍级局部退化；hotspot 与 idle Bank 能同时出现。
- **不能直接证明**：动态架构一定获得完整 9.16% 收益。该数值没有包含映射查询、分配和切换开销，必须由 exp4/exp5 的真实动态结果验证。
- **不能沿用**：DATE1 的 4:14:6 最优结论。DATE2 当前模型下的全局最优固定比例是 8:8:8。
""")]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
(ROOT/'fig/exp2.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(ROOT/'fig/exp2.ipynb')
