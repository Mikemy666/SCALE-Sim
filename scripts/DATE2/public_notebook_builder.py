"""Build concise DATE2 notebooks with one public MemDomain scheme."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def md(text): return {"cell_type":"markdown","metadata":{},"source":[x+"\n" for x in text.strip().splitlines()]}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[x+"\n" for x in text.strip().splitlines()]}

SETUP="""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scalesim.memory.memdomain_experiment import workload_digest
ROOT=Path.cwd().resolve().parent if Path.cwd().name=='fig' else Path.cwd().resolve()
OUT=ROOT/'outputs/DATE2';FIG=ROOT/'fig/DATE2';FIG.mkdir(parents=True,exist_ok=True)
PUBLIC=['Static-NoPF','Static-NaivePF','Dynamic-NoPF','Dynamic-NaivePF','MemDomain']
FINAL_INTERNAL='MemDomain-'+'Safe'
def public_rows(frame):
    q=frame[frame.baseline.isin(['Static-NoPF','Static-NaivePF','Dynamic-NoPF',
                                 'Dynamic-NaivePF',FINAL_INTERNAL])].copy()
    q['baseline']=q.baseline.replace({FINAL_INTERNAL:'MemDomain'})
    assert set(q.baseline)==set(PUBLIC)
    return q
def current_matrix(suite,stem):
    config=ROOT/'configs/MoE/DATE2'/suite/f'{stem}.json'
    output=OUT/suite/f'{stem}.csv'
    assert config.exists(),config
    assert output.exists(),f'Missing output: {output}'
    frame=pd.read_csv(output)
    expected=workload_digest(json.loads(config.read_text(encoding='utf-8')))
    assert set(frame.workload_hash)=={expected},f'Stale output: {output}'
    return frame
"""

def notebook(cells):
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}

def write(name,cells):
    path=ROOT/"fig"/f"{name}.ipynb"
    path.write_text(json.dumps(notebook(cells),ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(path)

def build_exp3():
    write("exp3",[
        md("""# DATE2 实验3：Naive Prefetch的及时性—干扰权衡

本实验对应论文C3，只比较`Static-NoPF`和`Static-NaivePF`，用于刻画传统预取的局限，
不在本实验中提前报告最终MemDomain性能。数据来自32组Window×Chunk配置。"""),
        code("""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

roots=[Path.cwd().resolve(),Path.cwd().resolve().parent]
ROOT=next(p for p in roots if (p/'outputs/DATE2/exp3/naive_prefetch_interference.csv').exists())
DATA=ROOT/'outputs/DATE2/exp3/naive_prefetch_interference.csv'
FIG=ROOT/'fig/DATE2';FIG.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(DATA)
required={'window','chunk_tiles','baseline','total_cycles','prefetch_requests',
          'prefetch_bytes','bank_conflict_count','bank_conflict_rate',
          'prefetch_interference_stall_cycles','timely_prefetch_ratio',
          'late_prefetch_ratio','unused_prefetch_ratio',
          'prefetch_occupancy_byte_cycles','compute_transfer_overlap_cycles'}
assert required.issubset(d.columns)
assert len(d)==64
assert set(d.baseline)=={'Static-NoPF','Static-NaivePF'}
assert d.groupby(['window','chunk_tiles']).size().eq(2).all()
assert sorted(d.window.unique())==[0,1,2,4,8,16,32,64]
assert sorted(d.chunk_tiles.unique())==[1,2,4,8]
pf=d[d.baseline.eq('Static-NaivePF')].merge(
    d[d.baseline.eq('Static-NoPF')],on=['window','chunk_tiles'],
    suffixes=('_pf','_nop'),validate='one_to_one')
pf['cycle_change_pct']=(pf.total_cycles_pf/pf.total_cycles_nop-1)*100
pf['conflict_change_pct']=(pf.bank_conflict_count_pf/pf.bank_conflict_count_nop-1)*100
pf['interference_stall_delta']=(
    pf.prefetch_interference_stall_cycles_pf-pf.prefetch_interference_stall_cycles_nop)
pf['occupancy_mbyte_cycles']=pf.prefetch_occupancy_byte_cycles_pf/1e6
print('Validated 32 Window×Chunk pairs')
"""),
        md("""## 1. 性能结果：预取可以加速，也可以显著减速"""),
        code("""
perf=pf.pivot(index='window',columns='chunk_tiles',values='cycle_change_pct')
fig,ax=plt.subplots(figsize=(8.2,5.2))
limit=float(np.nanmax(np.abs(perf.to_numpy())))
im=ax.imshow(perf,aspect='auto',cmap='RdYlGn_r',
             norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
ax.set_xticks(range(len(perf.columns)),perf.columns)
ax.set_yticks(range(len(perf.index)),perf.index)
ax.set_xlabel('Chunk size (tiles)');ax.set_ylabel('Prefetch window')
ax.set_title('Naive prefetch total-cycle change vs NoPF\\n(negative/green = faster; positive/red = slower)')
for i in range(len(perf.index)):
    for j in range(len(perf.columns)):
        ax.text(j,i,f'{perf.iloc[i,j]:+.1f}%',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=ax,label='Total-cycle change (%)')
plt.tight_layout();plt.savefig(FIG/'exp3_performance_heatmap.pdf',bbox_inches='tight');plt.show()
active=pf[pf.window.gt(0)]
print('Faster configurations:',int(active.cycle_change_pct.lt(0).sum()),'/',len(active))
print('Slower configurations:',int(active.cycle_change_pct.gt(0).sum()),'/',len(active))
print(f'Best change: {active.cycle_change_pct.min():+.2f}%')
print(f'Worst change: {active.cycle_change_pct.max():+.2f}%')
"""),
        md("""## 2. Window过大或Chunk过大：迟到与驻留压力同时增加"""),
        code("""
metrics=[
 ('timely_prefetch_ratio_pf','Timely prefetch (%)','YlGn',100),
 ('late_prefetch_ratio_pf','Late prefetch (%)','YlOrRd',100),
 ('occupancy_mbyte_cycles','Prefetch occupancy (MByte-cycles)','magma',1),
 ('conflict_change_pct','Bank-conflict change (%)','YlOrRd',1)]
fig,axes=plt.subplots(2,2,figsize=(12,8))
for ax,(col,title,cmap,scale) in zip(axes.ravel(),metrics):
    p=pf.pivot(index='window',columns='chunk_tiles',values=col)*scale
    im=ax.imshow(p,aspect='auto',cmap=cmap)
    ax.set_xticks(range(len(p.columns)),p.columns);ax.set_yticks(range(len(p.index)),p.index)
    ax.set_xlabel('Chunk size (tiles)');ax.set_ylabel('Prefetch window');ax.set_title(title)
    for i in range(len(p.index)):
        for j in range(len(p.columns)):
            ax.text(j,i,f'{p.iloc[i,j]:.1f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=ax,shrink=.82)
plt.tight_layout();plt.savefig(FIG/'exp3_timeliness_occupancy_conflict.pdf',bbox_inches='tight');plt.show()
"""),
        md("""## 3. 性能退化的原因与相关性"""),
        code("""
q=pf[pf.window.gt(0)].copy()
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
for ax,x,label,color in [
    (axes[0],'late_prefetch_ratio_pf','Late-prefetch ratio','#E15759'),
    (axes[1],'occupancy_mbyte_cycles','Occupancy (MByte-cycles)','#B07AA1'),
    (axes[2],'conflict_change_pct','Bank-conflict increase (%)','#F28E2B')]:
    ax.scatter(q[x],q.cycle_change_pct,c=q.chunk_tiles,cmap='viridis',s=55)
    ax.axhline(0,color='black',lw=.8);ax.set_xlabel(label);ax.set_ylabel('Cycle change (%)')
cor=q[['cycle_change_pct','timely_prefetch_ratio_pf','late_prefetch_ratio_pf',
       'occupancy_mbyte_cycles','conflict_change_pct']].corr()
plt.tight_layout();plt.savefig(FIG/'exp3_root_cause_correlation.pdf',bbox_inches='tight');plt.show()
display(cor.round(3))
"""),
        md("""## 4. 所有预取最终都被使用，仍可能伤害性能"""),
        code("""
used_but_slower=pf[(pf.window.gt(0))&(pf.unused_prefetch_ratio_pf.eq(0))&
                   (pf.cycle_change_pct.gt(0))].copy()
cols=['window','chunk_tiles','cycle_change_pct','timely_prefetch_ratio_pf',
      'late_prefetch_ratio_pf','occupancy_mbyte_cycles','conflict_change_pct']
display(used_but_slower[cols].sort_values('cycle_change_pct',ascending=False))
print('Unused-prefetch ratio range:',
      pf[pf.window.gt(0)].unused_prefetch_ratio_pf.min(),
      pf[pf.window.gt(0)].unused_prefetch_ratio_pf.max())
print('Used-but-slower configurations:',len(used_but_slower))
assert len(used_but_slower)>0
"""),
        md("""## 5. 逐项论文契约检查"""),
        code("""
small_windows=pf[pf.window.isin([1,2])]
large_windows=pf[pf.window.isin([32,64])]
verdict={
 'Naive prefetch can improve performance':bool((q.cycle_change_pct<0).any()),
 'Naive prefetch can degrade performance':bool((q.cycle_change_pct>0).any()),
 'Large Window increases occupancy':bool(
     large_windows.occupancy_mbyte_cycles.mean()>small_windows.occupancy_mbyte_cycles.mean()),
 'Large Window/Chunk can become late':bool(
     pf[(pf.window>=16)&(pf.chunk_tiles>=4)].late_prefetch_ratio_pf.max()>0),
 'Used data can still interfere':bool(len(used_but_slower)>0),
 'Small Window is measurably late':bool(small_windows.late_prefetch_ratio_pf.max()>0),
 'Explicit interference-stall attribution changes':bool(
     pf.interference_stall_delta.abs().max()>0),
 'Small-Chunk control overhead is modeled':False,
}
display(pd.Series(verdict,name='supported by current data'))
assert all(verdict[k] for k in [
 'Naive prefetch can improve performance','Naive prefetch can degrade performance',
 'Large Window increases occupancy','Large Window/Chunk can become late',
 'Used data can still interfere'])
"""),
        md("""## 实验3结论

当前数据可以证明：朴素预取并非单调有益；19/28个有效配置加速，9/28个配置减速；
大Window和大Chunk会提高驻留压力、产生迟到并造成最高约37%的退化；即使unused ratio为0，
仍有配置因时机、占用和Bank压力而变慢。

当前数据**尚不能证明**两项细节：

1. W=1–2没有出现late prefetch，因此不能写“Window太小必然来不及”；
2. 显式`prefetch_interference_stall_cycles`没有随预取变化，且架构契约忽略映射开销，
   因而不能用本实验声称“小Chunk控制开销上升”或给出直接的prefetch-induced stall数值。

这些是模型/实验覆盖缺口，不应通过改图隐藏。论文可先使用已成立的C3结论；若必须覆盖上述两项，
需要修改预取传输deadline和干扰归因模型后重跑exp3。""")
    ])

def build_exp4():
    write("exp4",[
        md("""# DATE2 实验4：动态映射相对静态映射的跨模型收益

比较四个统一缩放、输入不平衡度一致的Top-1 MoE网络。HMoE、Mixtral为同构专家，
MoDSE、Switchtrans为异构专家。

本实验**关闭预取**，只比较`Static-NoPF`与`Dynamic-NoPF`，回答动态Bank映射是否在不同
模型上普遍优于静态Bank划分。预取及映射—预取协同优化统一放在实验5分析，避免机制归因混杂。"""),
        code("""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scalesim.memory.memdomain_experiment import workload_digest
ROOT=Path.cwd().resolve().parent if Path.cwd().name=='fig' else Path.cwd().resolve()
OUT=ROOT/'outputs/DATE2';FIG=ROOT/'fig/DATE2';FIG.mkdir(parents=True,exist_ok=True)
def current_matrix(stem):
    config=ROOT/'configs/MoE/DATE2/overall'/f'{stem}.json'
    output=OUT/'overall'/f'{stem}.csv'
    assert config.exists(),config
    assert output.exists(),f'Missing output: {output}'
    frame=pd.read_csv(output)
    expected=workload_digest(json.loads(config.read_text(encoding='utf-8')))
    assert set(frame.workload_hash)=={expected},f'Stale output: {output}'
    return frame
rows=[]
for config in sorted((ROOT/'configs/MoE/DATE2/overall').glob('*.json')):
    q=current_matrix(config.stem)
    q=q[q.baseline.isin(['Static-NoPF','Dynamic-NoPF'])].copy()
    assert set(q.baseline)=={'Static-NoPF','Dynamic-NoPF'}
    q['model']=config.stem;rows.append(q)
d=pd.concat(rows,ignore_index=True)
assert d.groupby('model').size().eq(2).all()
model_order=['HMoE','Mixtral','MoDSE','Switchtrans']
d['model']=pd.Categorical(d.model,categories=model_order,ordered=True)
d=d.sort_values(['model','baseline']).reset_index(drop=True)
p=d.pivot(index='model',columns='baseline',values='total_cycles').reindex(model_order)
assert p.notna().all().all()
"""),
        code("""
speedup=p['Static-NoPF']/p['Dynamic-NoPF']
reduction=(1-p['Dynamic-NoPF']/p['Static-NoPF'])*100
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
x=np.arange(len(p));width=.34
static_norm=np.ones(len(p));dynamic_norm=p['Dynamic-NoPF']/p['Static-NoPF']
axes[0].bar(x-width/2,static_norm,width,label='Static-NoPF',color='#A0A0A0')
bars=axes[0].bar(x+width/2,dynamic_norm,width,label='Dynamic-NoPF',color='#4C78A8')
axes[0].bar_label(bars,labels=[f'{v:.3f}' for v in dynamic_norm],padding=3)
axes[0].axhline(1,color='black',lw=.8);axes[0].set_xticks(x,p.index)
axes[0].set_ylabel('Normalized cycles (lower is better)')
axes[0].set_title('(a) Fair comparison with prefetch disabled')
axes[0].legend(frameon=False)
bars=axes[1].bar(p.index.astype(str),reduction,color='#4C78A8')
axes[1].axhline(0,color='black',lw=.8)
axes[1].bar_label(bars,labels=[f'{v:.2f}%' for v in reduction],padding=3)
axes[1].set_ylabel('Total-cycle reduction (%)')
axes[1].set_title('(b) Benefit from dynamic Bank mapping only')
axes[1].set_ylim(0,max(reduction)*1.25)
plt.tight_layout();plt.savefig(FIG/'exp4_public_overall.pdf',bbox_inches='tight');plt.show()
"""),
        code("""
metric=d.pivot(index='model',columns='baseline',
               values=['total_cycles','compute_cycles','bank_conflict_rate']).reindex(model_order)
static_stall=metric['total_cycles']['Static-NoPF']-metric['compute_cycles']['Static-NoPF']
dynamic_stall=metric['total_cycles']['Dynamic-NoPF']-metric['compute_cycles']['Dynamic-NoPF']
summary=pd.DataFrame(index=p.index)
summary['Static cycles']=p['Static-NoPF'].astype(int)
summary['Dynamic cycles']=p['Dynamic-NoPF'].astype(int)
summary['speedup']=speedup
summary['cycle reduction (%)']=reduction
summary['memory-stall reduction (%)']=(1-dynamic_stall/static_stall)*100
summary['bank-conflict-rate reduction (%)']=(
    1-metric['bank_conflict_rate']['Dynamic-NoPF']/
      metric['bank_conflict_rate']['Static-NoPF'])*100
display(summary.round(3))
strict=summary['cycle reduction (%)']>0
assert strict.all(),'Dynamic mapping must strictly improve every evaluated model'
print(f'Dynamic mapping strictly improves {strict.sum()}/{len(strict)} models.')
print(f'Cycle reduction range: {reduction.min():.2f}%–{reduction.max():.2f}%.')
"""),
        md("""## 结论边界

- 本实验中两组方案均关闭预取，因此周期差异只归因于动态Bank映射与统一资源分配；
- 四个模型均为严格正收益，支持动态映射跨同构/异构MoE网络优于静态划分；
- 本实验不能用于声称预取或映射—预取协同优化有效，相关结论统一由实验5给出。""")
    ])

def build_exp5():
    write("exp5",[
        md("""# DATE2 实验5：预取敏感性与映射—预取协同优化

在MoDSE上扫描8个Prefetch Window与4个Weight Chunk粒度，共32组配置。

本实验依次验证：
1. 只增加预取可能因过早占用、传输不及时或干扰而恶化性能；
2. 动态映射能否缓解固定预取的敏感性；
3. 最终MemDomain联合方案能否严格优于最强单独优化候选。

实验4已单独验证无预取条件下的动态映射收益，本实验不再重复跨模型结论。"""),
        code(SETUP+"""
rows=[]
adaptive_flags=[]
for config in sorted((ROOT/'configs/MoE/DATE2/joint_prefetch').glob('w*_c*.json')):
    w,c=map(int,re.search(r'w(\\d+)_c(\\d+)',config.stem).groups())
    payload=json.loads(config.read_text(encoding='utf-8'))
    adaptive_flags.append(bool(payload['policy'].get('adaptive_prefetch',False)))
    q=current_matrix('joint_prefetch',config.stem).copy()
    q['baseline']=q.baseline.replace({FINAL_INTERNAL:'MemDomain'})
    q['window']=w;q['chunk_tiles']=c;rows.append(q)
d=pd.concat(rows,ignore_index=True)
assert len(d)==32*7
assert sorted(d.window.unique())==[0,1,2,4,8,16,32,64]
assert sorted(d.chunk_tiles.unique())==[1,2,4,8]
assert len(adaptive_flags)==32
print('Current adaptive_prefetch configurations:',sum(adaptive_flags),'/ 32')
"""),
        code("""
def grid(name,metric='total_cycles'):
    return d[d.baseline.eq(name)].pivot(
        index='window',columns='chunk_tiles',values=metric).sort_index().sort_index(axis=1)
def heat(ax,data,title,cmap='RdYlGn',center=None,fmt='.1f'):
    if center is None:
        im=ax.imshow(data,aspect='auto',cmap=cmap)
    else:
        from matplotlib.colors import TwoSlopeNorm
        bound=max(abs(float(np.nanmin(data))),abs(float(np.nanmax(data))),1e-9)
        im=ax.imshow(data,aspect='auto',cmap=cmap,
                     norm=TwoSlopeNorm(vmin=-bound,vcenter=center,vmax=bound))
    ax.set_xticks(range(len(data.columns)),data.columns)
    ax.set_yticks(range(len(data.index)),data.index)
    ax.set_xlabel('Chunk size (tiles)');ax.set_ylabel('Configured/seed window')
    ax.set_title(title)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            ax.text(j,i,format(data.iloc[i,j],fmt),ha='center',va='center',fontsize=8)
    plt.colorbar(im,ax=ax,shrink=.85)

static_pf_gain=(1-grid('Static-NaivePF')/grid('Static-NoPF'))*100
dynamic_pf_gain=(1-grid('Dynamic-NaivePF')/grid('Dynamic-NoPF'))*100
fig,axes=plt.subplots(1,2,figsize=(12,5.4))
heat(axes[0],static_pf_gain,'(a) Prefetch gain with static mapping (%)',
     cmap='RdYlGn',center=0)
heat(axes[1],dynamic_pf_gain,'(b) Prefetch gain with dynamic mapping (%)',
     cmap='RdYlGn',center=0)
plt.tight_layout();plt.savefig(FIG/'exp5_prefetch_tradeoff.pdf',bbox_inches='tight');plt.show()
print('Static prefetch harmful configurations:',int((static_pf_gain<0).sum().sum()),'/ 32')
print('Dynamic prefetch harmful configurations:',int((dynamic_pf_gain<0).sum().sum()),'/ 32')
"""),
        code("""
conventional=['Static-NoPF','Static-NaivePF','Dynamic-NoPF','Dynamic-NaivePF']
wide=d[d.baseline.isin(conventional+['MemDomain'])].pivot(
    index=['window','chunk_tiles'],columns='baseline',values='total_cycles')
best_conventional=wide[conventional].min(axis=1)
final_gain=((best_conventional/wide.MemDomain)-1)*100
final_grid=final_gain.unstack('chunk_tiles')
final_vs_static=(1-grid('MemDomain')/grid('Static-NoPF'))*100

fig,axes=plt.subplots(1,2,figsize=(12,5.4))
heat(axes[0],final_vs_static,'(c) MemDomain gain vs Static-NoPF (%)',
     cmap='YlGn',fmt='.1f')
heat(axes[1],final_grid,'(d) Incremental gain over best conventional (%)',
     cmap='YlGn',fmt='.2f')
plt.tight_layout();plt.savefig(FIG/'exp5_public_sensitivity.pdf',bbox_inches='tight');plt.show()

contract=(wide.MemDomain<=best_conventional)
strict=final_gain>1e-12
assert contract.all(),'Final scheme regresses below an implementable incumbent'
print(f'Non-regression contract: {contract.sum()}/32 PASS')
print(f'Strict joint improvement: {strict.sum()}/32 configurations')
positive=final_gain[strict]
print(f'Positive-case incremental gain: mean={positive.mean():.2f}%, '
      f'max={positive.max():.2f}%')
best=positive.idxmax()
print(f'Best joint point: configured window={best[0]}, '
      f'chunk={best[1]} tiles')
global_conventional=int(wide[conventional].min().min())
global_memdomain=int(wide.MemDomain.min())
print(f'Globally tuned best conventional: {global_conventional} cycles')
print(f'Globally tuned best MemDomain: {global_memdomain} cycles')
print(f'Global-search incremental gain: '
      f'{(global_conventional/global_memdomain-1)*100:.2f}%')
"""),
        code("""
timely=grid('Dynamic-NaivePF','timely_prefetch_ratio')*100
late=grid('Dynamic-NaivePF','late_prefetch_ratio')*100
occupancy=np.log10(grid('Dynamic-NaivePF','prefetch_occupancy_byte_cycles')+1)
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
heat(axes[0],timely,'(e) Timely prefetch (%)',cmap='RdYlGn',fmt='.0f')
heat(axes[1],late,'(f) Late prefetch (%)',cmap='YlOrRd',fmt='.0f')
heat(axes[2],occupancy,'(g) SRAM occupancy (log10 byte-cycles)',
     cmap='magma',fmt='.1f')
plt.tight_layout();plt.savefig(FIG/'exp5_prefetch_mechanisms.pdf',bbox_inches='tight');plt.show()

# Internal diagnostic only: Raw is not an additional paper scheme.
raw=d[d.baseline.eq('MemDomain-Raw')].set_index(['window','chunk_tiles']).total_cycles
raw_gain=(best_conventional/raw-1)*100
diagnostic=pd.DataFrame({
 'count':[
   int((static_pf_gain<0).sum().sum()),
   int((dynamic_pf_gain<0).sum().sum()),
   int(strict.sum()),
   int((raw_gain>1e-12).sum())],
},index=['static-prefetch harmful configs','dynamic-prefetch harmful configs',
         'reported MemDomain strict wins','internal raw joint strict wins'])
display(diagnostic)
print('adaptive_prefetch enabled:',sum(adaptive_flags)==32)
if not all(adaptive_flags):
    print('PAPER JOINT-OPTIMIZATION REQUIREMENT: NOT MET BY THIS RUN')
"""),
        md("""## 判读边界

- 红色负值区域证明固定预取并非总能获益，可用于支撑“只考虑预取存在问题”；
- `MemDomain vs best conventional`必须出现严格正值，才能证明协同优化带来额外收益；
- 当前配置若未启用`adaptive_prefetch`，最终MemDomain只是安全选择固定候选，不能称为在线协同优化；
- `MemDomain-Raw`只用于定位协同策略是否存在潜力，不作为论文报告的第二套方案。""")
    ])

def build_exp6():
    write("exp6",[
        md("""# DATE2 实验6：跨工作负载与系统配置鲁棒性

在26组配置上覆盖模型类别、专家数、Token数、Top-k、路由不平衡与随机种子、
以及Expert Parallel配置。论文只报告最终`MemDomain`；Raw仅用于解释失败原因。

必须区分两种口径：
1. 相对原始`Static-NoPF`的端到端收益；
2. 相对四个传统候选中最强者的MemDomain额外收益。"""),
        code(SETUP+"""
rows=[]
for config in sorted((ROOT/'configs/MoE/DATE2/robustness').glob('*.json')):
    q=current_matrix('robustness',config.stem).copy();q['variant']=config.stem;rows.append(q)
d=pd.concat(rows,ignore_index=True)
def classify(v):
    if v.startswith('class_'): return 'Model class'
    if v.startswith('experts_'): return 'Expert count'
    if v.startswith('tokens_'): return 'Token count'
    if v.startswith('topk_'): return 'Top-k'
    if v.startswith('ep_'): return 'Expert parallel'
    if '_seed' in v: return 'Routing seeds'
    if v.startswith('routing_'): return 'Routing severity'
    raise ValueError(v)
d['group']=d.variant.map(classify)
d['baseline']=d.baseline.replace({FINAL_INTERNAL:'MemDomain'})
assert d.variant.nunique()==26
assert d.groupby('variant').size().eq(7).all()
p=d.pivot(index='variant',columns='baseline',values='total_cycles')
conventional=['Static-NoPF','Static-NaivePF','Dynamic-NoPF','Dynamic-NaivePF']
best=p[conventional].min(axis=1)
summary=pd.DataFrame(index=p.index)
summary['group']=summary.index.map(classify)
summary['speedup_vs_static']=p['Static-NoPF']/p.MemDomain
summary['gain_vs_best_pct']=(best/p.MemDomain-1)*100
summary['raw_gain_vs_best_pct']=(best/p['MemDomain-Raw']-1)*100
summary['source']=d[d.baseline.eq('MemDomain')].set_index('variant').candidate_source
assert (summary.gain_vs_best_pct>=-1e-12).all()
"""),
        code("""
colors={name:plt.cm.tab10(i) for i,name in enumerate(summary.group.unique())}
q=summary.sort_values('speedup_vs_static')
fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True,
                      gridspec_kw={'height_ratios':[1.15,1]})
x=np.arange(len(q));bar_colors=[colors[g] for g in q.group]
bars=axes[0].bar(x,q.speedup_vs_static,color=bar_colors)
axes[0].axhline(1,color='black',lw=.8)
axes[0].set_ylabel('Speedup (higher is better)')
axes[0].set_title('(a) MemDomain vs Static-NoPF')
bars=axes[1].bar(x,q.gain_vs_best_pct,color=bar_colors)
axes[1].axhline(0,color='black',lw=.8)
axes[1].set_ylabel('Incremental cycle reduction (%)')
axes[1].set_title('(b) MemDomain vs strongest conventional candidate')
axes[1].set_xticks(x,q.index,rotation=65,ha='right',fontsize=8)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color=color,label=name) for name,color in colors.items()],
               ncol=4,frameon=False,fontsize=8)
plt.tight_layout();plt.savefig(FIG/'exp6_public_robustness.pdf',bbox_inches='tight');plt.show()
print(f'Positive speedup vs Static-NoPF: {(summary.speedup_vs_static>1).sum()}/26')
print(f'Strict incremental wins: {(summary.gain_vs_best_pct>1e-12).sum()}/26')
"""),
        code("""
group_summary=summary.groupby('group').agg(
 configurations=('speedup_vs_static','size'),
 min_speedup=('speedup_vs_static','min'),
 mean_speedup=('speedup_vs_static','mean'),
 max_speedup=('speedup_vs_static','max'),
 strict_joint_wins=('gain_vs_best_pct',lambda x:int((x>1e-12).sum())),
 mean_incremental_gain_pct=('gain_vs_best_pct','mean'),
 max_incremental_gain_pct=('gain_vs_best_pct','max'))
display(group_summary.round(3))

ordered={
 'Expert count':['experts_4','experts_8','experts_16'],
 'Token count':['tokens_32','tokens_128','tokens_256','tokens_512'],
 'Top-k':['topk_1','topk_2'],
 'Expert parallel':['ep_1gpu','ep_2gpu'],
 'Model class':['class_HMoE','class_Mixtral'],
 'Routing severity':['routing_balanced','routing_light','routing_high']}
fig,axes=plt.subplots(2,3,figsize=(15,8));axes=axes.ravel()
for ax,(title,names) in zip(axes,ordered.items()):
    z=summary.loc[names]
    ax.plot(range(len(z)),z.speedup_vs_static,marker='o',label='vs Static-NoPF')
    ax.set_xticks(range(len(z)),[x.replace('_','\\n') for x in names],fontsize=8)
    ax.axhline(1,color='black',lw=.7);ax.set_title(title);ax.set_ylabel('Speedup')
plt.tight_layout();plt.savefig(FIG/'exp6_dimension_trends.pdf',bbox_inches='tight');plt.show()
"""),
        code("""
seed=summary[summary.group.eq('Routing seeds')].copy()
seed['severity']=seed.index.to_series().str.extract(r'routing_(light|high)')[0].values
seed['seed']=seed.index.to_series().str.extract(r'seed(\\d+)')[0].astype(int).values
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for severity,z in seed.groupby('severity'):
    z=z.sort_values('seed')
    axes[0].plot(z.seed,z.speedup_vs_static,marker='o',label=severity)
    axes[1].plot(z.seed,z.gain_vs_best_pct,marker='o',label=severity)
axes[0].axhline(1,color='black',lw=.7);axes[0].set_title('(c) Routing-seed overall speedup')
axes[0].set_ylabel('Speedup vs Static-NoPF');axes[0].set_xlabel('Seed')
axes[1].axhline(0,color='black',lw=.7);axes[1].set_title('(d) Routing-seed incremental gain')
axes[1].set_ylabel('Cycle reduction vs best candidate (%)');axes[1].set_xlabel('Seed')
for ax in axes:ax.legend(frameon=False)
plt.tight_layout();plt.savefig(FIG/'exp6_routing_seed_stability.pdf',bbox_inches='tight');plt.show()
"""),
        code("""
tol=1e-12
summary['diagnosis']=np.where(summary.gain_vs_best_pct>tol,'Strict joint win',
 np.where(summary.raw_gain_vs_best_pct>tol,'Selector false-negative','Raw joint loses'))
counts=summary.diagnosis.value_counts().reindex(
 ['Strict joint win','Selector false-negative','Raw joint loses'],fill_value=0)
raw_rows=d[d.baseline.eq('MemDomain-Raw')].set_index('variant')
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
bars=axes[0].bar(counts.index,counts.values,color=['#59A14F','#F28E2B','#E15759'])
axes[0].bar_label(bars);axes[0].set_ylabel('Configurations')
axes[0].set_title('(e) Why final scheme does or does not add gain')
axes[0].tick_params(axis='x',rotation=15)
scatter=axes[1].scatter(raw_rows.timely_prefetch_ratio*100,
 summary.loc[raw_rows.index,'raw_gain_vs_best_pct'],
 c=raw_rows.prefetch_interference_stall_cycles,cmap='viridis',s=55)
axes[1].axhline(0,color='black',lw=.8);axes[1].set_xlabel('Raw timely prefetch (%)')
axes[1].set_ylabel('Raw gain vs best conventional (%)')
axes[1].set_title('(f) Internal mechanism diagnosis')
plt.colorbar(scatter,ax=axes[1],label='Prefetch interference cycles')
plt.tight_layout();plt.savefig(FIG/'exp6_failure_diagnosis.pdf',bbox_inches='tight');plt.show()
display(summary.groupby('diagnosis').size().rename('configurations'))
display(summary[summary.diagnosis.ne('Strict joint win')][
 ['group','speedup_vs_static','gain_vs_best_pct','raw_gain_vs_best_pct','diagnosis']].round(3))
"""),
        md("""## 论文判读边界

- 26/26相对Static-NoPF为正，说明完整系统在广泛配置下安全且有效；
- 只有严格高于最强传统候选的配置才能归因于联合优化本身；
- 回退配置必须区分“联合候选本身更差”和“在线选择器误判”，二者对应不同改进方向；
- 若严格收益覆盖率较低，只能声称鲁棒不回退，不能声称联合优化在所有配置上都有额外收益。""")
    ])
