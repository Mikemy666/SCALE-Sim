"""Create the standalone DATE2 exp3 naive-prefetch analysis notebook."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def md(t):return {"cell_type":"markdown","metadata":{},"source":[x+"\n" for x in t.strip().splitlines()]}
def code(t):return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[x+"\n" for x in t.strip().splitlines()]}
cells=[
md("""
# DATE2 实验 3：Naive Prefetch 的及时性与 Bank 干扰

主要数据源：`outputs/DATE2/exp3/naive_prefetch_interference.csv`。

为了获得完整 stall breakdown、coverage、mapping failures 和 peak occupancy，本 Notebook 还读取对应的 `outputs/DATE2/window_chunk/wW_cC.csv` 七基线矩阵，但只选择 `Static-NoPF` 和 `Static-NaivePF` 两个对照。

实验问题：固定距离、固定 Bank 集合的 Naive Prefetch 能否及时完成？Window 或 Chunk 过大时是否产生可观测的冲突和性能退化？
"""),
code("""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scalesim.memory.memdomain_experiment import workload_digest
roots=[Path.cwd().resolve(),Path.cwd().resolve().parent]
ROOT=next(p for p in roots if (p/'outputs/DATE2/exp3/naive_prefetch_interference.csv').exists())
OUT=ROOT/'outputs/DATE2';FIG=ROOT/'fig/DATE2';FIG.mkdir(parents=True,exist_ok=True)
summary=pd.read_csv(OUT/'exp3/naive_prefetch_interference.csv')
assert len(summary)==40
assert set(summary.window)=={0,1,2,4,8} and set(summary.chunk_tiles)=={1,2,4,8}
assert summary.groupby(['window','chunk_tiles']).size().eq(2).all()
assert set(summary.baseline)=={'Static-NoPF','Static-NaivePF'}
rows=[]
for path in sorted((OUT/'window_chunk').glob('w*_c*.csv')):
    window,chunk=map(int,re.search(r'w(\\d+)_c(\\d+)',path.stem).groups())
    data=pd.read_csv(path)
    config=json.loads((ROOT/f'configs/MoE/DATE2/window_chunk/{path.stem}.json').read_text())
    assert set(data.workload_hash)=={workload_digest(config)}, f'Stale matrix: {path}'
    for baseline in ('Static-NoPF','Static-NaivePF'):
        row=data[data.baseline==baseline].iloc[0].to_dict();row.update(window=window,chunk_tiles=chunk);rows.append(row)
detail=pd.DataFrame(rows)
assert len(detail)==40
print('Validated 20 Window x Chunk points with two measured controls each')
"""),
md("""
## 1. Naive Prefetch 相对 No Prefetch 的端到端变化
"""),
code("""
cycles=detail.pivot(index=['window','chunk_tiles'],columns='baseline',values='total_cycles').reset_index()
cycles['delta_cycles']=cycles['Static-NaivePF']-cycles['Static-NoPF']
cycles['change_percent']=(cycles['Static-NaivePF']/cycles['Static-NoPF']-1)*100
cycles['speedup']=cycles['Static-NoPF']/cycles['Static-NaivePF']
cycles.to_csv(FIG/'exp3_performance_delta.csv',index=False)
p=cycles.pivot(index='window',columns='chunk_tiles',values='change_percent').sort_index().sort_index(axis=1)
fig,axes=plt.subplots(1,2,figsize=(12,4.3));limit=max(abs(float(np.nanmin(p))),abs(float(np.nanmax(p))))
im=axes[0].imshow(p.values,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit);axes[0].set_xticks(range(len(p.columns)),p.columns);axes[0].set_yticks(range(len(p.index)),p.index);axes[0].set_xlabel('Chunk (tiles)');axes[0].set_ylabel('Window');axes[0].set_title('(a) NaivePF cycle change vs NoPF (%)');fig.colorbar(im,ax=axes[0])
for chunk,q in cycles.groupby('chunk_tiles'):axes[1].plot(q.window,q.change_percent,marker='o',label=f'C={chunk}')
axes[1].axhline(0,color='black',lw=.8);axes[1].set_xlabel('Window');axes[1].set_ylabel('Cycle change (%)');axes[1].set_title('(b) Window trend');axes[1].legend(ncol=2)
plt.tight_layout();plt.savefig(FIG/'exp3_performance_effect.pdf',bbox_inches='tight');plt.show()
display(cycles.sort_values('change_percent').round(4))
"""),
code("""
best_point=cycles.loc[cycles.change_percent.idxmin()]
worst_point=cycles.loc[cycles.change_percent.idxmax()]
print('Best NaivePF point:',best_point[['window','chunk_tiles','change_percent']].to_dict())
print('Worst NaivePF point:',worst_point[['window','chunk_tiles','change_percent']].to_dict())
assert cycles[cycles.window==0].delta_cycles.eq(0).all(), 'W=0 control must equal NoPF'
"""),
md("""
W=0 必须与 NoPF 完全相同。负的 `change_percent` 表示预取改善，正值表示退化；最佳/最差点由上方代码从当前数据计算，不在 Notebook 中硬编码旧结果。
"""),
md("""
## 2. Timely、Late 与 Coverage
"""),
code("""
naive=detail[detail.baseline=='Static-NaivePF'].sort_values(['window','chunk_tiles']).copy()
active=naive[naive.window>0]
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for chunk,q in naive.groupby('chunk_tiles'):
    axes[0].plot(q.window,q.prefetch_coverage,marker='o',label=f'C={chunk}')
    axes[1].plot(q.window,q.prefetch_occupancy_byte_cycles,marker='o',label=f'C={chunk}')
axes[0].set_xlabel('Window');axes[0].set_ylabel('Prefetch coverage');axes[0].set_title('(a) Coverage');axes[0].legend(ncol=2)
axes[1].set_xlabel('Window');axes[1].set_ylabel('Occupancy byte-cycles');axes[1].set_title('(b) SRAM occupancy');axes[1].legend(ncol=2)
plt.tight_layout();plt.savefig(FIG/'exp3_coverage_occupancy.pdf',bbox_inches='tight');plt.show()
print('Timely ratio range for W>0:',active.timely_prefetch_ratio.min(),active.timely_prefetch_ratio.max())
print('Late ratio range for W>0:',active.late_prefetch_ratio.min(),active.late_prefetch_ratio.max())
print('Coverage range for W>0:',active.prefetch_coverage.min(),active.prefetch_coverage.max())
"""),
code("""
timeliness=active.groupby('window').agg(timely_mean=('timely_prefetch_ratio','mean'),late_mean=('late_prefetch_ratio','mean'),coverage_mean=('prefetch_coverage','mean')).reset_index()
display(timeliness.round(4))
"""),
md("""
随着 Window 增大，合理的新时间轴应当使 timely ratio 上升、late ratio 下降。若仍然全部为 late，说明运行结果不是当前架构生成，或传输带宽/计算间隔仍不匹配。
"""),
md("""
## 3. Demand stall 与 Late-prefetch stall 分解
"""),
code("""
fig,axes=plt.subplots(1,2,figsize=(12,4.3))
for chunk,q in naive.groupby('chunk_tiles'):
    axes[0].plot(q.window,q.weight_load_stall_cycles,marker='o',label=f'C={chunk}')
    axes[1].plot(q.window,q.prefetch_miss_stall_cycles,marker='o',label=f'C={chunk}')
axes[0].set_xlabel('Window');axes[0].set_ylabel('Demand-load stall');axes[0].set_title('(a) Demand component');axes[0].legend(ncol=2)
axes[1].set_xlabel('Window');axes[1].set_ylabel('Late-prefetch stall');axes[1].set_title('(b) Late component');axes[1].legend(ncol=2)
plt.tight_layout();plt.savefig(FIG/'exp3_stall_breakdown.pdf',bbox_inches='tight');plt.show()
stall=naive[['window','chunk_tiles','weight_load_stall_cycles','prefetch_miss_stall_cycles','prefetch_interference_stall_cycles','total_cycles']]
stall.to_csv(FIG/'exp3_stall_breakdown.csv',index=False);display(stall)
"""),
md("""
随着 Window 增大，demand-load stall 略降，late-prefetch stall也略降，但没有变成 timely prefetch。Chunk 增大显著降低 demand stall，说明当前总周期对 Chunk 数量和请求粒度更敏感，而不是对预取及时性敏感。
"""),
md("""
## 4. Bank conflict、显式干扰与映射压力
"""),
code("""
fig,axes=plt.subplots(1,3,figsize=(14,4))
for chunk,q in naive.groupby('chunk_tiles'):
    axes[0].plot(q.window,q.bank_conflict_count,marker='o',label=f'C={chunk}')
    axes[1].plot(q.window,q.compute_transfer_overlap_cycles,marker='o',label=f'C={chunk}')
    axes[2].plot(q.window,q.mapping_failures,marker='o',label=f'C={chunk}')
axes[0].set_title('(a) Bank conflicts');axes[0].set_ylabel('Count');axes[1].set_title('(b) Compute-transfer overlap');axes[1].set_ylabel('Cycles');axes[2].set_title('(c) Mapping failures');axes[2].set_ylabel('Count')
for ax in axes:ax.set_xlabel('Window');ax.legend(fontsize=7)
plt.tight_layout();plt.savefig(FIG/'exp3_conflict_overlap_mapping.pdf',bbox_inches='tight');plt.show()
print('Explicit interference stall unique values:',sorted(naive.prefetch_interference_stall_cycles.unique()))
print('Conflict-count range:',naive.bank_conflict_count.min(),naive.bank_conflict_count.max())
"""),
md("""
## 实验 3 判断规则

- Window sweep 应同时覆盖 late-dominated、混合和 timely-dominated 区域。
- 若大 Window 提高及时性但增加 interference/occupancy，可支持预取时机权衡。
- 若某些 NaivePF 点慢于 NoPF，且退化与 interference、queue 或 conflict 同步，才能证明计算—预取 Bank 干扰。
- Chunk 结论必须联合 demand stall、mapping failures 和 occupancy，不能只看总周期。

Notebook 所有最佳点、最差点和及时性表均从当前 CSV 动态计算，避免再次引用旧架构结果。
""")]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
(ROOT/'fig/exp3.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(ROOT/'fig/exp3.ipynb')
