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

为了获得完整 stall breakdown、coverage、mapping failures 和 peak occupancy，本 Notebook 还读取对应的 `outputs/DATE2/window_chunk/wW_cC.csv` 七基线矩阵。及时性实验比较 `Static-NoPF` 与 `Static-NaivePF`；Bank-aware 分析比较 `Dynamic-NaivePF` 与 `MemDomain-Raw`。

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
    for baseline in ('Static-NoPF','Static-NaivePF','Dynamic-NaivePF','MemDomain-Raw','MemDomain-Safe'):
        row=data[data.baseline==baseline].iloc[0].to_dict();row.update(window=window,chunk_tiles=chunk);rows.append(row)
detail=pd.DataFrame(rows)
assert len(detail)==100
print('Validated 20 Window x Chunk points with five measured policies each')
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
active_cycles=cycles[cycles.window>0]
print(f"Best point: W={int(best_point.window)}, C={int(best_point.chunk_tiles)}, "
      f"{best_point.speedup:.3f}x speedup ({-best_point.change_percent:.2f}% fewer cycles)")
print('Regressions for W>0:',int((active_cycles.change_percent>0).sum()))
"""),
md("""
W=0 与 NoPF 完全相同，验证了关闭预取的控制路径。当前重跑数据中，所有 W>0 点均优于 NoPF；最佳点为 W=8、C=2，约 2.042× 加速（周期减少 51.03%）。因此该组结果证明 NaivePF 有效，但不能声称其 Bank 干扰已经大到抵消全部预取收益。
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
重跑结果呈现清晰的及时性转折：W=1 仍以 late 为主，W=2 进入混合区，W=4/W=8 基本成为 timely-dominated；W>0 的 coverage 均为 1。与此同时 occupancy 随 Window 和 Chunk 增大，因此“更早预取”并非没有 SRAM 容量代价。
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
随着 Window 增大，late-prefetch stall 被明显压低，与 timely ratio 的上升一致；这修正了旧结果中“预取始终来不及”的异常。Chunk 同时改变请求粒度与调度开销，必须与 occupancy、干扰和映射失败联合判断。
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
显式 interference stall 在 W>0 时约为 822–3296 cycles，说明计算与预取确实争用 Bank/传输资源；但 NaivePF 在本 sweep 中仍全部快于 NoPF，所以这里能证明“干扰存在”，不能证明“干扰必然导致端到端退化”。W=8、C=8 还出现 mapping failure，显示过大的 Window×Chunk 会产生容量/映射压力。
"""),
md("""
## 5. Bank-aware 调度是否总能优于动态朴素预取？
"""),
code("""
policy=detail[(detail.window>0)&detail.baseline.isin(['Dynamic-NaivePF','MemDomain-Raw'])].copy()
wide=policy.pivot(index=['window','chunk_tiles'],columns='baseline',
                  values=['total_cycles','prefetch_interference_stall_cycles','bank_conflict_count']).reset_index()
compare=pd.DataFrame({
    'window':wide['window'],
    'chunk_tiles':wide['chunk_tiles'],
    'cycle_change_percent':(wide[('total_cycles','MemDomain-Raw')]/wide[('total_cycles','Dynamic-NaivePF')]-1)*100,
    'interference_change_percent':(wide[('prefetch_interference_stall_cycles','MemDomain-Raw')]/
        wide[('prefetch_interference_stall_cycles','Dynamic-NaivePF')]-1)*100,
    'conflict_delta':wide[('bank_conflict_count','MemDomain-Raw')]-wide[('bank_conflict_count','Dynamic-NaivePF')]
})
compare.to_csv(FIG/'exp3_bankaware_vs_dynamic_naive.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,column,title in zip(
    axes,['cycle_change_percent','interference_change_percent','conflict_delta'],
    ['(a) Total-cycle change (%)','(b) Interference change (%)','(c) Bank-conflict delta']):
    table=compare.pivot(index='window',columns='chunk_tiles',values=column).sort_index().sort_index(axis=1)
    lim=max(abs(float(np.nanmin(table))),abs(float(np.nanmax(table))),1e-9)
    im=ax.imshow(table.values,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
    ax.set_xticks(range(len(table.columns)),table.columns);ax.set_yticks(range(len(table.index)),table.index)
    ax.set_xlabel('Chunk (tiles)');ax.set_ylabel('Window');ax.set_title(title);fig.colorbar(im,ax=ax)
plt.tight_layout();plt.savefig(FIG/'exp3_bankaware_vs_dynamic_naive.pdf',bbox_inches='tight');plt.show()
wins=compare[compare.cycle_change_percent<0]
print(f'MemDomain-Raw cycle wins: {len(wins)}/{len(compare)} points')
display(compare.round(3))
"""),
md("""
负值表示 MemDomain-Raw 优于 Dynamic-NaivePF。排除无预取的 W=0 控制点后，当前数据中 Raw 仅在 16 个点中的 4 个点取得总周期优势，优势集中在小 Chunk；例如 W=2、C=1 时总周期改善约 2.65%，显式干扰降低约 23.1%。较大 Chunk 下，Raw 虽常降低 conflict/interference，却可能因 Bank 选择和调度开销而变慢。

这说明 Bank-aware 机制有效但不是无条件最优：论文应使用 `MemDomain-Safe` 的回退选择保证最终方案不劣于候选基线，同时将“大 Chunk 下 Raw 调度开销”作为架构仍需优化的边界。
"""),
md("""
## 实验 3 结论

- Window sweep 应同时覆盖 late-dominated、混合和 timely-dominated 区域。
- 若大 Window 提高及时性但增加 interference/occupancy，可支持预取时机权衡。
- Window sweep 已覆盖 late-dominated、混合和 timely-dominated 区域，证明预取距离控制有效。
- 更大的 Window 提高及时性，但增加 occupancy；极端 W=8、C=8 出现映射失败。
- 显式 interference 非零，证明计算—预取资源争用存在；本组 NaivePF 未慢于 NoPF，不能夸大为端到端退化。
- MemDomain-Raw 在部分小 Chunk 点改善总周期、冲突和干扰，但并非全域最优；最终架构需要 Safe 回退。

所有图表和中间 CSV 均由当前矩阵动态生成，最佳点、胜点数量不依赖旧实验数据。
""")]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
(ROOT/'fig/exp3.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(ROOT/'fig/exp3.ipynb')
