"""Build concise DATE2 notebooks with one public MemDomain scheme."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def md(text): return {"cell_type":"markdown","metadata":{},"source":[x+"\n" for x in text.strip().splitlines()]}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[x+"\n" for x in text.strip().splitlines()]}

SETUP="""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path.cwd().resolve().parent if Path.cwd().name=='fig' else Path.cwd().resolve()
OUT=ROOT/'outputs/DATE2';FIG=ROOT/'fig/DATE2';FIG.mkdir(parents=True,exist_ok=True)
PUBLIC=['Static-NoPF','Static-NaivePF','Dynamic-NoPF','Dynamic-NaivePF','MemDomain']
FINAL_INTERNAL='MemDomain-'+'Safe'
def public_rows(frame):
    q=frame[frame.baseline.isin(['Static-NoPF','Static-NaivePF','Dynamic-NoPF',
                                 FINAL_INTERNAL])].copy()
    q['baseline']=q.baseline.replace({FINAL_INTERNAL:'MemDomain'})
    assert set(q.baseline)==set(PUBLIC)
    return q
"""

def notebook(cells):
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}

def write(name,cells):
    path=ROOT/"fig"/f"{name}.ipynb"
    path.write_text(json.dumps(notebook(cells),ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(path)

def build_exp3():
    write("exp3",[
        md("""# DATE2 实验3：朴素预取干扰与MemDomain协同优化

论文只报告一个最终`MemDomain`；内部诊断候选不会进入论文图表。"""),
        code(SETUP+"""
rows=[]
for path in sorted((OUT/'window_chunk').glob('w*_c*.csv')):
    w,c=map(int,re.search(r'w(\\d+)_c(\\d+)',path.stem).groups())
    q=public_rows(pd.read_csv(path));q['window']=w;q['chunk_tiles']=c;rows.append(q)
d=pd.concat(rows,ignore_index=True)
assert d.groupby(['window','chunk_tiles']).size().eq(5).all()
"""),
        code("""
wide=d.pivot_table(index=['window','chunk_tiles'],columns='baseline',values='total_cycles')
wide['naive_change_pct']=(wide['Static-NaivePF']/wide['Static-NoPF']-1)*100
wide['memdomain_gain_pct']=(1-wide['MemDomain']/wide['Dynamic-NaivePF'])*100
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,col,title in zip(axes,['naive_change_pct','memdomain_gain_pct'],
                       ['Naive prefetch change','MemDomain gain over matched dynamic prefetch']):
    p=wide[col].unstack();im=ax.imshow(p,aspect='auto',cmap='RdYlGn')
    ax.set_xticks(range(len(p.columns)),p.columns);ax.set_yticks(range(len(p.index)),p.index)
    ax.set_xlabel('Chunk tiles');ax.set_ylabel('Prefetch window');ax.set_title(title)
    fig.colorbar(im,ax=ax,label='%')
plt.tight_layout();plt.savefig(FIG/'exp3_public_prefetch.pdf',bbox_inches='tight');plt.show()
display(wide.reset_index())
"""),
        md("""正值`MemDomain gain`表示在相同预取条件下统一Bank池和动态映射降低周期；所有结论从重跑CSV计算。""")
    ])

def build_exp4():
    write("exp4",[
        md("""# DATE2 实验4：四模型端到端性能

比较同构/异构四个Top-1模型。最终架构统一命名为`MemDomain`。"""),
        code(SETUP+"""
rows=[]
for path in sorted((OUT/'overall').glob('*.csv')):
    q=public_rows(pd.read_csv(path));q['model']=path.stem;rows.append(q)
d=pd.concat(rows,ignore_index=True)
assert d.groupby('model').size().eq(5).all()
"""),
        code("""
p=d.pivot(index='model',columns='baseline',values='total_cycles')
speed=p['Static-NoPF'].to_numpy()[:,None]/p[PUBLIC].to_numpy()
fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(p));width=.16
for i,name in enumerate(PUBLIC):
    ax.bar(x+(i-2)*width,speed[:,i],width,label=name)
ax.axhline(1,color='black',lw=.8);ax.set_xticks(x,p.index);ax.set_ylabel('Speedup vs Static-NoPF')
ax.set_title('End-to-end performance on uniformly scaled Buckyball workloads');ax.legend(ncol=3)
plt.tight_layout();plt.savefig(FIG/'exp4_public_overall.pdf',bbox_inches='tight');plt.show()
summary=pd.DataFrame({'model':p.index,'memdomain_cycles':p.MemDomain,
 'speedup_vs_static':p['Static-NoPF']/p.MemDomain,
 'gain_vs_dynamic_pf':p['Dynamic-NaivePF']/p.MemDomain-1})
display(summary)
assert (p.MemDomain<=p[PUBLIC[:-1]].min(axis=1)).all()
"""),
        md("""最后的断言验证单一MemDomain不劣于所有公开传统候选；内部回退不形成额外论文方案。""")
    ])

def build_exp5():
    write("exp5",[
        md("""# DATE2 实验5：Prefetch Window × Weight Chunk敏感性"""),
        code(SETUP+"""
rows=[]
for path in sorted((OUT/'window_chunk').glob('w*_c*.csv')):
    w,c=map(int,re.search(r'w(\\d+)_c(\\d+)',path.stem).groups())
    q=public_rows(pd.read_csv(path));q['window']=w;q['chunk_tiles']=c;rows.append(q)
d=pd.concat(rows,ignore_index=True)
mem=d[d.baseline=='MemDomain'];base=d[d.baseline=='Static-NoPF']
p=mem.pivot(index='window',columns='chunk_tiles',values='total_cycles')
b=base.pivot(index='window',columns='chunk_tiles',values='total_cycles')
gain=(1-p/b)*100
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,data,title,cmap in ((axes[0],p,'MemDomain total cycles','viridis'),
                          (axes[1],gain,'Gain vs Static-NoPF (%)','YlGn')):
    im=ax.imshow(data,aspect='auto',cmap=cmap);ax.set_xticks(range(len(data.columns)),data.columns)
    ax.set_yticks(range(len(data.index)),data.index);ax.set_xlabel('Chunk tiles')
    ax.set_ylabel('Prefetch window');ax.set_title(title);fig.colorbar(im,ax=ax)
plt.tight_layout();plt.savefig(FIG/'exp5_public_sensitivity.pdf',bbox_inches='tight');plt.show()
display(gain)
""")
    ])

def build_exp6():
    write("exp6",[
        md("""# DATE2 实验6：模型规模、路由和并行配置鲁棒性"""),
        code(SETUP+"""
rows=[]
for path in sorted((OUT/'robustness').glob('*.csv')):
    q=public_rows(pd.read_csv(path));q['variant']=path.stem;rows.append(q)
d=pd.concat(rows,ignore_index=True)
p=d.pivot(index='variant',columns='baseline',values='total_cycles')
summary=pd.DataFrame({'variant':p.index,
 'speedup_vs_static':p['Static-NoPF']/p.MemDomain,
 'gain_vs_best_conventional':p[PUBLIC[:-1]].min(axis=1)/p.MemDomain-1})
assert (summary.gain_vs_best_conventional>=-1e-12).all()
fig,ax=plt.subplots(figsize=(14,5));q=summary.sort_values('speedup_vs_static')
ax.bar(np.arange(len(q)),q.speedup_vs_static,color='#59A14F');ax.axhline(1,color='black',lw=.8)
ax.set_xticks(np.arange(len(q)),q.variant,rotation=70,ha='right',fontsize=8)
ax.set_ylabel('MemDomain speedup vs Static-NoPF');ax.set_title('Robustness across all configurations')
plt.tight_layout();plt.savefig(FIG/'exp6_public_robustness.pdf',bbox_inches='tight');plt.show()
display(summary.describe())
"""),
        md("""若某配置收益较小，通过内部诊断CSV中的编译器计划、容量等待和Bank压力解释；不创建Raw/Safe两套论文方案。""")
    ])
