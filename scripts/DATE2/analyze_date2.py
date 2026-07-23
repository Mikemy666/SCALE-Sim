"""Generate DATE2 paper figures, reduced tables, and an evidence-based report."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs/DATE2"; FIG=ROOT/"fig/DATE2"
FIG.mkdir(parents=True,exist_ok=True)
BASELINES=["Static-NoPF","Static-NaivePF","Dynamic-NoPF","Dynamic-NaivePF","MemDomain-Raw","MemDomain-Safe","Oracle"]
COLORS=["#9e9e9e","#c7c7c7","#4e79a7","#76b7b2","#e15759","#59a14f","#b07aa1"]

def raw(path): return pd.read_csv(path)
def save(name):
    plt.tight_layout(); plt.savefig(FIG/name,bbox_inches="tight"); plt.close()

def characterization():
    d=raw(OUT/"exp1/layer_characterization.csv"); d.to_csv(FIG/"exp1_layer_data.csv",index=False)
    grouped=d.groupby("layer_type")[["compute_cycles","memory_stall_cycles"]].sum()
    grouped.plot(kind="bar",stacked=True,figsize=(8,4),color=["#4e79a7","#e15759"]);plt.ylabel("Cycles");plt.xlabel("")
    save("exp1_layer_bottleneck.pdf")
    fig,ax=plt.subplots(figsize=(6,4));
    for kind,q in d.groupby("layer_type"): ax.scatter(q.bank_imbalance,q.stall_ratio,label=kind)
    ax.set_xlabel("Bank imbalance");ax.set_ylabel("Stall ratio");ax.legend();save("exp1_stall_imbalance.pdf")

    sweep=raw(OUT/"exp2/static_bank_sweep.csv");best=raw(OUT/"exp2/per_stage_best.csv")
    best.to_csv(FIG/"exp2_per_stage_best.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(best));ax.plot(x,best.ia_banks,label="IA");ax.plot(x,best.weight_banks,label="Weight");ax.plot(x,best.oa_banks,label="OA");ax.set_xlabel("Layer/stage");ax.set_ylabel("Best static Banks");ax.legend();save("exp2_best_static_ratio.pdf")

    p=raw(OUT/"exp3/naive_prefetch_interference.csv");p.to_csv(FIG/"exp3_prefetch_data.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    for chunk,q in p[p.baseline=="Static-NaivePF"].groupby("chunk_tiles"): ax.plot(q.window,q.total_cycles.astype(float),marker="o",label=f"C={chunk}")
    ax.set_xlabel("Window");ax.set_ylabel("Static-NaivePF cycles");ax.legend();save("exp3_naive_interference.pdf")

def overall():
    frames=[]
    for path in sorted((OUT/"overall").glob("*.csv")):
        d=raw(path); base=float(d.loc[d.baseline=="Static-NoPF","total_cycles"].iloc[0])
        d["model"]=path.stem; d["normalized_cycles"]=d.total_cycles/base; d["speedup"]=base/d.total_cycles
        frames.append(d)
    d=pd.concat(frames,ignore_index=True); d.to_csv(FIG/"overall_data.csv",index=False)
    p=d.pivot(index="model",columns="baseline",values="normalized_cycles").reindex(columns=BASELINES)
    ax=p.plot(kind="bar",figsize=(11,4.5),color=COLORS,width=.85)
    ax.set_ylabel("Normalized total cycles"); ax.set_xlabel(""); ax.axhline(1,color="black",lw=.8); ax.legend(ncol=4,fontsize=8)
    save("exp4_overall_performance.pdf")
    safe=d[d.baseline=="MemDomain-Safe"].set_index("model")
    report={m:{"safe_speedup":float(r.speedup),"safe_selected":r.selected_candidate,
               "conflict_rate":float(r.bank_conflict_rate),"memory_stall":int(r.total_cycles-r.compute_cycles)}
            for m,r in safe.iterrows()}
    return report

def ablation():
    d=raw(OUT/"ablation/MoDSE_components.csv"); base=float(d.loc[d.baseline=="Static-NoPF","total_cycles"].iloc[0])
    d["speedup"]=base/d.total_cycles; d.to_csv(FIG/"ablation_data.csv",index=False)
    order=BASELINES[:-1]; q=d.set_index("baseline").loc[order]
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].bar(range(len(q)),q.speedup,color=COLORS[:-1]); axes[0].set_xticks(range(len(q)),order,rotation=35,ha="right"); axes[0].set_ylabel("Speedup vs Static-NoPF")
    comps=["bank_stall_cycles","weight_load_stall_cycles","prefetch_miss_stall_cycles","prefetch_interference_stall_cycles","mapping_overhead_cycles"]
    bottom=np.zeros(len(q))
    for comp in comps: axes[1].bar(range(len(q)),q[comp],bottom=bottom,label=comp.replace("_cycles","")); bottom+=q[comp].to_numpy()
    axes[1].set_xticks(range(len(q)),order,rotation=35,ha="right"); axes[1].set_ylabel("Stall cycles"); axes[1].legend(fontsize=7)
    save("exp5_ablation.pdf")
    raw_row=d[d.baseline=="MemDomain-Raw"].iloc[0]; safe=d[d.baseline=="MemDomain-Safe"].iloc[0]
    return {"raw_speedup":float(raw_row.speedup),"safe_speedup":float(safe.speedup),"safe_selected":safe.selected_candidate}

def window_chunk():
    rows=[]
    for path in sorted((OUT/"window_chunk").glob("w*_c*.csv")):
        w,c=map(int,re.search(r"w(\d+)_c(\d+)",path.stem).groups()); d=raw(path)
        for baseline in ("MemDomain-Raw","MemDomain-Safe"):
            r=d[d.baseline==baseline].iloc[0].to_dict(); r.update(window=w,chunk_tiles=c); rows.append(r)
    d=pd.DataFrame(rows); d.to_csv(FIG/"window_chunk_data.csv",index=False)
    rawd=d[d.baseline=="MemDomain-Raw"]
    metrics=[("total_cycles","Total cycles"),("prefetch_miss_stall_cycles","Late-prefetch stall"),("prefetch_occupancy_byte_cycles","Prefetch occupancy")]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,(metric,title) in zip(axes,metrics):
        p=rawd.pivot(index="window",columns="chunk_tiles",values=metric).sort_index().sort_index(axis=1)
        im=ax.imshow(p.values,aspect="auto",cmap="viridis"); ax.set_xticks(range(len(p.columns)),p.columns); ax.set_yticks(range(len(p.index)),p.index); ax.set_xlabel("Chunk (tiles)"); ax.set_ylabel("Window"); ax.set_title(title); fig.colorbar(im,ax=ax,shrink=.75)
    save("exp6_window_chunk_heatmap.pdf")
    best=rawd.loc[rawd.total_cycles.idxmin()]
    return {"best_window":int(best.window),"best_chunk_tiles":int(best.chunk_tiles),"best_raw_cycles":int(best.total_cycles),
            "timely_ratio_range":[float(rawd.timely_prefetch_ratio.min()),float(rawd.timely_prefetch_ratio.max())]}

def robustness():
    rows=[]
    for path in sorted((OUT/"robustness").glob("*.csv")):
        d=raw(path); base=float(d.loc[d.baseline=="Static-NoPF","total_cycles"].iloc[0]); r=d[d.baseline=="MemDomain-Safe"].iloc[0].to_dict(); r.update(variant=path.stem,speedup=base/r["total_cycles"]); rows.append(r)
    d=pd.DataFrame(rows); d.to_csv(FIG/"robustness_data.csv",index=False)
    groups=[("topk_","Top-K"),("experts_","Experts"),("tokens_","Tokens"),("routing_","Routing"),("class_","Expert sizes"),("ep_","EP")]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,(prefix,title) in zip(axes.ravel(),groups):
        q=d[d.variant.str.startswith(prefix)].sort_values("variant"); ax.bar(q.variant.str.replace(prefix,"",regex=False),q.speedup,color="#59a14f"); ax.set_title(title); ax.set_ylabel("Safe speedup"); ax.tick_params(axis="x",rotation=25)
    save("exp7_robustness.pdf")
    return {row.variant:{"safe_speedup":float(row.speedup),"communication_stall":int(row.communication_stall_cycles)} for row in d.itertuples()}

def notebook(name,title,files):
    cells=[{"cell_type":"markdown","metadata":{},"source":[f"# {title}\n","DATE2 source files:\n"]+[f"- `{f}`\n" for f in files]},
           {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["from pathlib import Path\n","ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'fig' else Path.cwd().resolve()\n",f"%run {{ROOT / 'scripts/DATE2/analyze_date2.py'}}\n"]}]
    payload={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
    (ROOT/"fig"/name).write_text(json.dumps(payload,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")

def main():
    characterization()
    results={"overall":overall(),"ablation":ablation(),"window_chunk":window_chunk(),"robustness":robustness()}
    (FIG/"analysis.json").write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    notebook("exp1.ipynb","DATE2 B1/C1 Layer and flow characterization",["outputs/DATE2/exp1/layer_characterization.csv"])
    notebook("exp2.ipynb","DATE2 C2 Static Bank ownership mismatch",["outputs/DATE2/exp2/static_bank_sweep.csv","outputs/DATE2/exp2/per_stage_best.csv"])
    notebook("exp3.ipynb","DATE2 C3 Naive prefetch interference",["outputs/DATE2/exp3/naive_prefetch_interference.csv"])
    notebook("exp4.ipynb","DATE2 IV-B Overall",["outputs/DATE2/overall/*.csv","outputs/DATE2/exp4/*/EXPERT_REPORT.csv"])
    notebook("exp5.ipynb","DATE2 IV-C Ablation",["outputs/DATE2/ablation/MoDSE_components.csv"])
    notebook("exp6.ipynb","DATE2 IV-D Window x Chunk",["outputs/DATE2/window_chunk/w*_c*.csv"])
    notebook("exp7.ipynb","DATE2 IV-E Robustness",["outputs/DATE2/robustness/*.csv","outputs/DATE2/exp7/*/EXPERT_REPORT.csv"])
    print(json.dumps(results,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
