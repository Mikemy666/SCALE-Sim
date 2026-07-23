"""Run DATE2 exp1-exp3 motivation and root-cause characterization."""
from __future__ import annotations
import csv, json, math, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.memdomain_experiment import workload_digest
from scalesim.memory.topology_workload import load_moe_topology
from scalesim.memory.unified_bank_domain import UnifiedBankDomain, UnifiedMemoryRequest

CFG=ROOT/"configs/MoE/DATE2"; TOP=ROOT/"topologies/MoE/DATE2"; OUT=ROOT/"outputs/DATE2"
RESOURCE=ResourceBudget(24,24*64*1024,384,1,32)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=tuple(rows[0])); w.writeheader(); w.writerows(rows)

def partitions(total=24):
    for ia in range(1,total-1):
        for weight in range(1,total-ia): yield ia,weight,total-ia-weight

def ideal_ratio(values,total=24):
    s=sum(values); raw=[v*total/s for v in values]; result=[max(1,int(v)) for v in raw]
    while sum(result)<total: result[max(range(3),key=lambda i:raw[i]-result[i])]+=1
    while sum(result)>total: result[max(range(3),key=lambda i:result[i]-raw[i] if result[i]>1 else -1)]-=1
    return tuple(result)

def simulate(name,traffic,banks):
    ia,w,oa=banks; groups=(tuple(range(ia)),tuple(range(ia,ia+w)),tuple(range(ia+w,24)))
    req=[]
    for idx,(tensor,size,group,kind) in enumerate(zip(("ia","weight","oa"),traffic,groups,("read","read","write"))):
        req.append(UnifiedMemoryRequest(f"{name}:{tensor}",0,tensor,name,0,max(1,size),kind,group))
    return UnifiedBankDomain(RESOURCE,1024).simulate(req)

def exp1_exp2():
    source=TOP/"models/MoDSE.csv"; topology=load_moe_topology(source); exp1=[]; exp2=[]
    for name,m,n,k in topology["layers"]:
        traffic=(m*k*2,(n*k*2+7)//8,m*n*2); compute=max(1,math.ceil(m*n*k/(64*64)))
        report=simulate(name,traffic,(8,8,8)); busy=list(report.per_bank_busy_cycles.values()); mean=sum(busy)/24
        layer_type="Expert-FF1" if "FF1" in name else "Expert-FF2" if "FF2" in name else "Non-MoE"
        ideal=ideal_ratio(traffic)
        exp1.append({"layer":name,"layer_type":layer_type,"M":m,"N":n,"K":k,
            "compute_cycles":compute,"memory_stall_cycles":report.finish_cycle,"total_cycles":compute+report.finish_cycle,
            "stall_ratio":report.finish_cycle/(compute+report.finish_cycle),"ia_bytes":traffic[0],"weight_bytes":traffic[1],"oa_bytes":traffic[2],
            "bank_conflict_count":sum(report.per_bank_conflicts.values()),"bank_conflict_rate":sum(report.per_bank_conflicts.values())/report.total_beats,
            "bank_imbalance":float(npstd(busy))/mean if mean else 0,"ideal_ia_banks":ideal[0],"ideal_weight_banks":ideal[1],"ideal_oa_banks":ideal[2]})
        for ia,w,oa in partitions():
            r=simulate(name,traffic,(ia,w,oa)); exp2.append({"layer":name,"layer_type":layer_type,"ia_banks":ia,"weight_banks":w,"oa_banks":oa,
                "total_cycles":compute+r.finish_cycle,"memory_stall_cycles":r.finish_cycle,"bank_conflict_count":sum(r.per_bank_conflicts.values()),
                "bank_conflict_rate":sum(r.per_bank_conflicts.values())/r.total_beats,"hotspot_bank_ratio":sum(v>sum(r.per_bank_busy_cycles.values())/24*1.5 for v in r.per_bank_busy_cycles.values())/24,
                "idle_bank_ratio":sum(v==0 for v in r.per_bank_accesses.values())/24,"max_queue_depth":max(r.per_bank_max_queue_depth.values())})
    write_csv(OUT/"exp1/layer_characterization.csv",exp1); write_csv(OUT/"exp2/static_bank_sweep.csv",exp2)
    best=[]
    for layer in sorted({r["layer"] for r in exp2}): best.append(min((r for r in exp2 if r["layer"]==layer),key=lambda x:(x["total_cycles"],x["ia_banks"],x["weight_banks"])))
    write_csv(OUT/"exp2/per_stage_best.csv",best)

def npstd(values):
    mean=sum(values)/len(values); return math.sqrt(sum((v-mean)**2 for v in values)/len(values))

def exp3():
    rows=[]
    for path in sorted((OUT/"window_chunk").glob("w*_c*.csv")):
        import re
        w,c=map(int,re.search(r"w(\d+)_c(\d+)",path.stem).groups())
        with path.open(newline="",encoding="utf-8") as f: data=list(csv.DictReader(f))
        config_path=CFG/"window_chunk"/f"{path.stem}.json"
        expected_hash=workload_digest(json.loads(config_path.read_text(encoding="utf-8")))
        actual_hashes={row["workload_hash"] for row in data}
        if actual_hashes!={expected_hash}:
            raise RuntimeError(
                f"stale exp3 source matrix {path}: regenerate {config_path.name} results"
            )
        for baseline in ("Static-NoPF","Static-NaivePF"):
            r=next(x for x in data if x["baseline"]==baseline)
            rows.append({"window":w,"chunk_tiles":c,"baseline":baseline,**{k:r[k] for k in
                ("total_cycles","prefetch_requests","prefetch_bytes","bank_conflict_count","bank_conflict_rate","prefetch_interference_stall_cycles","timely_prefetch_ratio","late_prefetch_ratio","unused_prefetch_ratio","prefetch_occupancy_byte_cycles","compute_transfer_overlap_cycles")}})
    write_csv(OUT/"exp3/naive_prefetch_interference.csv",rows)

def prepare():
    for exp in ("exp1","exp2","exp3"):
        (CFG/exp).mkdir(parents=True,exist_ok=True); (TOP/exp).mkdir(parents=True,exist_ok=True); (OUT/exp).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(TOP/"models/MoDSE.csv",TOP/exp/"MoDSE.csv")
    configs={"exp1":{"purpose":"B1_C1_layer_and_flow_characterization","topology":"topologies/MoE/DATE2/exp1/MoDSE.csv","static_banks":[8,8,8]},
             "exp2":{"purpose":"C2_exhaustive_static_bank_ownership","topology":"topologies/MoE/DATE2/exp2/MoDSE.csv","bank_total":24,"positive_partitions":253},
             "exp3":{"purpose":"C3_naive_prefetch_interference","source_suite":"configs/MoE/DATE2/window_chunk","windows":[0,1,2,4,8],"chunk_tiles":[1,2,4,8]}}
    for exp,payload in configs.items(): (CFG/exp/f"{exp}.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--exp",choices=("all","exp1","exp2","exp3"),default="all")
    args=parser.parse_args();prepare()
    # exp1 and exp2 share the same per-layer traffic construction and are
    # intentionally generated together when either is requested.
    if args.exp in ("all","exp1","exp2"): exp1_exp2()
    if args.exp in ("all","exp3"): exp3()
    print(f"DATE2 {args.exp} characterization completed")
if __name__=="__main__": main()
