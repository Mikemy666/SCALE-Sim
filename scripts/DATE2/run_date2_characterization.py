"""Run DATE2 exp1-exp3 motivation and root-cause characterization."""
from __future__ import annotations
import csv, json, math, shutil, sys
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.memdomain_experiment import workload_digest
from scalesim.memory.topology_workload import load_moe_topology
from scalesim.memory.unified_bank_domain import UnifiedBankDomain, UnifiedMemoryRequest
from scalesim.memory.buckyball_memdomain import CONTRACT

CFG=ROOT/"configs/MoE/DATE2"; TOP=ROOT/"topologies/MoE/DATE2"; OUT=ROOT/"outputs/DATE2"
RESOURCE=ResourceBudget(30,30*128*16,480,1,32)
IA_BYTES_PER_ELEMENT=1
WEIGHT_BYTES_PER_ELEMENT=1
ACCUMULATOR_BYTES_PER_ELEMENT=4
OUTPUT_BYTES_PER_ELEMENT=1
ARRAY_DIM=16

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=tuple(rows[0])); w.writeheader(); w.writerows(rows)

def partitions(total=15):
    for ia in range(1,total-1):
        for weight in range(1,total-ia): yield ia,weight,total-ia-weight

def ideal_ratio(values,total=15):
    s=sum(values); raw=[v*total/s for v in values]; result=[max(1,int(v)) for v in raw]
    while sum(result)<total: result[max(range(3),key=lambda i:raw[i]-result[i])]+=1
    while sum(result)>total: result[max(range(3),key=lambda i:result[i]-raw[i] if result[i]>1 else -1)]-=1
    return tuple(result)

def simulate(name,traffic,banks,compute_cycles,m,n,k,collect_temporal=False):
    """Causal 16x16-tile model for the fixed SP/ACC baseline.

    Each output tile executes K folds in order. IA and Weight tile reads may
    proceed in parallel, but compute waits for both; every fold then commits
    its INT32 partial sum to a four-Bank ACC stripe. Later folds use atomic
    three-cycle RMW. Requantization and the INT8 OA write occur only after the
    final ACC operation. Objects are tile-sized and released at the end of the
    tile, so every live object is checked against its owned Bank capacity.
    """
    ia,w,oa=banks
    groups=(tuple(range(ia)),tuple(range(ia,ia+w)),tuple(range(ia+w,15)))
    acc_groups=(tuple(range(15,19)),tuple(range(19,23)),tuple(range(23,27)))
    available=[0]*30; busy=[0]*30; accesses=[0]*30; conflicts=[0]*30
    queue_wait=[0]*30; max_queue=[0]*30
    total_bytes=0; total_beats=0

    def schedule(issue,address,size,group,multiplier=1):
        nonlocal total_bytes,total_beats
        if not group: raise ValueError("a tile requires at least one owned Bank")
        if size>len(group)*CONTRACT.bank_bytes:
            raise MemoryError(
                f"{name}: tile object {size} B exceeds {len(group)}-Bank capacity"
            )
        lines=math.ceil(size/16); first=(address//16)%len(group)
        counts={bank:0 for bank in group}
        for line in range(lines): counts[group[(first+line)%len(group)]]+=1
        completion=issue
        for bank,count in counts.items():
            if not count: continue
            start=max(issue,available[bank]); duration=count*multiplier
            if start>issue:
                conflicts[bank]+=count
                queue_wait[bank]+=start-issue
                max_queue[bank]=max(max_queue[bank],2)
            # Multiple lines mapped to one single-ported Bank serialize.
            conflicts[bank]+=max(0,count-1)
            max_queue[bank]=max(max_queue[bank],1)
            available[bank]=start+duration; completion=max(completion,available[bank])
            busy[bank]+=duration; accesses[bank]+=count
        total_bytes+=size; total_beats+=lines
        return completion

    mt,nt,kt=(math.ceil(value/16) for value in (m,n,k))
    cursor=0; mac_cycles=0; requant_cycles=0
    ia_stall=weight_stall=shared_stall=acc_stall=oa_stall=0
    temporal=[]
    # Weight-stationary tile residency. A tile remains live in the fixed
    # Weight pool until LRU eviction; this avoids the old, incorrect behavior
    # that reloaded the same Weight tile for every M tile.
    weight_capacity=len(groups[1])*CONTRACT.bank_bytes
    weight_cache={}; weight_resident_bytes=0
    for ti in range(mt):
        tm=min(16,m-ti*16)
        for tj in range(nt):
            tn=min(16,n-tj*16); tile=ti*nt+tj
            tile_start=cursor; tile_ia=tile_weight=0
            acc_group=acc_groups[tile%len(acc_groups)]
            acc_bytes=tm*tn*ACCUMULATOR_BYTES_PER_ELEMENT
            for fold in range(kt):
                tk=min(16,k-fold*16)
                ia_bytes=tm*tk*IA_BYTES_PER_ELEMENT
                weight_bytes=tk*tn*WEIGHT_BYTES_PER_ELEMENT
                ia_address=(ti*16*k+fold*16)*IA_BYTES_PER_ELEMENT
                weight_address=(fold*16*n+tj*16)*WEIGHT_BYTES_PER_ELEMENT
                ia_done=schedule(cursor,ia_address,ia_bytes,groups[0])
                weight_key=(fold,tj)
                weight_loaded=0
                if weight_key in weight_cache:
                    # Refresh deterministic LRU order.
                    cached=weight_cache.pop(weight_key)
                    weight_cache[weight_key]=cached
                    weight_done=cursor
                else:
                    while weight_cache and weight_resident_bytes+weight_bytes>weight_capacity:
                        oldest=next(iter(weight_cache))
                        weight_resident_bytes-=weight_cache.pop(oldest)
                    if weight_resident_bytes+weight_bytes>weight_capacity:
                        raise MemoryError(f"{name}: Weight tile cannot fit fixed Weight pool")
                    weight_done=schedule(cursor,weight_address,weight_bytes,groups[1])
                    weight_cache[weight_key]=weight_bytes
                    weight_resident_bytes+=weight_bytes
                    weight_loaded=weight_bytes
                ia_latency=ia_done-cursor; weight_latency=weight_done-cursor
                shared=min(ia_latency,weight_latency)
                shared_stall+=shared
                ia_stall+=max(0,ia_latency-weight_latency)
                weight_stall+=max(0,weight_latency-ia_latency)
                compute_start=max(ia_done,weight_done)
                fold_compute=max(1,math.ceil(tm*tn*tk/(ARRAY_DIM*ARRAY_DIM)))
                compute_end=compute_start+fold_compute; mac_cycles+=fold_compute
                acc_done=schedule(
                    compute_end,tile*CONTRACT.accumulator_tile_bytes,
                    acc_bytes,acc_group,CONTRACT.rmw_cycles if fold else 1
                )
                acc_stall+=acc_done-compute_end; cursor=acc_done
                tile_ia+=ia_bytes; tile_weight+=weight_loaded
            requant_cycles+=CONTRACT.requant_tile_cycles
            cursor+=CONTRACT.requant_tile_cycles
            oa_bytes=tm*tn*OUTPUT_BYTES_PER_ELEMENT
            oa_done=schedule(cursor,(ti*16*n+tj*16),oa_bytes,groups[2])
            oa_stall+=oa_done-cursor; cursor=oa_done
            if collect_temporal:
                ideal=ideal_ratio((tile_ia,tile_weight,oa_bytes))
                temporal.append({
                    "layer":name,"tile_index":tile,"tile_m":tm,"tile_n":tn,
                    "k_folds":kt,"start_cycle":tile_start,"end_cycle":cursor,
                    "ia_working_set_bytes":tile_ia,
                    "weight_load_bytes":tile_weight,
                    "weight_resident_bytes":weight_resident_bytes,
                    "weight_working_set_bytes":min(weight_capacity,k*tn),
                    "oa_working_set_bytes":oa_bytes,
                    "accumulator_working_set_bytes":acc_bytes,
                    "active_ia_banks":ideal[0],"active_weight_banks":ideal[1],
                    "active_oa_banks":ideal[2],
                    "fixed_ia_banks":ia,"fixed_weight_banks":w,"fixed_oa_banks":oa,
                })
    # The exact edge-tile MAC sum is the authoritative compute count.
    compute_cycles=mac_cycles+requant_cycles
    breakdown={
        "ia_stall_cycles":ia_stall,"weight_stall_cycles":weight_stall,
        "shared_operand_stall_cycles":shared_stall,
        "accumulator_stall_cycles":acc_stall,"oa_stall_cycles":oa_stall,
        "requant_cycles":requant_cycles,
    }
    report=SimpleNamespace(
        finish_cycle=cursor,
        per_bank_busy_cycles={i:v for i,v in enumerate(busy)},
        per_bank_accesses={i:v for i,v in enumerate(accesses)},
        per_bank_conflicts={i:v for i,v in enumerate(conflicts)},
        per_bank_queue_wait={i:v for i,v in enumerate(queue_wait)},
        per_bank_max_queue_depth={i:v for i,v in enumerate(max_queue)},
        total_bytes=total_bytes,total_beats=total_beats,
        compute_cycles=compute_cycles,breakdown=breakdown,
        temporal_rows=temporal,
    )
    operand_stall=ia_stall+weight_stall+shared_stall
    output_path_stall=acc_stall+oa_stall
    return report,operand_stall,output_path_stall

def exp1_exp2():
    source=TOP/"models/MoDSE.csv"; topology=load_moe_topology(source)
    exp1=[]; exp2=[]; acc_activity=[]; temporal=[]
    for name,m,n,k in topology["layers"]:
        traffic=(m*k*IA_BYTES_PER_ELEMENT,n*k*WEIGHT_BYTES_PER_ELEMENT,
                 m*n*OUTPUT_BYTES_PER_ELEMENT)
        compute=max(1,math.ceil(m*n*k/(ARRAY_DIM*ARRAY_DIM)))
        report,input_stall,output_stall=simulate(
            name,traffic,(5,5,5),compute,m,n,k,collect_temporal=True
        )
        compute=report.compute_cycles
        exposed_stall=sum(report.breakdown[key] for key in (
            "ia_stall_cycles","weight_stall_cycles","shared_operand_stall_cycles",
            "accumulator_stall_cycles","oa_stall_cycles"
        ))
        busy=list(report.per_bank_busy_cycles.values()); mean=sum(busy)/30
        critical_bank_service=max(busy)
        aggregate_bank_service=sum(busy)
        pool_pressure=[
            sum(busy[0:5])/5,
            sum(busy[5:10])/5,
            sum(busy[10:15])/5,
        ]
        pressure_mean=sum(pool_pressure)/3
        bank_imbalance=npstd(pool_pressure)/pressure_mean if pressure_mean else 0
        layer_type="Expert-FF1" if "FF1" in name else "Expert-FF2" if "FF2" in name else "Non-MoE"
        ideal=ideal_ratio(traffic)
        conflict_rate=sum(report.per_bank_conflicts.values())/report.total_beats
        exp1.append({"layer":name,"layer_type":layer_type,"M":m,"N":n,"K":k,
            "compute_cycles":compute,"mac_cycles":compute-report.breakdown["requant_cycles"],
            "requant_cycles":report.breakdown["requant_cycles"],
            "memory_service_cycles":critical_bank_service,
            "operand_load_stall_cycles":input_stall,"output_store_stall_cycles":output_stall,
            **report.breakdown,
            "memory_stall_cycles":exposed_stall,"total_cycles":report.finish_cycle,
            "stall_ratio":exposed_stall/report.finish_cycle,
            "critical_bank_service_cycles":critical_bank_service,
            "aggregate_bank_service_cycles":aggregate_bank_service,
            "memory_to_compute_ratio":exposed_stall/compute,
            "ia_bytes":traffic[0],"weight_bytes":traffic[1],
            "accumulator_bytes":math.ceil(m/16)*math.ceil(n/16)*1024,
            "oa_bytes":traffic[2],"accumulator_mode":"banked_rmw",
            "bank_conflict_count":sum(report.per_bank_conflicts.values()),
            "bank_conflict_rate":conflict_rate,
            "pressure_weighted_bank_conflict":conflict_rate*exposed_stall/compute,
            "bank_imbalance":bank_imbalance,
            "pressure_weighted_bank_imbalance":bank_imbalance*exposed_stall/compute,
            "physical_bank_busy_cv":float(npstd(busy))/mean if mean else 0,
            "ownership_mismatch":sum(abs(value-5) for value in ideal)/30,
            "ideal_ia_banks":ideal[0],"ideal_weight_banks":ideal[1],"ideal_oa_banks":ideal[2]})
        for row in report.temporal_rows:
            row["layer_type"]=layer_type
            temporal.append(row)
        k_folds=max(1,math.ceil(k/ARRAY_DIM))
        acc_activity.append({
            "layer":name,"layer_type":layer_type,"M":m,"N":n,"K":k,
            "k_folds":k_folds,
            "accumulator_working_set_bytes":math.ceil(m/16)*math.ceil(n/16)*1024,
            "accumulator_stall_cycles":report.breakdown["accumulator_stall_cycles"],
            "accumulator_mode":"banked_rmw","accumulator_banks":15,
            "stripe_banks":4,"rmw_cycles_per_line":3,
        })
        for ia,w,oa in partitions():
            r,input_wait,output_wait=simulate(
                name,traffic,(ia,w,oa),compute,m,n,k
            )
            stall=input_wait+output_wait
            demand=ideal_ratio(traffic)
            owned=(ia,w,oa)
            shortage=sum(max(0,need-have) for need,have in zip(demand,owned))
            surplus=sum(max(0,have-need) for need,have in zip(demand,owned))
            max_pressure=max(need/have for need,have in zip(demand,owned))
            exp2.append({"layer":name,"layer_type":layer_type,"ia_banks":ia,"weight_banks":w,"oa_banks":oa,
                "total_cycles":r.finish_cycle,"memory_service_cycles":r.finish_cycle,
                "memory_stall_cycles":stall,"bank_conflict_count":sum(r.per_bank_conflicts.values()),
                "bank_conflict_rate":sum(r.per_bank_conflicts.values())/r.total_beats,
                "hotspot_bank_ratio":shortage/15,
                "idle_bank_ratio":surplus/15,
                "max_bank_pressure":max_pressure,
                "max_queue_depth":max(r.per_bank_max_queue_depth.values())})
    write_csv(OUT/"exp1/layer_characterization.csv",exp1)
    write_csv(OUT/"exp1/accumulator_sensitivity.csv",acc_activity)
    write_csv(OUT/"exp1/temporal_bank_demand.csv",temporal)
    write_csv(OUT/"exp2/static_bank_sweep.csv",exp2)
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
    precision={"original_model_format":"FP32","compute_format":"INT8xINT8_INT32",
               "ia_bytes_per_element":IA_BYTES_PER_ELEMENT,
               "weight_bytes_per_element":WEIGHT_BYTES_PER_ELEMENT,
               "accumulator_bytes_per_element":ACCUMULATOR_BYTES_PER_ELEMENT,
               "output_bytes_per_element":OUTPUT_BYTES_PER_ELEMENT,
               "accumulator_mode":"banked_rmw","spill_sensitivity":False}
    hardware={"bank_count":30,"bank_width_bits":128,"bank_entries":128,
              "capacity_bytes":61440,"ports_per_bank":1,
              "bandwidth_bytes_per_cycle":480,
              "mapping_overhead_per_object":0}
    configs={"exp1":{"purpose":"B1_C1_fixed_SP_ACC_boundary","topology":"topologies/MoE/DATE2/exp1/MoDSE.csv","static_sp_banks":[5,5,5],"static_acc_banks":15,"precision":precision,"hardware":hardware},
             "exp2":{"purpose":"C2_exhaustive_static_SP_ownership","topology":"topologies/MoE/DATE2/exp2/MoDSE.csv","sp_bank_total":15,"acc_bank_total":15,"positive_partitions":91,"precision":precision,"hardware":hardware},
             "exp3":{"purpose":"C3_naive_prefetch_interference","source_suite":"configs/MoE/DATE2/window_chunk","windows":[0,1,2,4,8,16,32,64],"chunk_tiles":[1,2,4,8]}}
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
