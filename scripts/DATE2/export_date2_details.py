"""Export real P9 request/Bank/Chunk/expert detail for one DATE2 matrix."""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scalesim.memory.memdomain_experiment import Baseline
from scalesim.memory.memdomain_runner import (
    load_runner_config,
    run_best_static_baseline_with_details,
    run_dominating_dynamic_baseline_with_details,
    run_raw_baseline_with_details,
)

RAW=(Baseline.STATIC_NOPF,Baseline.STATIC_NAIVEPF,Baseline.DYNAMIC_NOPF,Baseline.DYNAMIC_NAIVEPF,Baseline.MEMDOMAIN_RAW)

def write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=tuple(rows[0]));w.writeheader();w.writerows(rows)

def export(config_path,output_dir):
    config=load_runner_config(Path(config_path)); output_dir=Path(output_dir); by_chunk={c.chunk_id:c for c in config.chunks}
    chunk_rows=[];bank_rows=[];request_rows=[];layer_acc=defaultdict(lambda:defaultdict(int));expert_acc=defaultdict(lambda:defaultdict(int)); selections=[]
    static_results={
        Baseline.STATIC_NOPF:run_best_static_baseline_with_details(config,Baseline.STATIC_NOPF),
        Baseline.STATIC_NAIVEPF:run_best_static_baseline_with_details(config,Baseline.STATIC_NAIVEPF),
    }
    for baseline in RAW:
        if baseline in static_results:
            result=static_results[baseline]
        elif baseline==Baseline.DYNAMIC_NOPF:
            result=run_dominating_dynamic_baseline_with_details(config,baseline,static_results[Baseline.STATIC_NOPF])
        elif baseline==Baseline.DYNAMIC_NAIVEPF:
            result=run_dominating_dynamic_baseline_with_details(config,baseline,static_results[Baseline.STATIC_NAIVEPF])
        else:
            result=run_raw_baseline_with_details(config,baseline)
        selections.append({"baseline":baseline.value,"total_cycles":result.row.total_cycles,"candidate_source":result.row.candidate_source})
        services={s.request_id:s for s in result.memory_report.services}
        for item in result.chunks:
            chunk=by_chunk[item.chunk_id]; service=services[f"load:{item.chunk_id}"]
            row={"baseline":baseline.value,"chunk_id":item.chunk_id,"expert_id":chunk.expert_id,"ffn_part":chunk.ffn_part,"tile_id":chunk.tile_id,"size_bytes":chunk.size_bytes,
                 "planned_kind":item.planned_kind,"effective_kind":item.effective_kind,"planned_issue_cycle":item.planned_issue_cycle,"actual_issue_cycle":item.actual_issue_cycle,"completion_cycle":item.completion_cycle,
                 "use_cycle":item.use_cycle,"consume_cycle":item.consume_cycle,"allocation_wait_cycles":item.allocation_wait_cycles,"miss_stall_cycles":item.miss_stall_cycles,"classification":item.classification,
                 "physical_banks":"|".join(map(str,item.physical_banks)),"queue_wait_cycles":service.queue_wait_cycles}
            chunk_rows.append(row)
            for key in ((baseline.value,chunk.expert_id),(baseline.value,chunk.expert_id,chunk.ffn_part)):
                acc=expert_acc[key] if len(key)==2 else layer_acc[key]; acc["chunks"]+=1;acc["bytes"]+=chunk.size_bytes;acc["miss_stall_cycles"]+=item.miss_stall_cycles;acc["allocation_wait_cycles"]+=item.allocation_wait_cycles;acc["timely"]+=item.classification=="timely";acc["late"]+=item.classification=="late";acc["demand_miss"]+=item.classification=="demand_miss"
        report=result.memory_report
        for bank in range(config.resources.bank_count):
            bank_rows.append({"baseline":baseline.value,"bank":bank,"accesses":report.per_bank_accesses[bank],"busy_cycles":report.per_bank_busy_cycles[bank],"conflicts":report.per_bank_conflicts[bank],"queue_wait_cycles":report.per_bank_queue_wait[bank],"max_queue_depth":report.per_bank_max_queue_depth[bank]})
        for s in report.services: request_rows.append({"baseline":baseline.value,"request_id":s.request_id,"issue_cycle":s.issue_cycle,"start_cycle":s.start_cycle,"completion_cycle":s.completion_cycle,"queue_wait_cycles":s.queue_wait_cycles,"banks":"|".join(map(str,s.banks)),"beat_count":s.beat_count})
    expert_rows=[{"baseline":k[0],"expert_id":k[1],**v} for k,v in sorted(expert_acc.items())]
    layer_rows=[{"baseline":k[0],"expert_id":k[1],"ffn_part":k[2],**v} for k,v in sorted(layer_acc.items())]
    prov=json.loads(Path(config_path).read_text()).get("topology_provenance",{}); counts=prov.get("token_counts",[])
    input_rows=[{"expert_id":i,"tokens":v,"top_k":prov.get("top_k",1),"routing_mode":prov.get("routing_mode",""),"routing_severity":prov.get("routing_severity","")} for i,v in enumerate(counts)]
    write(output_dir/"CHUNK_REPORT.csv",chunk_rows);write(output_dir/"EXPERT_REPORT.csv",expert_rows);write(output_dir/"FFN_STAGE_REPORT.csv",layer_rows);write(output_dir/"BANK_REPORT.csv",bank_rows);write(output_dir/"REQUEST_REPORT.csv",request_rows);write(output_dir/"EXPERT_INPUT_REPORT.csv",input_rows);write(output_dir/"MEASURED_SELECTIONS.csv",selections)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("config",type=Path);p.add_argument("output",type=Path);a=p.parse_args();export(a.config,a.output)
