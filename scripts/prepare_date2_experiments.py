"""Generate DATE2 simulator configs, topologies, and output hierarchy."""
from __future__ import annotations
import csv, json, shutil, sys, random
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scalesim.memory.moe_workload_catalog import write_runner_payload
from scalesim.memory.topology_workload import generate_topology_runner_payload
from scalesim.memory.date2_network_contract import validate_network_set
CONFIG_ROOT = ROOT / "configs/MoE/DATE2"
TOPOLOGY_ROOT = ROOT / "topologies/MoE/DATE2"
OUTPUT_ROOT = ROOT / "outputs/DATE2"
MODELS = (("HMoE", "heterogeneous"), ("Mixtral", "homogeneous"),
          ("MoDSE", "heterogeneous"), ("Switchtrans", "homogeneous"))
UNIFORM_DIMENSION_DIVISOR = 4

def balanced(tokens, experts, top_k=1):
    total = tokens * top_k
    return tuple(total // experts + (i < total % experts) for i in range(experts))

def skewed(tokens, severity):
    weights = {"balanced": [1]*8, "light": [4,3,2,2,1,1,1,1],
               "high": [16,8,4,2,1,1,1,1]}[severity]
    values = [tokens*w//sum(weights) for w in weights]
    for i in range(tokens-sum(values)): values[i] += 1
    return tuple(values)

def skewed_seed(tokens,severity,seed):
    weights={"light":[4,3,2,2,1,1,1,1],"high":[16,8,4,2,1,1,1,1]}[severity]
    rng=random.Random(seed); counts=[0]*8
    population=[]
    for expert,weight in enumerate(weights): population.extend([expert]*weight)
    for _ in range(tokens): counts[rng.choice(population)]+=1
    return tuple(counts)

def variant_topology(path, source, counts):
    rows = []
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if row and row[0].startswith("MoE-E"):
                row[1] = str(counts[int(row[0].split("-")[1][1:])])
            elif row and row[0] != "Layer": row[1] = str(sum(counts))
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)

def uniformly_scale_topology(path, source):
    rows=[]
    with source.open(newline="",encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if row and row[0] and row[0]!="Layer":
                # M carries routed token counts and is intentionally unchanged.
                # Router N is the expert count; every matrix dimension is
                # divided by the same factor then aligned to the 16x16 array.
                for index in (2,3):
                    if index==2 and row[0]=="Router_logits":
                        continue
                    value=int(row[index])
                    row[index]=str(max(16,((value//UNIFORM_DIMENSION_DIVISOR+15)//16)*16))
            rows.append(row)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)

def save(suite, name, payload):
    payload = deepcopy(payload)
    if suite in {
        "overall", "joint_prefetch", "robustness", "robustness_factorial"
    }:
        payload["policy"].update({
            "adaptive_prefetch": True,
            "max_prefetch_window": 32,
            "max_prefetch_capacity_fraction": 0.25,
        })
    payload.update(experiment_id=f"date2-{suite}-{name}", date2_suite=suite,
                   date2_variant=name)
    write_runner_payload(CONFIG_ROOT / suite / f"{name}.json", payload)
    (OUTPUT_ROOT / suite).mkdir(parents=True, exist_ok=True)

def main():
    for root in (CONFIG_ROOT, TOPOLOGY_ROOT, OUTPUT_ROOT): root.mkdir(parents=True, exist_ok=True)
    bases = {}
    for model, kind in MODELS:
        target = TOPOLOGY_ROOT / "models" / f"{model}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        uniformly_scale_topology(
            target, ROOT / f"topologies/MoE/{model}.csv"
        )
        bases[model] = generate_topology_runner_payload(target, kind)
        save("overall", model, bases[model])
    compatibility = validate_network_set([
        TOPOLOGY_ROOT / "models" / f"{model}.csv" for model, _ in MODELS
    ])

    for window in (0,1,2,4,8,16,32,64):
        for tiles in (1,2,4,8):
            payload = generate_topology_runner_payload(
                TOPOLOGY_ROOT/"models/MoDSE.csv", "heterogeneous", tiles * 256
            )
            payload["policy"]["prefetch_window"] = window
            payload["sweep"] = {"prefetch_window": window, "chunk_tiles": tiles}
            save("window_chunk", f"w{window}_c{tiles}", payload)
            save("joint_prefetch", f"w{window}_c{tiles}", payload)

    source = TOPOLOGY_ROOT / "models/Mixtral.csv"
    for top_k in (1,2):
        topo = TOPOLOGY_ROOT/"robustness"/f"topk{top_k}.csv"
        variant_topology(topo, source, balanced(256,8,top_k))
        save("robustness", f"topk_{top_k}",
             generate_topology_runner_payload(topo,"homogeneous",top_k=top_k))
    for tokens in (32,128,256,512):
        topo = TOPOLOGY_ROOT/"robustness"/f"tokens{tokens}.csv"
        variant_topology(topo, source, balanced(tokens,8))
        save("robustness", f"tokens_{tokens}", generate_topology_runner_payload(topo,"homogeneous"))
    for mode in ("balanced","light","high"):
        topo = TOPOLOGY_ROOT/"robustness"/f"routing_{mode}.csv"
        variant_topology(topo, source, skewed(256,mode))
        payload = generate_topology_runner_payload(topo,"homogeneous")
        payload["topology_provenance"]["routing_severity"] = mode
        save("robustness", f"routing_{mode}", payload)
    for mode in ("light","high"):
        for seed in range(40,45):
            topo=TOPOLOGY_ROOT/"robustness"/f"routing_{mode}_seed{seed}.csv"
            variant_topology(topo,source,skewed_seed(256,mode,seed))
            payload=generate_topology_runner_payload(topo,"homogeneous")
            payload["topology_provenance"].update(routing_severity=mode,routing_seed=seed)
            save("robustness",f"routing_{mode}_seed{seed}",payload)
    for model in ("HMoE","Mixtral"):
        save("robustness", f"class_{model}", bases[model])
    for gpus in (1,2):
        save("robustness", f"ep_{gpus}gpu", generate_topology_runner_payload(
            TOPOLOGY_ROOT/"models/MoDSE.csv","heterogeneous",num_gpus=gpus))

    widths=[432,48,384,96,288,192,240,240]
    for experts in (4,8,16):
        counts=balanced(256,experts); topo=TOPOLOGY_ROOT/"robustness"/f"experts_{experts}.csv"
        lines=["Layer,M,N,K,"]
        for expert in range(experts):
            width=widths[expert%8]
            lines += [f"MoE-E{expert}-FF1,{counts[expert]},{width},96,",
                      f"MoE-E{expert}-FF2,{counts[expert]},96,{width},"]
        topo.parent.mkdir(parents=True,exist_ok=True); topo.write_text("\n".join(lines)+"\n",encoding="utf-8")
        save("robustness",f"experts_{experts}",generate_topology_runner_payload(topo,"heterogeneous"))

    # Characterization config/topology namespaces are part of DATE2 too.
    from scripts.DATE2.run_date2_characterization import prepare as prepare_characterization
    prepare_characterization()
    suites={s:len(list((CONFIG_ROOT/s).glob("*.json"))) for s in
            ("overall","window_chunk","joint_prefetch","robustness",
             "robustness_factorial")}
    suites["characterization"]=3
    manifest={"schema_version":2,"simulator_only":True,"rtl_dc_out_of_scope":True,
              "config_root":str(CONFIG_ROOT),"topology_root":str(TOPOLOGY_ROOT),
              "output_root":str(OUTPUT_ROOT),"suites":suites,
              "paper_experiments":{
                  "exp1":"layer_characterization",
                  "exp2":"static_bank_sweep",
                  "exp3":"naive_prefetch_interference",
                  "exp4":"overall",
                  "exp5":"joint_prefetch",
                  "exp6":"robustness_factorial",
              },
              "precision":{
                  "original_model_format":"FP32",
                  "compute_format":"INT8xINT8_INT32",
                  "ia_bytes_per_element":1,
                  "weight_bytes_per_element":1,
                  "accumulator_bytes_per_element":4,
                  "output_bytes_per_element":1,
                  "accumulator_mode":"banked_rmw",
                  "bank_count":30,
                  "bank_width_bits":128,
                  "bank_entries":128,
                  "tile_size":16,
              },
              "network_compatibility":{
                  name:{
                      "hidden_size":item.hidden_size,
                      "total_tokens":item.total_tokens,
                      "expert_count":item.expert_count,
                      "padded_rows":item.padded_rows,
                      "homogeneous_expert_weights":
                          item.homogeneous_expert_weights,
                  } for name,item in compatibility.items()
              },
              "uniform_dimension_divisor":UNIFORM_DIMENSION_DIVISOR}
    (CONFIG_ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (OUTPUT_ROOT/"README.md").write_text(
        "# DATE2 outputs\n\n"
        "Experiments are numbered exp1-exp6; exp5 is Window x Chunk and "
        "exp6 is Robustness.\n\n"
        "Run `python3 run_date2_experiments.py --suite all`.\n",
        encoding="utf-8")

if __name__ == "__main__": main()
