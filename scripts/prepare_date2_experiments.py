"""Generate DATE2 simulator configs, topologies, and output hierarchy."""
from __future__ import annotations
import csv, json, shutil, sys
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scalesim.memory.moe_workload_catalog import write_runner_payload
from scalesim.memory.topology_workload import generate_topology_runner_payload
CONFIG_ROOT = ROOT / "configs/MoE/DATE2"
TOPOLOGY_ROOT = ROOT / "topologies/MoE/DATE2"
OUTPUT_ROOT = ROOT / "outputs/DATE2"
MODELS = (("HMoE", "heterogeneous"), ("Mixtral", "homogeneous"),
          ("MoDSE", "heterogeneous"), ("Switchtrans", "homogeneous"))

def balanced(tokens, experts, top_k=1):
    total = tokens * top_k
    return tuple(total // experts + (i < total % experts) for i in range(experts))

def skewed(tokens, severity):
    weights = {"balanced": [1]*8, "light": [4,3,2,2,1,1,1,1],
               "high": [16,8,4,2,1,1,1,1]}[severity]
    values = [tokens*w//sum(weights) for w in weights]
    for i in range(tokens-sum(values)): values[i] += 1
    return tuple(values)

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

def save(suite, name, payload):
    payload = deepcopy(payload)
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
        shutil.copyfile(ROOT / f"topologies/MoE/{model}.csv", target)
        bases[model] = generate_topology_runner_payload(target, kind)
        save("overall", model, bases[model])

    payload = deepcopy(bases["MoDSE"])
    payload["ablation_order"] = ["Static-NoPF", "Dynamic-NoPF", "Static-NaivePF",
                                 "Dynamic-NaivePF", "MemDomain-Raw", "MemDomain-Safe"]
    save("ablation", "MoDSE_components", payload)

    for window in (0,1,2,4,8):
        for tiles in (1,2,4,8):
            payload = generate_topology_runner_payload(TOPOLOGY_ROOT/"models/MoDSE.csv",
                                                       "heterogeneous", tiles*16*1024)
            payload["policy"]["prefetch_window"] = window
            payload["sweep"] = {"prefetch_window": window, "chunk_tiles": tiles}
            save("window_chunk", f"w{window}_c{tiles}", payload)

    source = ROOT / "topologies/MoE/Mixtral.csv"
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
    for model in ("HMoE","Mixtral"):
        save("robustness", f"class_{model}", bases[model])
    for gpus in (1,2):
        save("robustness", f"ep_{gpus}gpu", generate_topology_runner_payload(
            TOPOLOGY_ROOT/"models/MoDSE.csv","heterogeneous",num_gpus=gpus))

    widths=[1728,192,1536,384,1152,768,960,960]
    for experts in (4,8,16):
        counts=balanced(256,experts); topo=TOPOLOGY_ROOT/"robustness"/f"experts_{experts}.csv"
        lines=["Layer,M,N,K,"]
        for expert in range(experts):
            width=widths[expert%8]
            lines += [f"MoE-E{expert}-FF1,{counts[expert]},{width},384,",
                      f"MoE-E{expert}-FF2,{counts[expert]},384,{width},"]
        topo.parent.mkdir(parents=True,exist_ok=True); topo.write_text("\n".join(lines)+"\n",encoding="utf-8")
        save("robustness",f"experts_{experts}",generate_topology_runner_payload(topo,"heterogeneous"))

    suites={s:len(list((CONFIG_ROOT/s).glob("*.json"))) for s in
            ("overall","ablation","window_chunk","robustness")}
    manifest={"schema_version":1,"simulator_only":True,"rtl_dc_out_of_scope":True,
              "config_root":str(CONFIG_ROOT),"topology_root":str(TOPOLOGY_ROOT),
              "output_root":str(OUTPUT_ROOT),"suites":suites}
    (CONFIG_ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (OUTPUT_ROOT/"README.md").write_text("# DATE2 outputs\n\nRun `python3 run_date2_experiments.py --suite all`.\n",encoding="utf-8")

if __name__ == "__main__": main()
