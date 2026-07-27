"""Generate four-model, one-variable-at-a-time DATE2 exp6 configs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scalesim.memory.topology_workload import generate_topology_runner_payload
from scripts.prepare_date2_experiments import (
    CONFIG_ROOT,
    MODELS,
    TOPOLOGY_ROOT,
    save,
    skewed,
    skewed_seed,
)

SUITE = "robustness_factorial"
BASE_COUNTS = (32, 48, 50, 24, 34, 28, 21, 19)
VARIABLE_LEVELS = {
    "expert_count": ("4", "8", "16"),
    "token_count": ("32", "128", "256", "512"),
    "top_k": ("1", "2"),
    "expert_parallel": ("1", "2"),
    "routing_severity": ("balanced", "light", "high"),
    "routing_seed": tuple(
        f"{severity}_seed{seed}"
        for severity in ("light", "high")
        for seed in range(40, 45)
    ),
}


def proportional_counts(total: int, experts: int):
    weights = [BASE_COUNTS[index % len(BASE_COUNTS)] for index in range(experts)]
    raw = [total * weight / sum(weights) for weight in weights]
    counts = [int(value) for value in raw]
    order = sorted(
        range(experts), key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return tuple(counts)


def write_variant(
    source: Path, target: Path, counts, token_count: int, expert_count: int
):
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    expert_templates = {}
    retained = []
    for row in rows:
        if not row or not row[0]:
            continue
        name = row[0]
        if name.startswith("MoE-E"):
            expert = int(name.split("-")[1][1:])
            part = name.rsplit("-", 1)[1]
            expert_templates[(expert, part)] = row
            continue
        copied = list(row)
        if name != "Layer":
            copied[1] = str(token_count)
            if name == "Router_logits":
                copied[2] = str(expert_count)
        retained.append(copied)
    if not expert_templates:
        raise ValueError(f"no MoE expert rows in {source}")
    template_experts = 1 + max(index for index, _ in expert_templates)
    generated = []
    for expert in range(expert_count):
        template = expert % template_experts
        for part in ("FF1", "FF2"):
            row = list(expert_templates[(template, part)])
            row[0] = f"MoE-E{expert}-{part}"
            row[1] = str(counts[expert])
            generated.append(row)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([*retained, *generated])


def emit(model, kind, variable, value, topology, *, top_k=1, num_gpus=1):
    payload = generate_topology_runner_payload(
        topology, kind, top_k=top_k, num_gpus=num_gpus
    )
    payload["sweep"] = {
        "variable": variable,
        "value": str(value),
        "model": model,
    }
    payload["topology_provenance"].update({
        "exp6_model": model,
        "exp6_variable": variable,
        "exp6_value": str(value),
    })
    save(SUITE, f"{variable}__{model}__{value}", payload)


def main():
    topology_root = TOPOLOGY_ROOT / "robustness_factorial"
    for model, kind in MODELS:
        source = TOPOLOGY_ROOT / "models" / f"{model}.csv"

        for experts in (4, 8, 16):
            counts = proportional_counts(256, experts)
            target = topology_root / "expert_count" / f"{model}__{experts}.csv"
            write_variant(source, target, counts, 256, experts)
            emit(model, kind, "expert_count", experts, target)

        for tokens in (32, 128, 256, 512):
            counts = proportional_counts(tokens, 8)
            target = topology_root / "token_count" / f"{model}__{tokens}.csv"
            write_variant(source, target, counts, tokens, 8)
            emit(model, kind, "token_count", tokens, target)

        for top_k in (1, 2):
            counts = proportional_counts(256 * top_k, 8)
            target = topology_root / "top_k" / f"{model}__{top_k}.csv"
            write_variant(source, target, counts, 256, 8)
            emit(model, kind, "top_k", top_k, target, top_k=top_k)

        for gpus in (1, 2):
            emit(
                model, kind, "expert_parallel", gpus, source, num_gpus=gpus
            )

        for severity in ("balanced", "light", "high"):
            counts = skewed(256, severity)
            target = (
                topology_root / "routing_severity"
                / f"{model}__{severity}.csv"
            )
            write_variant(source, target, counts, 256, 8)
            emit(model, kind, "routing_severity", severity, target)

        for severity in ("light", "high"):
            for seed in range(40, 45):
                value = f"{severity}_seed{seed}"
                counts = skewed_seed(256, severity, seed)
                target = (
                    topology_root / "routing_seed" / f"{model}__{value}.csv"
                )
                write_variant(source, target, counts, 256, 8)
                emit(model, kind, "routing_seed", value, target)

    configs = sorted((CONFIG_ROOT / SUITE).glob("*.json"))
    expected = len(MODELS) * sum(map(len, VARIABLE_LEVELS.values()))
    if len(configs) != expected:
        raise AssertionError(f"expected {expected} configs, found {len(configs)}")
    manifest_path = CONFIG_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["suites"][SUITE] = expected
    manifest["paper_experiments"]["exp6"] = SUITE
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {expected} exp6 factorial configurations")


if __name__ == "__main__":
    main()
