"""Generate the DATE3 Exp1--Exp7 paper figure families.

All sources are below outputs/DATE3.  Existing DATE2 figure semantics are
preserved; DATE3-only EP and online-policy evidence is emitted as supplemental
figures rather than silently changing an old axis definition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from scripts.DATE3.experiment_contract import (
    CHUNKS, FIG_ROOT, MODELS, OUTPUT_ROOT, ROBUSTNESS_SCHEMES, WINDOWS,
)

COLORS = {
    "Static-555-NoPF": "#9E9E9E", "Static-Opt-NoPF": "#F28E2B",
    "Dynamic-NoPF": "#4E79A7", "Static-Opt-FixedPF": "#EDC948",
    "Dynamic-FixedPF": "#76B7B2", "PIVOT": "#59A14F",
    "Ideal-NoPF": "#B07AA1",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"run the corresponding DATE3 experiment first: {path}")
    return pd.read_csv(path)


def _save(fig, name: str) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / name, bbox_inches="tight")
    plt.close(fig)


def _heat(ax, frame, title, *, cmap="viridis", center=None, fmt=".1f"):
    values = frame.to_numpy(dtype=float)
    if values.size == 0 or np.isnan(values).all():
        raise ValueError(f"no finite values available for heatmap: {title}")
    if center is None:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
    else:
        bound = max(float(np.nanmax(np.abs(values))), 1e-9)
        image = ax.imshow(values, aspect="auto", cmap=cmap,
                          norm=TwoSlopeNorm(vmin=-bound, vcenter=center, vmax=bound))
    ax.set_xticks(range(len(frame.columns)), frame.columns)
    ax.set_yticks(range(len(frame.index)), frame.index)
    ax.set_xlabel("Chunk size (tiles)")
    ax.set_ylabel("Prefetch Window")
    ax.set_title(title)
    for i in range(len(frame.index)):
        for j in range(len(frame.columns)):
            ax.text(j, i, format(frame.iloc[i, j], fmt), ha="center", va="center",
                    fontsize=7)
    plt.colorbar(image, ax=ax, shrink=.8)


def plot_exp1() -> None:
    data = _read(OUTPUT_ROOT / "exp1/layer_characterization.csv")
    grouped = data.groupby("layer_type", sort=False)[
        ["compute_cycles", "memory_stall_cycles"]
    ].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped.plot(kind="bar", stacked=True, ax=ax,
                 color=["#59A14F", "#F28E2B"])
    ax.set_ylabel("Cycles")
    ax.set_xlabel("")
    ax.set_title("Exp1: per-layer compute and exposed memory stall (P=1)")
    _save(fig, "exp1_cycle_breakdown.pdf")

    stages = data[data.layer_type.str.contains("Expert")].copy()
    stages = stages.sort_values(["layer_type", "layer"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    x = np.arange(len(stages))
    bottom = np.zeros(len(stages))
    for column, label, color in (
        ("ia_bytes", "IA", "#4E79A7"),
        ("weight_bytes", "Weight", "#E15759"),
        ("oa_bytes", "OA", "#76B7B2"),
    ):
        axes[0].bar(x, stages[column], bottom=bottom, label=label, color=color)
        bottom += stages[column].to_numpy()
    for column, label, marker in (
        ("ideal_ia_banks", "IA", "o"),
        ("ideal_weight_banks", "Weight", "s"),
        ("ideal_oa_banks", "OA", "^"),
    ):
        axes[1].plot(x, stages[column], marker=marker, label=label)
    axes[0].set_ylabel("Traffic bytes")
    axes[1].set_ylabel("Ideal SP Banks (sum=15)")
    axes[1].set_xticks(x, stages.layer, rotation=60, ha="right", fontsize=8)
    axes[0].legend(ncol=3); axes[1].legend(ncol=3)
    axes[0].set_title("(a) IA/Weight/OA traffic")
    axes[1].set_title("(b) Per-stage ideal static allocation")
    _save(fig, "exp1_flow_and_ideal_banks.pdf")


def plot_exp2() -> None:
    sweep = _read(OUTPUT_ROOT / "exp2/static_bank_sweep.csv")
    best = _read(OUTPUT_ROOT / "exp2/per_stage_best.csv")
    if len(sweep) != best.layer.nunique() * 91:
        raise ValueError("Exp2 must contain 91 positive IA:Weight:OA partitions/stage")
    ordered = best.sort_values(["layer_type", "layer"])
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(ordered)); bottom = np.zeros(len(ordered))
    for column, label, color in (
        ("ia_banks", "IA", "#4E79A7"),
        ("weight_banks", "Weight", "#E15759"),
        ("oa_banks", "OA", "#76B7B2"),
    ):
        ax.bar(x, ordered[column], bottom=bottom, label=label, color=color)
        bottom += ordered[column].to_numpy()
    ax.set_xticks(x, ordered.layer, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("SP Banks (sum=15)")
    ax.set_title("Exp2: per-stage best static IA:Weight:OA allocation")
    ax.legend(ncol=3)
    _save(fig, "exp2_per_stage_best_ratio.pdf")

    global_result = sweep.groupby(
        ["ia_banks", "weight_banks", "oa_banks"], as_index=False
    ).total_cycles.sum()
    global_result["normalized_cycles"] = (
        global_result.total_cycles / global_result.total_cycles.min()
    )
    pivot = global_result.pivot(
        index="ia_banks", columns="weight_banks", values="normalized_cycles"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot, origin="lower", aspect="auto", cmap="viridis",
                      vmin=1, vmax=np.nanpercentile(pivot, 90))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Weight Banks"); ax.set_ylabel("IA Banks")
    ax.set_title("Global fixed allocation (OA=15-IA-Weight)")
    fig.colorbar(image, ax=ax, label="Normalized cycles")
    _save(fig, "exp2_static_ratio_heatmap.pdf")


def plot_exp3() -> None:
    data = _read(OUTPUT_ROOT / "exp3/naive_prefetch_interference.csv")
    pf = data[data.baseline.eq("Static-NaivePF")].merge(
        data[data.baseline.eq("Static-NoPF")],
        on=["window", "chunk_tiles"], suffixes=("_pf", "_nop"),
        validate="one_to_one",
    )
    pf["cycle_change_pct"] = (pf.total_cycles_pf / pf.total_cycles_nop - 1) * 100
    performance = pf.pivot(index="window", columns="chunk_tiles",
                           values="cycle_change_pct").reindex(WINDOWS, columns=CHUNKS)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    _heat(ax, performance,
          "Exp3: fixed prefetch cycle change vs NoPF (%)\nnegative is better",
          cmap="RdYlGn_r", center=0, fmt="+.1f")
    _save(fig, "exp3_performance_heatmap.pdf")

    pf["occupancy_mbyte_cycles"] = pf.prefetch_occupancy_byte_cycles_pf / 1e6
    metrics = (
        ("timely_prefetch_ratio_pf", "Timely prefetch (%)", 100, "RdYlGn"),
        ("late_prefetch_ratio_pf", "Late prefetch (%)", 100, "YlOrRd"),
        ("occupancy_mbyte_cycles", "Occupancy (MByte-cycles)", 1, "magma"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (column, title, scale, cmap) in zip(axes, metrics):
        frame = pf.pivot(index="window", columns="chunk_tiles", values=column)
        _heat(ax, frame * scale, title, cmap=cmap)
    _save(fig, "exp3_timeliness_occupancy_conflict.pdf")


def plot_exp4() -> None:
    data = _read(OUTPUT_ROOT / "exp4/mapping_comparison.csv")
    order = ["Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF", "Ideal-NoPF"]
    pivot = data.pivot(index="model", columns="policy_name", values="total_cycles")
    pivot = pivot.reindex(MODELS, columns=order)
    normalized = pivot.div(pivot["Static-555-NoPF"], axis=0)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    normalized.plot(kind="bar", ax=ax, width=.86,
                    color=[COLORS[name] for name in order])
    ax.axhline(1, color="black", lw=.8)
    ax.set_ylabel("Normalized system cycles (lower is better)")
    ax.set_xlabel("")
    ax.set_title("Exp4: Static-555, Static-Opt, Dynamic, and Ideal (NoPF)")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, "exp4_mapping_four_way.pdf")

    stalls = data.pivot(index="model", columns="policy_name",
                        values="local_memory_stall_cycles").reindex(MODELS)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    stalls[order].plot(
        kind="bar", ax=ax, color=[COLORS[x] for x in order]
    )
    ax.set_ylabel("Local memory stall cycles")
    ax.set_xlabel("")
    ax.set_title("Local Bank/memory path only (EP wait excluded)")
    _save(fig, "exp4_mapping_stall_and_critical_path.pdf")

    metrics = pd.DataFrame(index=MODELS)
    metrics["Static tuning"] = (1-pivot["Static-Opt-NoPF"]/pivot["Static-555-NoPF"])*100
    metrics["Dynamic over Static-Opt"] = (1-pivot["Dynamic-NoPF"]/pivot["Static-Opt-NoPF"])*100
    metrics["Dynamic gap to Ideal"] = (pivot["Dynamic-NoPF"]/pivot["Ideal-NoPF"]-1)*100
    fig, ax = plt.subplots(figsize=(10, 4.8))
    metrics.plot(kind="bar", ax=ax, color=["#F28E2B", "#4E79A7", "#B07AA1"])
    ax.set_ylabel("Percent (%)"); ax.set_xlabel("")
    ax.set_title("Exp4 mapping benefit decomposition")
    _save(fig, "exp4_mapping_benefit_decomposition.pdf")


def plot_exp5() -> None:
    data = _read(OUTPUT_ROOT / "exp5/joint_prefetch.csv")
    def grid(name, field="total_cycles"):
        return data[data.policy_name.eq(name)].pivot(
            index="window", columns="chunk_tiles", values=field
        ).reindex(WINDOWS, columns=CHUNKS)
    static_gain = (1 - grid("Static-Opt-FixedPF") / grid("Static-Opt-NoPF")) * 100
    dynamic_gain = (1 - grid("Dynamic-FixedPF") / grid("Dynamic-NoPF")) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    _heat(axes[0], static_gain, "(a) Fixed prefetch gain: static mapping (%)",
          cmap="RdYlGn", center=0)
    _heat(axes[1], dynamic_gain, "(b) Fixed prefetch gain: dynamic mapping (%)",
          cmap="RdYlGn", center=0)
    _save(fig, "exp5_prefetch_tradeoff.pdf")

    conventional = ["Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF",
                    "Static-Opt-FixedPF", "Dynamic-FixedPF"]
    wide = data.pivot(index=["window", "chunk_tiles"],
                      columns="policy_name", values="total_cycles")
    best = wide[conventional].min(axis=1)
    final = (1 - wide["PIVOT"] / best) * 100
    versus_static = (1 - grid("PIVOT") / grid("Static-555-NoPF")) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    _heat(axes[0], versus_static, "(c) PIVOT vs Static-555 (%)", cmap="YlGn")
    _heat(axes[1], final.unstack("chunk_tiles"),
          "(d) PIVOT vs best conventional (%)", cmap="RdYlGn", center=0,
          fmt=".2f")
    _save(fig, "exp5_public_sensitivity.pdf")

    pivot_rows = data[data.policy_name.eq("PIVOT")].copy()
    best_variant = pivot_rows.loc[pivot_rows.total_cycles.idxmin(), "variant"]
    directory = OUTPUT_ROOT / "joint_prefetch" / best_variant
    epochs = _read(directory / "quality_epochs.csv")
    decisions = _read(directory / "decision_detail.csv")
    decisions = decisions[decisions.selected.astype(str).str.lower().eq("true")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(epochs.epoch_id, epochs.coverage, "o-", label="Coverage")
    axes[0].plot(epochs.epoch_id, epochs.accuracy, "s-", label="Accuracy")
    axes[0].set_ylim(0, 1.05); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Measured byte ratio"); axes[0].legend()
    axes[1].step(decisions.decision_id, decisions.candidate_chunk,
                 where="mid", label="Chunk")
    axes[1].step(decisions.decision_id, decisions.candidate_window,
                 where="mid", label="Window")
    axes[1].set_xlabel("Decision"); axes[1].set_ylabel("Selected value")
    axes[1].legend()
    fig.suptitle(f"Supplement: online trajectory ({best_variant})")
    _save(fig, "exp5_online_adaptation.pdf")

    guard = _read(directory / "online_incumbent_guard.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(guard.epoch_id, guard.proposal_prefix_cost_cycles, "o-",
            label="Adaptive proposal")
    ax.plot(guard.epoch_id, guard.fixed_prefix_cost_cycles, "s-",
            label="Fixed-prefetch incumbent")
    ax.plot(guard.epoch_id, guard.noprefetch_prefix_cost_cycles, "^-",
            label="NoPF incumbent")
    ax.plot(guard.epoch_id, guard.applied_prefix_cost_cycles, "k--",
            label="Applied three-way minimum")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Prefix memory cost (cycles)")
    ax.set_title("Supplement: pre-issue online incumbent protection")
    ax.legend()
    _save(fig, "exp5_online_guard.pdf")

    ablation = _read(OUTPUT_ROOT / "exp5/pivot_ca_ablation.csv")
    full = float(ablation.loc[ablation.variant.eq("full"), "total_cycles"].iloc[0])
    ablation["slowdown_vs_full_pct"] = (ablation.total_cycles / full - 1) * 100
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(ablation.variant, ablation.slowdown_vs_full_pct, color="#4E79A7")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Slowdown vs full PIVOT (%)")
    ax.set_title("Exp5 supplement: PIVOT policy ablation (MoDSE)")
    _save(fig, "exp5_pivot_ca_ablation.pdf")


def plot_exp6() -> None:
    data = _read(OUTPUT_ROOT / "exp6/robustness_comparison.csv")
    for variable, group in data.groupby("variable", sort=False):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=False)
        for ax, model in zip(axes, MODELS):
            current = group[group.model.eq(model)].copy()
            values = list(dict.fromkeys(current.value.astype(str)))
            x = np.arange(len(values)); width = .13
            static = (current[current.policy_name.eq("Static-555-NoPF")]
                      .assign(value_key=lambda frame: frame.value.astype(str))
                      .set_index("value_key").total_cycles)
            for index, scheme in enumerate(ROBUSTNESS_SCHEMES):
                q = current[current.policy_name.eq(scheme)].copy()
                q = q.assign(value_key=q.value.astype(str)).set_index("value_key")
                normalized = q.total_cycles.reindex(values).astype(float) / static.reindex(values).astype(float)
                ax.bar(x + (index - (len(ROBUSTNESS_SCHEMES)-1)/2) * width, normalized, width,
                       color=COLORS[scheme], label=scheme)
            ax.axhline(1, color="black", lw=.8)
            ax.set_xticks(x, values, rotation=25)
            ax.set_title(model); ax.set_xlabel(variable)
            ax.set_ylabel("Cycles / Static-555")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=6)
        fig.suptitle(f"Exp6 sensitivity: {variable}", y=1.04)
        _save(fig, f"exp6_{variable}.pdf")

    wide = data.pivot(index=["variable", "value", "model"],
                      columns="policy_name", values="total_cycles")
    diagnosis = pd.DataFrame(index=wide.index)
    diagnosis["dynamic_slower_than_static"] = (
        wide["Dynamic-NoPF"] > wide["Static-Opt-NoPF"]
    )
    diagnosis["pivot_not_best"] = wide["PIVOT"] > wide.min(axis=1)
    counts = diagnosis.groupby(level=0).sum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    counts.plot(kind="bar", ax=ax, color=["#E15759", "#F28E2B"])
    ax.set_ylabel("Configuration count")
    ax.set_xlabel("")
    ax.set_title("Supplement: sensitivity contract/failure diagnosis")
    _save(fig, "exp6_failure_diagnosis.pdf")


def plot_exp7() -> None:
    data = _read(OUTPUT_ROOT / "exp7/end_to_end_summary.csv")
    order = ["Static-555-NoPF", "Static-Opt-NoPF", "Dynamic-NoPF",
             "Static-Opt-FixedPF", "Dynamic-FixedPF", "PIVOT"]
    speedup = data.pivot(
        index="model", columns="policy_name",
        values="end_to_end_speedup_vs_static",
    ).reindex(MODELS, columns=order)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    speedup.plot(kind="bar", ax=ax, width=.86,
                 color=[COLORS[name] for name in order])
    ax.axhline(1, color="black", lw=.8)
    ax.set_ylabel("Approx. block speedup vs Static-555")
    ax.set_xlabel("")
    ax.set_title("Exp7: end-to-end speedup of four complete MoE Transformer blocks")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, "exp7_end_to_end_speedup.pdf")

    selected = data[data.policy_name.isin(["Static-555-NoPF", "PIVOT"])].copy()
    selected["label"] = selected.model + "\n" + selected.policy_name
    selected = selected.set_index("label").loc[
        [f"{model}\n{policy}" for model in MODELS
         for policy in ("Static-555-NoPF", "PIVOT")]
    ]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    selected[["non_moe_full_cycles", "moe_ep_cycles"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#B9D7EA", "#E15759"],
    )
    ax.set_ylabel("Approx. block cycles")
    ax.set_xlabel("")
    ax.set_title("End-to-end decomposition: invariant non-MoE + optimized MoE/EP")
    ax.legend(["Non-MoE full path", "MoE + EP critical path"])
    _save(fig, "exp7_block_decomposition.pdf")

    layers = _read(OUTPUT_ROOT / "exp7/non_moe_layer_breakdown.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=False)
    for ax, model in zip(axes.flat, MODELS):
        current = layers[layers.model.eq(model)].set_index("layer")
        current[["compute_cycles", "memory_stall_cycles"]].plot(
            kind="bar", stacked=True, ax=ax,
            color=["#59A14F", "#F28E2B"], legend=False,
        )
        ax.set_title(model)
        ax.set_xlabel("")
        ax.set_ylabel("Cycles")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Exp7 supplement: non-MoE layer cost used by the approximation")
    _save(fig, "exp7_non_moe_layer_breakdown.pdf")


PLOTS = {
    "exp1": plot_exp1, "exp2": plot_exp2, "exp3": plot_exp3,
    "exp4": plot_exp4, "exp5": plot_exp5, "exp6": plot_exp6,
    "exp7": plot_exp7,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=("all", *PLOTS), required=True)
    args = parser.parse_args()
    selected = PLOTS if args.exp == "all" else {args.exp: PLOTS[args.exp]}
    for name, function in selected.items():
        function()
        print(f"plotted DATE3 {name}")


if __name__ == "__main__":
    main()
