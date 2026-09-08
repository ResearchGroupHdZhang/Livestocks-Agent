import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"ppo": "#D97706", "milp": "#0F766E"}
LABELS = {"ppo": "Paper PPO", "milp": "Certified MILP"}


def main():
    report = json.loads((ROOT / "analysis" / "paper_ppo_vs_weighted_milp.json").read_text(encoding="utf-8"))
    countries = ["eu", "usa", "aus"]
    display = {"eu": "EU\n(v9)", "usa": "USA\n(v9)", "aus": "Australia\n(v8*)"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))

    panels = [
        ("source_n_resolved_ratio", "Source N resolved", "Higher is better", (0, 1.08)),
        ("environment_score_normalized", "Normalized environmental score", "Higher is better", (0, 0.9)),
        ("weighted_objective_4_2_1_posthoc", "Post-hoc common score J", "Higher is better; not PPO reward", (0, 5.0)),
        ("total_moved", "Animals moved", "Descriptive only; not an optimized objective", None),
    ]
    x = np.arange(len(countries))
    width = 0.34
    for ax, (metric, title, subtitle, ylim) in zip(axes.ravel(), panels):
        for offset, method in [(-width / 2, "ppo"), (width / 2, "milp")]:
            values = [report["results"][country][method]["metrics"][metric] for country in countries]
            bars = ax.bar(x + offset, values, width, color=COLORS[method], label=LABELS[method])
            for bar, value in zip(bars, values):
                label = f"{value:.3f}" if metric != "total_moved" else f"{value / 1e9:.2f}B"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [display[country] for country in countries])
        ax.set_title(f"{title}\n", fontsize=11, weight="bold", pad=8)
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#475569")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Paper-method PPO versus certified weighted MILP\nMatched first-stage national inputs; independently recomputed outcomes", fontsize=14, weight="bold")
    figure.text(
        0.5,
        0.025,
        "Caution: PPO's native action-local reward differs from the global MILP objective. J is a post-hoc common rescore.",
        ha="center",
        fontsize=9,
        color="#7F1D1D",
    )
    figure.text(0.5, 0.008, "* Australia is an input-matched, version-qualified PPO v8 robustness comparison.", ha="center", fontsize=8, color="#475569")
    figure.tight_layout(rect=(0, 0.06, 1, 0.87))
    output = ROOT / "to_human" / "paper_ppo_vs_weighted_milp.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
