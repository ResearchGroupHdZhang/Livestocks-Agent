import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ["LIVESTOCK_DATA_ROOT"] = str(ROOT.parents[1] / "data")
from dataLoader import load_datas  # noqa: E402


def main():
    comparison = json.loads(
        (ROOT / "analysis" / "weighted_4_2_1_vs_lexicographic.json").read_text(encoding="utf-8")
    )
    weighted = comparison["weighted_4_2_1"]
    lexicographic = comparison["lexicographic"]
    data = load_datas("eu", "欧盟更新PB后第一步.xlsx")
    amount_out = data[2]
    livestock = list(amount_out.columns)
    original = amount_out.to_numpy(dtype=np.int64)

    remain = {}
    for name, result_root in {
        "Weighted 4:2:1": ROOT / "results" / "weighted_4_2_1" / "eu",
        "Lexicographic": ROOT / "results" / "milp" / "eu",
    }.items():
        saved = pd.read_excel(result_root / "MILPresult_move_out.xlsx")[livestock].to_numpy(dtype=np.int64)
        remain[name] = saved.sum(axis=1) / original.sum(axis=1)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    labels = ["N resolved", "Composition\nproxy quality", "Environment"]
    weighted_values = [
        weighted["source_n_resolved_ratio"],
        1 - weighted["source_structure_loss_normalized"],
        weighted["environment_score_normalized"],
    ]
    lex_values = [
        lexicographic["source_n_resolved_ratio"],
        1 - lexicographic["source_structure_loss_normalized"],
        lexicographic["environment_score_normalized"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, lex_values, width, label="Lexicographic", color="#94a3b8")
    axes[0].bar(x + width / 2, weighted_values, width, label="Weighted 4:2:1", color="#0f766e")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Normalized score (higher is better)")
    axes[0].set_title("Objective outcomes")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    bins = np.linspace(0, 1, 21)
    axes[1].hist(remain["Lexicographic"], bins=bins, alpha=0.65, label="Lexicographic", color="#94a3b8")
    axes[1].hist(remain["Weighted 4:2:1"], bins=bins, alpha=0.7, label="Weighted 4:2:1", color="#0f766e")
    axes[1].set_xlabel("Remaining livestock fraction per source")
    axes[1].set_ylabel("Source regions")
    axes[1].set_title("Why share-L1 worsens despite proxy gain")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    figure.tight_layout()
    output = ROOT / "to_human" / "weighted_4_2_1_eu_comparison.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
