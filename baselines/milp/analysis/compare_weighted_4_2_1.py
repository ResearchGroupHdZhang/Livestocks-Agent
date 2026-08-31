import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ["LIVESTOCK_DATA_ROOT"] = str(ROOT.parents[1] / "data")

from dataLoader import load_datas  # noqa: E402


def method_metrics(saved_out, saved_in, structure, data, environment_upper_bound):
    amount_out, amount_in = data[2:4]
    sensitivity, pm25 = data[8:10]
    n_out, nc_out = data[10:12]
    livestock = list(amount_out.columns)
    old_out = amount_out.to_numpy(dtype=np.int64)
    old_in = amount_in.to_numpy(dtype=np.int64)
    new_out = saved_out[livestock].to_numpy(dtype=np.int64)
    new_in = saved_in[livestock].to_numpy(dtype=np.int64)
    moved_out = old_out - new_out
    moved_in = new_in - old_in
    references = structure["reference_removal_fraction"].to_numpy(dtype=float)
    losses = []
    share_l1 = []
    for i in range(len(old_out)):
        present = old_out[i] > 0
        rates = moved_out[i, present] / old_out[i, present]
        losses.append(float(np.abs(rates - references[i]).mean()))
        share_l1.append(
            float(np.abs(old_out[i] / old_out[i].sum() - new_out[i] / new_out[i].sum()).sum())
            if new_out[i].sum() > 0
            else 0.0
        )
    environment_coefficients = (
        2 * (1 - sensitivity.to_numpy(dtype=float).ravel())
        + (1 - pm25.to_numpy(dtype=float).ravel())
    )
    source_n = float((moved_out * nc_out.to_numpy(dtype=float)).sum())
    source_n_ratio = source_n / float(n_out.to_numpy(dtype=float).sum())
    structure_loss = float(np.mean(losses))
    environment_score = float((moved_in * environment_coefficients[:, None]).sum())
    environment_normalized = environment_score / environment_upper_bound
    weighted_objective = 4 * source_n_ratio - 2 * structure_loss + environment_normalized
    return {
        "source_n_resolved_kg": source_n,
        "source_n_resolved_ratio": source_n_ratio,
        "source_structure_loss_normalized": structure_loss,
        "source_structure_loss_max": float(np.max(losses)),
        "source_share_l1_change_mean": float(np.mean(share_l1)),
        "source_share_l1_change_max": float(np.max(share_l1)),
        "environment_score": environment_score,
        "environment_score_normalized": environment_normalized,
        "weighted_objective_4_2_1": weighted_objective,
        "total_moved": int(moved_out.sum()),
        "moved_by_kind": {column: int(value) for column, value in zip(livestock, moved_out.sum(axis=0))},
    }


def source_structure_for(saved_out, data):
    ids_out, amount_out = data[1], data[2]
    livestock = list(amount_out.columns)
    old_out = amount_out.to_numpy(dtype=np.int64)
    new_out = saved_out[livestock].to_numpy(dtype=np.int64)
    moved_out = old_out - new_out
    references = []
    for i in range(len(old_out)):
        present = old_out[i] > 0
        references.append(float(np.median(moved_out[i, present] / old_out[i, present])))
    return pd.concat([
        ids_out.reset_index(drop=True),
        pd.DataFrame({"reference_removal_fraction": references}),
    ], axis=1)


def main():
    data = load_datas("eu", "欧盟更新PB后第一步.xlsx")
    weighted_root = ROOT / "results" / "weighted_4_2_1"
    weighted_summary = json.loads((weighted_root / "eu.json").read_text(encoding="utf-8"))
    environment_upper_bound = float(weighted_summary["normalization"]["environment_upper_bound"])

    weighted = method_metrics(
        pd.read_excel(weighted_root / "eu" / "MILPresult_move_out.xlsx"),
        pd.read_excel(weighted_root / "eu" / "MILPresult_move_in.xlsx"),
        pd.read_excel(weighted_root / "eu" / "MILPresult_source_structure.xlsx"),
        data,
        environment_upper_bound,
    )

    lex_root = ROOT / "results" / "milp"
    lex_out = pd.read_excel(lex_root / "eu" / "MILPresult_move_out.xlsx")
    lexicographic = method_metrics(
        lex_out,
        pd.read_excel(lex_root / "eu" / "MILPresult_move_in.xlsx"),
        source_structure_for(lex_out, data),
        data,
        environment_upper_bound,
    )

    comparisons = []
    for metric in (
        "source_n_resolved_kg",
        "source_n_resolved_ratio",
        "source_structure_loss_normalized",
        "source_share_l1_change_mean",
        "environment_score",
        "environment_score_normalized",
        "weighted_objective_4_2_1",
        "total_moved",
    ):
        weighted_value = float(weighted[metric])
        lex_value = float(lexicographic[metric])
        comparisons.append({
            "metric": metric,
            "weighted_4_2_1": weighted_value,
            "lexicographic": lex_value,
            "weighted_minus_lexicographic": weighted_value - lex_value,
            "percent_change_vs_lexicographic": 100 * (weighted_value - lex_value) / lex_value
            if lex_value
            else None,
        })

    report = {
        "scope": {
            "dataset": "欧盟更新PB后第一步.xlsx",
            "weighted_objective": "4*N_hat - 2*L_hat + E_hat",
            "comparison": "corrected source-region composition metrics recomputed from both serialized solutions",
        },
        "weighted_4_2_1_solver": weighted_summary["solve"],
        "weighted_4_2_1_timing": {
            key: weighted_summary[key]
            for key in (
                "input_load_elapsed_seconds",
                "build_elapsed_seconds",
                "serialization_elapsed_seconds",
                "total_elapsed_seconds",
                "peak_rss_mb",
            )
        },
        "weighted_4_2_1": weighted,
        "lexicographic": lexicographic,
        "comparison": comparisons,
    }
    analysis = ROOT / "analysis"
    (analysis / "weighted_4_2_1_vs_lexicographic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(comparisons).to_csv(
        analysis / "weighted_4_2_1_vs_lexicographic.csv",
        index=False,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
