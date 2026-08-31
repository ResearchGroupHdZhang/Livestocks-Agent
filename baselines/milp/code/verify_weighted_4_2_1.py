import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dataLoader import load_datas


FILES = {
    "cn": "中国国家尺度更新PB第一步.xlsx",
    "eu": "欧盟更新PB后第一步.xlsx",
    "aus": "澳大利亚空间优化更新PB第一步.xlsx",
    "usa": "美国数据国家尺度第一步1223.xlsx",
    "br": "巴西指标国家优化更新PB第一步.xlsx",
}


def normalize(value):
    text = "" if pd.isna(value) else str(value)
    return text[:-2] if text.endswith(".0") else text


def id_keys(frame, columns):
    return [tuple(normalize(value) for value in row) for row in frame[columns].to_numpy()]


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else "eu"
    data = load_datas(country, FILES[country])
    ids_in, ids_out, amount_out, amount_in = data[:4]
    n_in, nc_in, ammonia_in, ac_in, sensitivity, pm25 = data[4:10]
    n_out, nc_out, _ammonia_out, _ac_out = data[10:14]
    root = Path(os.environ["WEIGHTED_OUTPUT_DIR"])
    output = root / country
    saved_out = pd.read_excel(output / "MILPresult_move_out.xlsx")
    saved_in = pd.read_excel(output / "MILPresult_move_in.xlsx")
    routes = pd.read_excel(output / "MILPresult_routes.xlsx")
    source_structure = pd.read_excel(output / "MILPresult_source_structure.xlsx")
    summary = json.loads((root / f"{country}.json").read_text(encoding="utf-8"))

    livestock = list(amount_out.columns)
    id_columns = list(ids_out.columns)
    old_out = amount_out.to_numpy(dtype=np.int64)
    old_in = amount_in.to_numpy(dtype=np.int64)
    new_out = saved_out[livestock].to_numpy(dtype=np.int64)
    new_in = saved_in[livestock].to_numpy(dtype=np.int64)
    moved_out = old_out - new_out
    moved_in = new_in - old_in
    nc_out_v = nc_out.to_numpy(dtype=float)
    nc_in_v = nc_in.to_numpy(dtype=float)
    ac_in_v = ac_in.to_numpy(dtype=float)
    n_out_v = n_out.to_numpy(dtype=float).ravel()
    n_in_v = n_in.to_numpy(dtype=float).ravel()
    ammonia_in_v = ammonia_in.to_numpy(dtype=float).ravel()
    final_n_out = n_out_v - (moved_out * nc_out_v).sum(axis=1)
    final_n_in = n_in_v + (moved_in * nc_in_v).sum(axis=1)
    final_ammonia_in = ammonia_in_v - (moved_in * ac_in_v).sum(axis=1)

    out_lookup = {key: i for i, key in enumerate(id_keys(ids_out, id_columns))}
    in_lookup = {key: j for j, key in enumerate(id_keys(ids_in, id_columns))}
    route_out = np.zeros_like(old_out)
    route_in = np.zeros_like(old_in)
    destination_species_structure = True
    for _, route in routes.iterrows():
        i = out_lookup[tuple(normalize(route[f"source_{column}"]) for column in id_columns)]
        j = in_lookup[tuple(normalize(route[f"destination_{column}"]) for column in id_columns)]
        moved = route[livestock].to_numpy(dtype=np.int64)
        destination_species_structure &= bool(np.all((moved == 0) | (old_in[j] > 0)))
        route_out[i] += moved
        route_in[j] += moved

    references = source_structure["reference_removal_fraction"].to_numpy(dtype=float)
    structure_loss = np.zeros(len(old_out))
    share_l1 = np.zeros(len(old_out))
    for i in range(len(old_out)):
        present = old_out[i] > 0
        removal_rates = moved_out[i, present] / old_out[i, present]
        structure_loss[i] = np.abs(removal_rates - references[i]).mean()
        if new_out[i].sum() > 0:
            share_l1[i] = np.abs(old_out[i] / old_out[i].sum() - new_out[i] / new_out[i].sum()).sum()

    environment_coefficients = (
        2 * (1 - sensitivity.to_numpy(dtype=float).ravel())
        + (1 - pm25.to_numpy(dtype=float).ravel())
    )
    environment_score = float((moved_in * environment_coefficients[:, None]).sum())
    environment_upper_bound = float(summary["normalization"]["environment_upper_bound"])
    n_normalized = float((moved_out * nc_out_v).sum() / n_out_v.sum())
    structure_normalized = float(structure_loss.mean())
    environment_normalized = environment_score / environment_upper_bound
    objective = 4 * n_normalized - 2 * structure_normalized + environment_normalized

    scale_source = np.maximum(1e-6, 1e-6 * np.abs(n_out_v))
    scale_destination_n = np.maximum(1e-6, 1e-6 * np.abs(n_in_v))
    scale_destination_ammonia = np.maximum(1e-6, 1e-6 * ammonia_in_v)
    checks = {
        "row_counts": len(saved_out) == len(old_out) and len(saved_in) == len(old_in),
        "id_order": id_keys(saved_out, id_columns) == id_keys(ids_out, id_columns)
        and id_keys(saved_in, id_columns) == id_keys(ids_in, id_columns),
        "integer_nonnegative_moves": bool((moved_out >= 0).all() and (moved_in >= 0).all()),
        "inventory_nonnegative": bool((new_out >= 0).all()),
        "livestock_conservation": bool(np.array_equal(moved_out.sum(axis=0), moved_in.sum(axis=0))),
        "routes_recompute_aggregates": bool(np.array_equal(route_out, moved_out) and np.array_equal(route_in, moved_in)),
        "destination_species_structure": destination_species_structure,
        "source_structure_rows_and_ids": len(source_structure) == len(ids_out)
        and id_keys(source_structure, id_columns) == id_keys(ids_out, id_columns),
        "source_structure_recomputes": bool(
            np.allclose(
                source_structure["mean_abs_removal_rate_deviation"],
                structure_loss,
                atol=1e-9,
            )
            and abs(structure_normalized - summary["source_structure_loss_normalized"]) <= 1e-9
        ),
        "source_share_l1_recomputes": bool(
            np.allclose(source_structure["livestock_share_l1_change"], share_l1, atol=1e-9)
            and abs(share_l1.mean() - summary["source_share_l1_change_mean"]) <= 1e-9
        ),
        "source_n_within_scale_tolerance": bool((final_n_out >= -scale_source).all()),
        "destination_n_within_scale_tolerance": bool((final_n_in <= scale_destination_n).all()),
        "destination_ammonia_within_scale_tolerance": bool((final_ammonia_in >= -scale_destination_ammonia).all()),
        "normalized_components_in_unit_interval": bool(
            0 <= n_normalized <= 1 + 1e-9
            and 0 <= structure_normalized <= 1 + 1e-9
            and 0 <= environment_normalized <= 1 + 1e-9
        ),
        "source_n_recomputes": abs(n_normalized - summary["source_n_resolved_ratio"]) <= 1e-9,
        "environment_recomputes": abs(environment_normalized - summary["environment_score_normalized"]) <= 1e-9,
        "weighted_objective_recomputes": abs(objective - summary["weighted_objective"]) <= 1e-9
        and abs(objective - summary["solver_objective"]) <= 1e-6,
        "saved_source_n_recomputes": bool(np.allclose(saved_out["N_demand"], final_n_out, rtol=1e-9, atol=1e-3)),
        "saved_destination_n_recomputes": bool(np.allclose(saved_in["N_demand"], final_n_in, rtol=1e-9, atol=1e-3)),
        "saved_destination_ammonia_recomputes": bool(
            np.allclose(saved_in["ammonia"], final_ammonia_in, rtol=1e-9, atol=1e-3)
        ),
    }
    report = {
        "country": country,
        "checks": checks,
        "all_passed": all(checks.values()),
        "source_n_resolved_ratio": n_normalized,
        "source_structure_loss_normalized": structure_normalized,
        "source_share_l1_change_mean": float(share_l1.mean()),
        "environment_score_normalized": environment_normalized,
        "weighted_objective": objective,
        "total_moved": int(moved_out.sum()),
        "moved_by_kind": moved_out.sum(axis=0).tolist(),
        "source_n_max_overshoot_kg": float(max(0, -final_n_out.min())),
        "destination_n_max_excess_kg": float(max(0, final_n_in.max())),
        "destination_ammonia_max_excess_kg": float(max(0, -final_ammonia_in.min())),
    }
    (root / f"{country}.validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
