import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ["LIVESTOCK_DATA_ROOT"] = str(REPO / "data")

from dataLoader import load_datas  # noqa: E402


CASES = {
    "eu": {
        "dataset": "欧盟更新PB后第一步.xlsx",
        "ppo_version": "v9",
        "ppo_root": REPO / "results" / "v9" / "eu",
        "evidence_tier": "confirmatory_same_ppo_version",
    },
    "usa": {
        "dataset": "美国数据国家尺度第一步1223.xlsx",
        "ppo_version": "v9",
        "ppo_root": REPO / "results" / "v9" / "usa",
        "evidence_tier": "confirmatory_same_ppo_version",
    },
    "aus": {
        "dataset": "澳大利亚空间优化更新PB第一步.xlsx",
        "ppo_version": "v8",
        "ppo_root": REPO / "results" / "v8" / "aus",
        "evidence_tier": "version_qualified_robustness",
    },
}


def normalize(value):
    text = "" if pd.isna(value) else str(value)
    return text[:-2] if text.endswith(".0") else text


def id_keys(frame, columns):
    return [tuple(normalize(value) for value in row) for row in frame[columns].to_numpy()]


def source_metrics(old_out, new_out, moved_out, references=None):
    losses = np.zeros(len(old_out), dtype=float)
    share_l1 = np.zeros(len(old_out), dtype=float)
    fitted_references = np.zeros(len(old_out), dtype=float)
    remaining_fraction = np.zeros(len(old_out), dtype=float)
    for i in range(len(old_out)):
        present = old_out[i] > 0
        rates = moved_out[i, present] / old_out[i, present]
        fitted_references[i] = float(np.median(rates)) if references is None else float(references[i])
        losses[i] = float(np.abs(rates - fitted_references[i]).mean())
        remaining_fraction[i] = float(new_out[i].sum() / old_out[i].sum())
        if new_out[i].sum() > 0:
            before = old_out[i] / old_out[i].sum()
            after = new_out[i] / new_out[i].sum()
            share_l1[i] = float(np.abs(before - after).sum())
    return {
        "source_structure_loss_normalized": float(losses.mean()),
        "source_structure_loss_max": float(losses.max()),
        "source_share_l1_change_mean": float(share_l1.mean()),
        "source_share_l1_change_max": float(share_l1.max()),
        "sources_remaining_below_10pct": int((remaining_fraction < 0.1).sum()),
        "sources_fully_depleted": int((new_out.sum(axis=1) == 0).sum()),
        "reference_removal_fraction_mean": float(fitted_references.mean()),
    }


def evaluate(country, method, saved_out, saved_in, routes, data, environment_upper_bound, structure=None):
    ids_in, ids_out, amount_out, amount_in = data[:4]
    n_in, nc_in, ammonia_in, ac_in, sensitivity, pm25 = data[4:10]
    n_out, nc_out = data[10:12]
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

    out_keys = id_keys(ids_out, id_columns)
    in_keys = id_keys(ids_in, id_columns)
    out_lookup = {key: i for i, key in enumerate(out_keys)}
    in_lookup = {key: j for j, key in enumerate(in_keys)}
    route_out = np.zeros_like(old_out)
    route_in = np.zeros_like(old_in)
    route_nonnegative_integer = True
    route_ids_valid = True
    destination_species_structure = True
    nonzero_routes = 0
    reward_sum = 0.0
    for _, route in routes.iterrows():
        try:
            if method == "milp":
                i = out_lookup[tuple(normalize(route[f"source_{column}"]) for column in id_columns)]
                j = in_lookup[tuple(normalize(route[f"destination_{column}"]) for column in id_columns)]
            else:
                i, j = int(route["move_out_idx"]), int(route["move_in_idx"])
                route_ids_valid &= 0 <= i < len(old_out) and 0 <= j < len(old_in)
                if route_ids_valid:
                    route_ids_valid &= tuple(normalize(route[f"{column}_move_out"]) for column in id_columns) == out_keys[i]
                    route_ids_valid &= tuple(normalize(route[f"{column}_move_in"]) for column in id_columns) == in_keys[j]
        except (KeyError, ValueError, IndexError):
            route_ids_valid = False
            continue
        raw = route[livestock].to_numpy(dtype=float)
        route_nonnegative_integer &= bool(np.isfinite(raw).all() and (raw >= 0).all() and np.equal(raw, np.rint(raw)).all())
        moved = np.rint(raw).astype(np.int64)
        if moved.sum() > 0:
            nonzero_routes += 1
        destination_species_structure &= bool(np.all((moved == 0) | (old_in[j] > 0)))
        route_out[i] += moved
        route_in[j] += moved
        if "reward" in route:
            reward_sum += float(route["reward"])

    references = structure["reference_removal_fraction"].to_numpy(dtype=float) if structure is not None else None
    source = source_metrics(old_out, new_out, moved_out, references)
    sensitivity_reward = float((moved_in * (1 - sensitivity.to_numpy(dtype=float).ravel())[:, None]).sum())
    pm25_reward = float((moved_in * (1 - pm25.to_numpy(dtype=float).ravel())[:, None]).sum())
    environment_score = 2 * sensitivity_reward + pm25_reward
    source_n_resolved = float((moved_out * nc_out_v).sum())
    source_n_total = float(n_out_v.sum())
    n_normalized = source_n_resolved / source_n_total
    environment_normalized = environment_score / environment_upper_bound
    weighted_objective = 4 * n_normalized - 2 * source["source_structure_loss_normalized"] + environment_normalized
    total_moved = int(moved_out.sum())

    source_scale = np.maximum(1e-6, 1e-6 * np.abs(n_out_v))
    destination_n_scale = np.maximum(1e-6, 1e-6 * np.abs(n_in_v))
    destination_ammonia_scale = np.maximum(1e-6, 1e-6 * np.abs(ammonia_in_v))
    checks = {
        "row_counts": len(saved_out) == len(old_out) and len(saved_in) == len(old_in),
        "id_order": id_keys(saved_out, id_columns) == out_keys and id_keys(saved_in, id_columns) == in_keys,
        "inventory_nonnegative": bool((new_out >= 0).all() and (new_in >= 0).all()),
        "integer_nonnegative_aggregate_moves": bool((moved_out >= 0).all() and (moved_in >= 0).all()),
        "species_conservation": bool(np.array_equal(moved_out.sum(axis=0), moved_in.sum(axis=0))),
        "route_ids_valid": bool(route_ids_valid),
        "route_integer_nonnegative": bool(route_nonnegative_integer),
        "routes_recompute_outbound": bool(np.array_equal(route_out, moved_out)),
        "routes_recompute_inbound": bool(np.array_equal(route_in, moved_in)),
        "destination_species_structure": bool(destination_species_structure),
        "source_n_within_scale_tolerance": bool((final_n_out >= -source_scale).all()),
        "destination_n_within_scale_tolerance": bool((final_n_in <= destination_n_scale).all()),
        "destination_ammonia_within_scale_tolerance": bool((final_ammonia_in >= -destination_ammonia_scale).all()),
        "normalized_components_in_unit_interval": bool(
            0 <= n_normalized <= 1 + 1e-9
            and 0 <= source["source_structure_loss_normalized"] <= 1 + 1e-9
            and 0 <= environment_normalized <= 1 + 1e-9
        ),
    }
    physically_validated = all(checks.values())
    metrics = {
        "source_n_total_kg": source_n_total,
        "source_n_resolved_kg": source_n_resolved,
        "source_n_resolved_ratio": n_normalized,
        "source_n_remaining_kg": float(final_n_out.sum()),
        "source_n_90pct_target_rate": float((final_n_out <= 0.1 * n_out_v + source_scale).mean()),
        "source_n_strict_negative_rows": int((final_n_out < 0).sum()),
        "source_n_scale_tolerance_violations": int((final_n_out < -source_scale).sum()),
        "source_n_max_overshoot_kg": float(max(0, -final_n_out.min())),
        "destination_n_strict_positive_rows": int((final_n_in > 0).sum()),
        "destination_n_scale_tolerance_violations": int((final_n_in > destination_n_scale).sum()),
        "destination_n_max_excess_kg": float(max(0, final_n_in.max())),
        "destination_ammonia_strict_negative_rows": int((final_ammonia_in < 0).sum()),
        "destination_ammonia_scale_tolerance_violations": int((final_ammonia_in < -destination_ammonia_scale).sum()),
        "destination_ammonia_max_excess_kg": float(max(0, -final_ammonia_in.min())),
        "sensitivity_reward": sensitivity_reward,
        "pm25_reward": pm25_reward,
        "environment_score": environment_score,
        "environment_score_normalized": environment_normalized,
        "weighted_objective_4_2_1_posthoc": weighted_objective,
        "total_moved": total_moved,
        "n_resolved_per_moved_animal_kg": source_n_resolved / total_moved if total_moved else None,
        "environment_score_per_moved_animal": environment_score / total_moved if total_moved else None,
        "nonzero_routes": nonzero_routes,
        "archived_route_reward_sum": reward_sum if method == "ppo" else None,
        "moved_by_kind": {name: int(value) for name, value in zip(livestock, moved_out.sum(axis=0))},
        **source,
    }
    return {
        "country": country,
        "method": method,
        "checks": checks,
        "physically_validated": physically_validated,
        "metrics": metrics,
    }


def comparison_rows(results):
    rows = []
    for country, methods in results.items():
        ppo, milp = methods["ppo"], methods["milp"]
        metrics = sorted(set(ppo["metrics"]) & set(milp["metrics"]))
        for metric in metrics:
            left, right = ppo["metrics"][metric], milp["metrics"][metric]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)) and left is not None and right is not None:
                rows.append({
                    "country": country,
                    "metric": metric,
                    "ppo": left,
                    "milp": right,
                    "ppo_minus_milp": left - right,
                    "ppo_over_milp": left / right if right else None,
                    "percent_change_ppo_vs_milp": 100 * (left - right) / right if right else None,
                })
    return rows


def main():
    results = {}
    weighted_root = ROOT / "results" / "weighted_4_2_1"
    for country, case in CASES.items():
        data = load_datas(country, case["dataset"])
        summary = json.loads((weighted_root / f"{country}.json").read_text(encoding="utf-8"))
        environment_upper_bound = float(summary["normalization"]["environment_upper_bound"])
        milp_root = weighted_root / country
        ppo_root = case["ppo_root"]
        methods = {
            "ppo": (
                pd.read_excel(ppo_root / "move_out_result.xlsx"),
                pd.read_excel(ppo_root / "move_in_result3.xlsx"),
                pd.read_excel(ppo_root / "PPO_concated3.xlsx"),
            ),
            "milp": (
                pd.read_excel(milp_root / "MILPresult_move_out.xlsx"),
                pd.read_excel(milp_root / "MILPresult_move_in.xlsx"),
                pd.read_excel(milp_root / "MILPresult_routes.xlsx"),
                pd.read_excel(milp_root / "MILPresult_source_structure.xlsx"),
            ),
        }
        results[country] = {
            "ppo": evaluate(country, "ppo", *methods["ppo"], data, environment_upper_bound),
            "milp": evaluate(country, "milp", *methods["milp"][:3], data, environment_upper_bound, methods["milp"][3]),
        }
        results[country]["ppo"]["provenance"] = {
            "version": case["ppo_version"],
            "root": str(ppo_root.relative_to(REPO)),
            "evidence_tier": case["evidence_tier"],
            "native_reward": "4*destination_ammonia_reward + 2*inverse_sensitivity + inverse_relative_pm25 per accepted action",
        }
        results[country]["milp"]["provenance"] = {
            "root": str(milp_root.relative_to(REPO)),
            "solver": summary["solve"],
            "timing": {
                key: summary[key] for key in (
                    "input_load_elapsed_seconds", "build_elapsed_seconds",
                    "serialization_elapsed_seconds", "total_elapsed_seconds", "peak_rss_mb",
                )
            },
        }

    rows = comparison_rows(results)
    report = {
        "scope": {
            "status": "matched_input_rescoring_with_version_qualified_australia",
            "countries": list(CASES),
            "confirmatory_same_version": ["eu", "usa"],
            "version_qualified_robustness": ["aus"],
            "ppo_native_objective": "action-local 4:2:1 reward over destination ammonia, sensitivity, and PM2.5",
            "milp_native_objective": "global 4*N_hat - 2*source_composition_proxy + E_hat",
            "warning": "Native objectives differ; the common 4:2:1 score is a post-hoc MILP-objective rescore for PPO, not PPO training reward.",
        },
        "results": results,
        "comparison": rows,
    }
    analysis = ROOT / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "paper_ppo_vs_weighted_milp.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(analysis / "paper_ppo_vs_weighted_milp.csv", index=False)
    summary_rows = []
    for country, methods in results.items():
        for method, result in methods.items():
            m = result["metrics"]
            summary_rows.append({
                "country": country,
                "method": method,
                "physically_validated": result["physically_validated"],
                "source_n_resolved_ratio": m["source_n_resolved_ratio"],
                "source_n_90pct_target_rate": m["source_n_90pct_target_rate"],
                "environment_score_normalized": m["environment_score_normalized"],
                "source_structure_loss_normalized": m["source_structure_loss_normalized"],
                "source_share_l1_change_mean": m["source_share_l1_change_mean"],
                "weighted_objective_4_2_1_posthoc": m["weighted_objective_4_2_1_posthoc"],
                "total_moved": m["total_moved"],
                "n_resolved_per_moved_animal_kg": m["n_resolved_per_moved_animal_kg"],
                "nonzero_routes": m["nonzero_routes"],
                "source_n_scale_tolerance_violations": m["source_n_scale_tolerance_violations"],
                "destination_n_scale_tolerance_violations": m["destination_n_scale_tolerance_violations"],
                "destination_ammonia_scale_tolerance_violations": m["destination_ammonia_scale_tolerance_violations"],
            })
    pd.DataFrame(summary_rows).to_csv(analysis / "paper_ppo_vs_weighted_milp_summary.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(not result["physically_validated"] for methods in results.values() for result in methods.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
