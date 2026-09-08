import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from dataLoader import load_datas


FILES = {"cn": "中国国家尺度更新PB第一步.xlsx"}
WEIGHTS = {"source_n": 4.0, "source_composition": 2.0, "environment": 1.0}
SOURCE_N_GUARD_KG = 1.0


def construct_one_route_seed(data):
    (
        _ids_in,
        _ids_out,
        amount_out,
        amount_in,
        n_in,
        n_coef_in,
        ammonia_in,
        ammonia_coef_in,
        sensitivity_in,
        pm25_in,
        n_out,
        n_coef_out,
        *_rest,
    ) = data
    a_out = amount_out.to_numpy(dtype=np.int64)
    a_in = amount_in.to_numpy(dtype=np.int64)
    n_in_capacity = -n_in.to_numpy(dtype=float).ravel()
    nc_in = n_coef_in.to_numpy(dtype=float)
    ammonia_capacity = ammonia_in.to_numpy(dtype=float).ravel()
    ac_in = ammonia_coef_in.to_numpy(dtype=float)
    n_out_v = n_out.to_numpy(dtype=float).ravel()
    nc_out = n_coef_out.to_numpy(dtype=float)
    environment = (
        2.0 * (1.0 - sensitivity_in.to_numpy(dtype=float).ravel())
        + (1.0 - pm25_in.to_numpy(dtype=float).ravel())
    )
    total_source_n = float(n_out_v.sum())
    best_environment_by_kind = np.array([
        environment[a_in[:, k] > 0].max() if np.any(a_in[:, k] > 0) else 0.0
        for k in range(a_out.shape[1])
    ])
    environment_upper_bound = float((a_out * best_environment_by_kind).sum())
    source_kind_counts = (a_out > 0).sum(axis=1)

    best = None
    for i, k in np.argwhere(a_out > 0):
        source_n_cap = max(0.0, n_out_v[i] - SOURCE_N_GUARD_KG)
        if nc_out[i, k] <= 0 or source_n_cap < nc_out[i, k]:
            continue
        destinations = np.logical_and.reduce(
            (a_in[:, k] > 0, nc_in[:, k] > 0, ac_in[:, k] > 0)
        )
        for j in np.flatnonzero(destinations):
            amount = min(
                int(a_out[i, k]),
                int(np.floor(source_n_cap / nc_out[i, k] + 1e-9)),
                int(np.floor(n_in_capacity[j] / nc_in[j, k] + 1e-9)),
                int(np.floor(ammonia_capacity[j] / ac_in[j, k] + 1e-9)),
            )
            if amount <= 0:
                continue
            removal_rate = amount / a_out[i, k]
            removal_rates = np.zeros(source_kind_counts[i])
            removal_rates[np.flatnonzero(np.flatnonzero(a_out[i] > 0) == k)[0]] = (
                amount / a_out[i, k]
            )
            reference_rate = float(np.median(removal_rates))
            structure_loss = float(
                np.abs(removal_rates - reference_rate).mean() / len(a_out)
            )
            n_normalized = amount * nc_out[i, k] / total_source_n
            environment_normalized = amount * environment[j] / environment_upper_bound
            objective = (
                WEIGHTS["source_n"] * n_normalized
                - WEIGHTS["source_composition"] * structure_loss
                + WEIGHTS["environment"] * environment_normalized
            )
            candidate = {
                "source_index": int(i),
                "destination_index": int(j),
                "species_index": int(k),
                "amount": amount,
                "source_removal_rate": removal_rate,
                "source_reference_rate": reference_rate,
                "source_kind_count": int(source_kind_counts[i]),
                "source_n_resolved_kg": amount * nc_out[i, k],
                "source_n_normalized": n_normalized,
                "source_structure_loss_normalized": structure_loss,
                "environment_score": amount * environment[j],
                "environment_normalized": environment_normalized,
                "weighted_objective": objective,
                "source_n_residual_kg": n_out_v[i] - amount * nc_out[i, k],
                "destination_n_residual_kg": n_in_capacity[j] - amount * nc_in[j, k],
                "destination_ammonia_residual_kg": ammonia_capacity[j] - amount * ac_in[j, k],
            }
            if best is None or candidate["weighted_objective"] > best["weighted_objective"]:
                best = candidate
    if best is None or best["weighted_objective"] <= 0:
        raise RuntimeError("failed to construct a positive-objective one-route feasible seed")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("country", choices=FILES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = perf_counter()
    data = load_datas(args.country, FILES[args.country])
    seed = construct_one_route_seed(data)
    seed["country"] = args.country
    seed["dataset"] = FILES[args.country]
    seed["construction"] = "deterministic_best_positive_one_route"
    seed["build_elapsed_seconds"] = perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(seed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(seed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
