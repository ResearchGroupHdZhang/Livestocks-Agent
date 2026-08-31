import json
import os
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pulp


WEIGHTS = {"source_n": 4.0, "source_composition": 2.0, "environment": 1.0}
OBJECTIVE_SCALE = 1_000_000.0


def classify_termination(status, solutions, gap, target_gap):
    if not solutions:
        return False, "no_feasible_solution"
    if status == "optimal":
        return True, "optimal"
    if status == "gaplimit":
        return gap <= target_gap, "gap_reached" if gap <= target_gap else "other_solver_failure"
    if status == "timelimit":
        certified = gap <= target_gap
        return certified, "time_limit_certified" if certified else "time_limit_uncertified"
    return False, "other_solver_failure"


def solve_weighted_4_2_1(
    country,
    dataset,
    data,
    *,
    input_load_elapsed_seconds=0.0,
    time_limit_seconds=7200,
    target_gap=1e-3,
    solver_msg=True,
    preflight_only=False,
):
    run_started = perf_counter()
    build_started = perf_counter()
    (
        ids_in,
        ids_out,
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
        ammonia_out,
        ammonia_coef_out,
        _sensitivity_out,
        _pm25_out,
    ) = data

    a_out = amount_out.to_numpy(dtype=np.int64)
    a_in = amount_in.to_numpy(dtype=np.int64)
    n_in_v = n_in.to_numpy(dtype=float).ravel()
    n_out_v = n_out.to_numpy(dtype=float).ravel()
    nc_in = n_coef_in.to_numpy(dtype=float)
    nc_out = n_coef_out.to_numpy(dtype=float)
    ammonia_in_v = ammonia_in.to_numpy(dtype=float).ravel()
    ammonia_out_v = ammonia_out.to_numpy(dtype=float).ravel()
    ac_in = ammonia_coef_in.to_numpy(dtype=float)
    ac_out = ammonia_coef_out.to_numpy(dtype=float)
    sensitivity_score = 1.0 - sensitivity_in.to_numpy(dtype=float).ravel()
    pm25_score = 1.0 - pm25_in.to_numpy(dtype=float).ravel()
    environment_coefficients = 2.0 * sensitivity_score + pm25_score
    n_out_count, n_in_count, n_kind = len(a_out), len(a_in), a_out.shape[1]
    physical_scale = 1_000.0
    structure_scale = 1_000.0
    source_n_guard_kg = 1.0

    if not (np.isfinite(environment_coefficients).all() and (environment_coefficients >= 0).all()):
        raise ValueError("destination environment coefficients must be finite and nonnegative")

    move_keys = [
        (i, j, k)
        for i in range(n_out_count)
        for j in range(n_in_count)
        for k in range(n_kind)
        if a_out[i, k] > 0 and a_in[j, k] > 0
    ]
    route_keys = sorted({(i, j) for i, j, _k in move_keys})
    moves = {
        key: pulp.LpVariable(
            f"Move_{key[0]}_{key[1]}_{key[2]}",
            0,
            int(a_out[key[0], key[2]]),
            cat="Integer",
        )
        for key in move_keys
    }
    problem = pulp.LpProblem("UnifiedWeighted421LivestockTransfer", pulp.LpMaximize)

    by_source_kind = {(i, k): [] for i in range(n_out_count) for k in range(n_kind)}
    by_source = {i: [] for i in range(n_out_count)}
    by_destination = {j: [] for j in range(n_in_count)}
    for (i, j, k), variable in moves.items():
        by_source_kind[i, k].append(variable)
        by_source[i].append((k, variable))
        by_destination[j].append((i, k, variable))

    for (i, k), variables in by_source_kind.items():
        if variables:
            problem += pulp.lpSum(variables) <= int(a_out[i, k]), f"source_inventory_{i}_{k}"
    for i, variables in by_source.items():
        problem += (
            pulp.lpSum(nc_out[i, k] / physical_scale * variable for k, variable in variables)
            <= max(0.0, n_out_v[i] - source_n_guard_kg) / physical_scale
        ), f"source_n_{i}"
    for j, variables in by_destination.items():
        problem += (
            pulp.lpSum(nc_in[j, k] / physical_scale * variable for _i, k, variable in variables)
            <= -n_in_v[j] / physical_scale
        ), f"destination_n_{j}"
        problem += (
            pulp.lpSum(ac_in[j, k] / physical_scale * variable for _i, k, variable in variables)
            <= ammonia_in_v[j] / physical_scale
        ), f"destination_ammonia_{j}"

    source_fraction_scaled = {
        i: pulp.LpVariable(f"SourceRemovalFractionScaled_{i}", 0, structure_scale)
        for i in range(n_out_count)
    }
    deviations_scaled = []
    source_kind_counts = np.zeros(n_out_count, dtype=np.int64)
    for i in range(n_out_count):
        source_kinds = np.flatnonzero(a_out[i] > 0)
        source_kind_counts[i] = len(source_kinds)
        for k in source_kinds:
            deviation = pulp.LpVariable(
                f"SourceCompositionDeviationScaled_{i}_{k}",
                lowBound=0,
                upBound=structure_scale,
            )
            moved = pulp.lpSum(by_source_kind[i, k])
            scaled_rate = structure_scale / a_out[i, k] * moved
            problem += deviation >= scaled_rate - source_fraction_scaled[i]
            problem += deviation >= source_fraction_scaled[i] - scaled_rate
            deviations_scaled.append(deviation * (1.0 / len(source_kinds)))

    resolved_n_scaled = pulp.lpSum(
        nc_out[i, k] / physical_scale * variable
        for (i, _j, k), variable in moves.items()
    )
    total_source_n_scaled = float(n_out_v.sum() / physical_scale)
    source_n_normalized = resolved_n_scaled * (1.0 / total_source_n_scaled)

    source_structure_loss_scaled = pulp.lpSum(deviations_scaled)
    source_structure_loss_normalized = source_structure_loss_scaled * (
        1.0 / (n_out_count * structure_scale)
    )

    environment_score = pulp.lpSum(
        environment_coefficients[j] * variable
        for (_i, j, _k), variable in moves.items()
    )
    best_environment_by_kind = np.array([
        environment_coefficients[a_in[:, k] > 0].max() if np.any(a_in[:, k] > 0) else 0.0
        for k in range(n_kind)
    ])
    environment_upper_bound = float((a_out * best_environment_by_kind).sum())
    if environment_upper_bound <= 0:
        raise ValueError("fixed environment upper bound must be positive")
    environment_normalized = environment_score * (1.0 / environment_upper_bound)

    normalized_coefficient_abs = np.array([
        abs(
            WEIGHTS["source_n"] * nc_out[i, k] / n_out_v.sum()
            + WEIGHTS["environment"] * environment_coefficients[j] / environment_upper_bound
        )
        for i, j, k in move_keys
    ])
    scaled_coefficient_abs = OBJECTIVE_SCALE * normalized_coefficient_abs

    unified_objective = (
        WEIGHTS["source_n"] * source_n_normalized
        - WEIGHTS["source_composition"] * source_structure_loss_normalized
        + WEIGHTS["environment"] * environment_normalized
    )
    # Scale the whole objective uniformly so small per-animal normalized
    # coefficients are not discarded by solver numerical tolerances.
    problem.setObjective(OBJECTIVE_SCALE * unified_objective)
    build_elapsed_seconds = perf_counter() - build_started
    decision_variables = len(problem.variables())
    constraints = len(problem.constraints)

    preflight = {
        "country": country,
        "dataset": dataset,
        "source_regions": n_out_count,
        "destination_regions": n_in_count,
        "livestock_kinds": n_kind,
        "integer_movement_variables": len(move_keys),
        "continuous_structure_variables": n_out_count + int(source_kind_counts.sum()),
        "decision_variables": decision_variables,
        "constraints": constraints,
        "environment_upper_bound": environment_upper_bound,
        "movement_objective_coefficient_abs_range": {
            "normalized_min": float(normalized_coefficient_abs.min()),
            "normalized_max": float(normalized_coefficient_abs.max()),
            "solver_scaled_min": float(scaled_coefficient_abs.min()),
            "solver_scaled_max": float(scaled_coefficient_abs.max()),
        },
        "build_elapsed_seconds": build_elapsed_seconds,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }

    output_root = Path(os.environ["WEIGHTED_OUTPUT_DIR"])
    output = output_root / country
    output.mkdir(parents=True, exist_ok=True)
    if preflight_only:
        (output_root / f"{country}.preflight.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return preflight
    solver = pulp.SCIP_PY(
        msg=solver_msg,
        logPath=str(output_root / f"{country}_weighted.log"),
        timeLimit=time_limit_seconds,
        gapRel=target_gap,
        options=["numerics/feastol=1e-7"],
    )
    solve_started = perf_counter()
    problem.solve(solver)
    solve_elapsed_seconds = perf_counter() - solve_started
    model = problem.solverModel
    status = str(model.getStatus())
    gap = float(model.getGap())
    solutions = int(model.getNSols())
    certified, termination_reason = classify_termination(status, solutions, gap, target_gap)
    solve_record = {
        "status": status,
        "termination_reason": termination_reason,
        "certified": certified,
        "target_gap": target_gap,
        "gap": gap,
        "solutions": solutions,
        "primal_bound": float(model.getPrimalbound()),
        "dual_bound": float(model.getDualbound()),
        "nodes": int(model.getNNodes()),
        "solver_seconds": float(model.getSolvingTime()),
        "solve_elapsed_seconds": solve_elapsed_seconds,
    }
    if not solutions:
        failure = {
            "country": country,
            "dataset": dataset,
            "model": "integer_unified_weighted_4_2_1",
            "solve": solve_record,
            "build_elapsed_seconds": build_elapsed_seconds,
            "input_load_elapsed_seconds": input_load_elapsed_seconds,
            "decision_variables": decision_variables,
            "constraints": constraints,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "total_elapsed_seconds": input_load_elapsed_seconds + perf_counter() - run_started,
        }
        (output_root / f"{country}.failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"weighted solve produced no feasible incumbent: {solve_record}")

    serialization_started = perf_counter()
    result = np.zeros((n_out_count, n_in_count, n_kind), dtype=np.int64)
    for key, variable in moves.items():
        result[key] = int(round(variable.value() or 0))

    moved_out = result.sum(axis=1)
    moved_in = result.sum(axis=0)
    final_out = a_out - moved_out
    final_in = a_in + moved_in
    final_n_out = n_out_v - (moved_out * nc_out).sum(axis=1)
    final_n_in = n_in_v + (moved_in * nc_in).sum(axis=1)
    final_ammonia_out = ammonia_out_v - (moved_out * ac_out).sum(axis=1)
    final_ammonia_in = ammonia_in_v - (moved_in * ac_in).sum(axis=1)

    source_reference_fractions = np.array([
        float(source_fraction_scaled[i].value() or 0) / structure_scale
        for i in range(n_out_count)
    ])
    source_structure_loss = np.zeros(n_out_count)
    source_share_l1_change = np.zeros(n_out_count)
    for i in range(n_out_count):
        present = a_out[i] > 0
        removal_rates = moved_out[i, present] / a_out[i, present]
        source_structure_loss[i] = float(np.abs(removal_rates - source_reference_fractions[i]).mean())
        if final_out[i].sum() > 0:
            source_share_l1_change[i] = float(
                np.abs(a_out[i] / a_out[i].sum() - final_out[i] / final_out[i].sum()).sum()
            )

    resolved_n_kg = float((moved_out * nc_out).sum())
    source_n_normalized_value = resolved_n_kg / n_out_v.sum()
    environment_score_value = float((moved_in * environment_coefficients[:, None]).sum())
    environment_normalized_value = environment_score_value / environment_upper_bound
    structure_normalized_value = float(source_structure_loss.mean())
    weighted_contributions = {
        "source_n": WEIGHTS["source_n"] * source_n_normalized_value,
        "source_composition": -WEIGHTS["source_composition"] * structure_normalized_value,
        "environment": WEIGHTS["environment"] * environment_normalized_value,
    }
    objective_value = float(sum(weighted_contributions.values()))

    id_out_columns = list(ids_out.columns)
    id_in_columns = list(ids_in.columns)
    livestock_columns = list(amount_out.columns)
    route_rows = []
    for i, j in route_keys:
        moved = result[i, j]
        if not moved.sum():
            continue
        route_rows.append({
            **{f"source_{column}": ids_out.iloc[i][column] for column in id_out_columns},
            **{f"destination_{column}": ids_in.iloc[j][column] for column in id_in_columns},
            **{column: int(moved[k]) for k, column in enumerate(livestock_columns)},
            "source_n_resolved_kg": float((moved * nc_out[i]).sum()),
            "destination_n_added_kg": float((moved * nc_in[j]).sum()),
            "destination_ammonia_used_kg": float((moved * ac_in[j]).sum()),
            "environment_score": float(moved.sum() * environment_coefficients[j]),
        })

    out_values = pd.DataFrame(
        np.column_stack((final_out, final_n_out, final_ammonia_out)),
        columns=[*livestock_columns, "N_demand", "ammonia"],
    )
    in_values = pd.DataFrame(
        np.column_stack((final_in, final_n_in, final_ammonia_in)),
        columns=[*livestock_columns, "N_demand", "ammonia"],
    )
    pd.concat([ids_out.reset_index(drop=True), out_values], axis=1).to_excel(
        output / "MILPresult_move_out.xlsx", index=False
    )
    pd.concat([ids_in.reset_index(drop=True), in_values], axis=1).to_excel(
        output / "MILPresult_move_in.xlsx", index=False
    )
    pd.DataFrame(route_rows).to_excel(output / "MILPresult_routes.xlsx", index=False)
    pd.concat([
        ids_out.reset_index(drop=True),
        pd.DataFrame({
            "reference_removal_fraction": source_reference_fractions,
            "mean_abs_removal_rate_deviation": source_structure_loss,
            "livestock_share_l1_change": source_share_l1_change,
        }),
    ], axis=1).to_excel(output / "MILPresult_source_structure.xlsx", index=False)
    serialization_elapsed_seconds = perf_counter() - serialization_started

    summary = {
        "country": country,
        "dataset": dataset,
        "model": "integer_unified_weighted_4_2_1",
        "objectives": [
            "maximize_source_n_resolved",
            "minimize_source_composition_change",
            "maximize_destination_environment",
        ],
        "weights": WEIGHTS,
        "solver_objective_scale": OBJECTIVE_SCALE,
        "normalization": {
            "source_n_denominator_kg": float(n_out_v.sum()),
            "source_composition_denominator": "equal-source mean of equal-species mean absolute removal-rate deviations",
            "environment_upper_bound": environment_upper_bound,
            "environment_best_coefficient_by_kind": best_environment_by_kind.tolist(),
        },
        "movement_objective_coefficient_abs_range": {
            "normalized_min": float(normalized_coefficient_abs.min()),
            "normalized_max": float(normalized_coefficient_abs.max()),
            "solver_scaled_min": float(scaled_coefficient_abs.min()),
            "solver_scaled_max": float(scaled_coefficient_abs.max()),
        },
        "environment_internal_weights": [2, 1],
        "source_n_numeric_guard_kg_per_region": source_n_guard_kg,
        "solve_policy": {
            "target_relative_gap": target_gap,
            "time_limit_seconds": time_limit_seconds,
            "time_limit_role": "hard_safety_cap",
            "concurrent_with_australia_exact_primary": True,
        },
        "solve": solve_record,
        "source_n_total_kg": float(n_out_v.sum()),
        "source_n_resolved_kg": resolved_n_kg,
        "source_n_resolved_ratio": source_n_normalized_value,
        "source_n_remaining_kg": float(final_n_out.sum()),
        "source_structure_loss_normalized": structure_normalized_value,
        "source_structure_loss_max": float(source_structure_loss.max()),
        "source_share_l1_change_mean": float(source_share_l1_change.mean()),
        "source_share_l1_change_max": float(source_share_l1_change.max()),
        "source_reference_fraction_mean": float(source_reference_fractions.mean()),
        "environment_score": environment_score_value,
        "environment_score_normalized": environment_normalized_value,
        "weighted_contributions": weighted_contributions,
        "weighted_objective": objective_value,
        "solver_objective": float(pulp.value(problem.objective) / OBJECTIVE_SCALE),
        "destination_ammonia_used_kg": float((moved_in * ac_in).sum()),
        "source_ammonia_removed_kg": float((moved_out * ac_out).sum()),
        "total_moved": int(moved_out.sum()),
        "moved_by_kind": moved_out.sum(axis=0).tolist(),
        "nonzero_routes": len(route_rows),
        "allowed_route_species": len(move_keys),
        "decision_variables": decision_variables,
        "integer_movement_variables": len(move_keys),
        "continuous_structure_variables": n_out_count + int(source_kind_counts.sum()),
        "constraints": constraints,
        "result_shape": list(result.shape),
        "input_load_elapsed_seconds": input_load_elapsed_seconds,
        "build_elapsed_seconds": build_elapsed_seconds,
        "serialization_elapsed_seconds": serialization_elapsed_seconds,
        "total_elapsed_seconds": input_load_elapsed_seconds + perf_counter() - run_started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (output_root / f"{country}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
