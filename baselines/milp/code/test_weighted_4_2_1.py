import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).parent))
from weighted_4_2_1 import classify_termination, solve_weighted_4_2_1


def frame(values, columns):
    return pd.DataFrame(values, columns=columns)


def toy_data():
    livestock = ["cattle", "pigs"]
    return (
        frame([[2, "d", "d"]], ["ID", "city", "county"]),
        frame([[1, "s", "s"]], ["ID", "city", "county"]),
        frame([[100, 50]], livestock),
        frame([[1, 1]], livestock),
        frame([[-1000]], ["最优氮需求"]),
        frame([[1, 1]], ["manure变化量_cattle", "manure变化量_pigs"]),
        frame([[1000]], ["ammonia指标"]),
        frame([[1, 1]], ["氨变化量_cattle", "氨变化量_pigs"]),
        frame([[0]], ["敏感度"]),
        frame([[0]], ["PM2.5相关性"]),
        frame([[75]], ["最优氮需求"]),
        frame([[1, 1]], ["manure变化量_cattle", "manure变化量_pigs"]),
        frame([[1000]], ["ammonia指标"]),
        frame([[1, 1]], ["氨变化量_cattle", "氨变化量_pigs"]),
        frame([[0]], ["敏感度"]),
        frame([[0]], ["PM2.5相关性"]),
    )


def test_termination_policy():
    assert classify_termination("optimal", 1, 0.0, 1e-3) == (True, "optimal")
    assert classify_termination("gaplimit", 1, 9e-4, 1e-3) == (True, "gap_reached")
    assert classify_termination("timelimit", 1, 0.2, 1e-3) == (False, "time_limit_uncertified")
    assert classify_termination("timelimit", 0, np.inf, 1e-3) == (False, "no_feasible_solution")


def test_unlimited_solver_configuration():
    original_solver = pulp.SCIP_PY
    captured = {}

    def capture_solver(**kwargs):
        captured.update(kwargs)
        return original_solver(**kwargs)

    pulp.SCIP_PY = capture_solver
    try:
        solve_weighted_4_2_1(
            "unlimited",
            "toy",
            toy_data(),
            time_limit_seconds=None,
            target_gap=0,
            solver_msg=False,
        )
    finally:
        pulp.SCIP_PY = original_solver
    assert "timeLimit" not in captured, captured
    assert captured["gapRel"] == 0, captured


def demo(output_root):
    os.environ["WEIGHTED_OUTPUT_DIR"] = str(output_root)
    data = toy_data()
    summary = solve_weighted_4_2_1(
        "toy",
        "toy",
        data,
        time_limit_seconds=60,
        target_gap=0,
        solver_msg=False,
    )
    livestock = ["cattle", "pigs"]
    original = np.array([100, 50])
    initial_in = np.array([1, 1])
    saved_out = pd.read_excel(output_root / "toy" / "MILPresult_move_out.xlsx")
    saved_in = pd.read_excel(output_root / "toy" / "MILPresult_move_in.xlsx")
    structure = pd.read_excel(output_root / "toy" / "MILPresult_source_structure.xlsx")
    removed = original - saved_out[livestock].to_numpy(dtype=np.int64)[0]
    added = saved_in[livestock].to_numpy(dtype=np.int64)[0] - initial_in
    rates = removed / original
    n_normalized = removed.sum() / 75
    structure_normalized = np.abs(
        rates - structure["reference_removal_fraction"].iloc[0]
    ).mean()
    environment_normalized = 3 * added.sum() / 450
    objective = 4 * n_normalized - 2 * structure_normalized + environment_normalized
    assert np.array_equal(removed, added)
    assert np.allclose(rates[0], rates[1], atol=0.02), rates
    assert summary["solve"]["certified"], summary
    assert summary["source_n_resolved_kg"] >= 74, summary
    assert summary["source_share_l1_change_mean"] < 0.02, summary
    assert 0 <= summary["source_n_resolved_ratio"] <= 1
    assert 0 <= summary["source_structure_loss_normalized"] <= 1
    assert 0 <= summary["environment_score_normalized"] <= 1
    assert abs(n_normalized - summary["source_n_resolved_ratio"]) <= 1e-9
    assert abs(structure_normalized - summary["source_structure_loss_normalized"]) <= 1e-9
    assert abs(environment_normalized - summary["environment_score_normalized"]) <= 1e-9
    assert abs(objective - summary["weighted_objective"]) <= 1e-9
    assert abs(summary["weighted_objective"] - summary["solver_objective"]) <= 1e-7
    print(summary)


def test_positive_full_warm_start(output_root):
    os.environ["WEIGHTED_OUTPUT_DIR"] = str(output_root)
    seed = {
        "source_index": 0,
        "destination_index": 0,
        "species_index": 0,
        "amount": 10,
        "weighted_objective": 4 * (10 / 75) - 2 * 0.05 + 30 / 450,
    }
    summary = solve_weighted_4_2_1(
        "warm",
        "toy",
        toy_data(),
        time_limit_seconds=60,
        target_gap=0,
        solver_msg=False,
        warm_start_seed=seed,
    )
    assert summary["warm_start"]["weighted_objective"] > 0, summary
    assert summary["warm_start"]["all_model_variables_initialized"], summary
    log = (output_root / "warm_weighted.log").read_text(encoding="utf-8")
    assert "objective value 5.000000e+05" in log, log
    assert "completion of a partial solution failed" not in log, log


if __name__ == "__main__":
    test_termination_policy()
    os.environ["WEIGHTED_OUTPUT_DIR"] = sys.argv[1]
    test_unlimited_solver_configuration()
    demo(Path(sys.argv[1]))
    test_positive_full_warm_start(Path(sys.argv[1]))
