from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from 强化学习 import LivestockEnv, LivestockEnvConfig


def _write_input(path: Path, incoming_n: float = -0.5) -> None:
    row = {
        "ID": "a", "city": "a", "county": "a", "province": "p",
        "num_cattle": 10, "最优氮需求": 2.0, "ammonia指标": 2.0,
        "cattle_manure变化量": 0.1, "cattle_氨变化量": 0.1,
        "敏感度": 0, "PM2.5相关性": 0,
    }
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([row]).to_excel(writer, sheet_name="移出", index=False)
        pd.DataFrame([{**row, "最优氮需求": incoming_n}]).to_excel(writer, sheet_name="移入", index=False)


def test_reward_priority_requires_exactly_three_documented_components():
    with pytest.raises(ValueError, match="exactly three"):
        LivestockEnvConfig("cn", [4, 4, 3, 2, 1], [0, 0], "input.xlsx")


def test_reset_observation_matches_declared_space(tmp_path):
    path = tmp_path / "tiny.xlsx"
    _write_input(path)
    env = LivestockEnv(LivestockEnvConfig("eu", [4, 2, 1], [0, 0], str(path)))
    observation, _ = env.reset()
    assert env.observation_space.contains(observation)


def test_step_rejects_infeasible_transfer_before_mutating_state(tmp_path):
    path = tmp_path / "tiny.xlsx"
    _write_input(path)
    env = LivestockEnv(LivestockEnvConfig("eu", [4, 2, 1], [0, 0], str(path)))
    before, _ = env.reset()
    before = {key: value.copy() for key, value in before.items()}
    env.move_amount = torch.tensor([100], dtype=torch.int64, device=env.device)
    env.action_mask_left = 1
    with pytest.raises(ValueError, match="Infeasible transfer"):
        env.step(0)
    assert all(np.array_equal(before[key], env.state[key]) for key in before)


@pytest.mark.parametrize("result", [
    type("Result", (), {"success": False})(),
    type("Result", (), {"success": True, "x": np.array([100.0])})(),
])
def test_failed_or_invalid_quantity_solve_returns_zero(tmp_path, monkeypatch, result):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "tiny.xlsx"
    _write_input(path, incoming_n=-0.1)
    config = LivestockEnvConfig("eu", [4, 2, 1], [0, 0], str(path), mobility_ratio=0.25)
    env = make_vec_env(lambda: LivestockEnv(config), n_envs=1)
    model = PPO_action_mask_v2(
        "MultiInputPolicy", env, n_steps=2, batch_size=2, n_epochs=1,
        learning_rate=0, device="cpu", seed=0,
    )
    monkeypatch.setattr("stable_baselines3.common.on_policy_algorithm_v2.linprog", lambda *args, **kwargs: result)
    assert not model.amount_adapt(0, 0).any()
    env.close()


@pytest.mark.parametrize("script", ["train_ppo_v2.py", "train_ppo_v2_province.py"])
def test_training_script_uses_an_independent_evaluation_env(script):
    source = (Path(__file__).parents[2] / "强化学习" / script).read_text(encoding="utf-8")
    assert "eval_env = make_vec_env" in source
    assert "EvalCallback(eval_env," in source


def test_evaluation_config_supplies_the_required_input_path():
    source = (Path(__file__).parents[2] / "强化学习" / "evaluation_ppo.py").read_text(encoding="utf-8")
    assert "df_path=" in source


def test_evaluation_entrypoint_is_import_safe():
    import 强化学习.evaluation_ppo  # noqa: F401