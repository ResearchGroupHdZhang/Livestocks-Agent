import numpy as np
import pandas as pd
import pytest
import torch

from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation_action_mask_v2 import evaluate_policy
from 强化学习 import LivestockEnv, LivestockEnvConfig


def test_all_masked_distribution_is_not_uniform():
    with pytest.raises(ValueError, match='feasible'):
        CategoricalDistribution(2).proba_distribution(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))


def test_exhausted_rollout_keeps_draw_mask_and_never_moves(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / 'tiny.xlsx'
    row = {'ID': 'a', 'city': 'a', 'county': 'a', 'province': 'p', 'num_cattle': 100,
           '最优氮需求': 0.5, 'ammonia指标': 0.5, 'cattle_manure变化量': 0.1,
           'cattle_氨变化量': 0.1, '敏感度': 0, 'PM2.5相关性': 0}
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([row]).to_excel(writer, sheet_name='移出', index=False)
        pd.DataFrame([{**row, '最优氮需求': -0.5}]).to_excel(writer, sheet_name='移入', index=False)
    config = LivestockEnvConfig('eu', [4, 2, 1], [0, 0], str(path), max_steps=4)
    env = make_vec_env(lambda: LivestockEnv(config), n_envs=1)
    model = PPO_action_mask_v2('MultiInputPolicy', env, n_steps=2, batch_size=2, n_epochs=1,
                               learning_rate=0, device='cpu', seed=0)
    # Force the exact retry-exhaustion boundary, independent of solver tolerances.
    model.amount_adapt = lambda *args: torch.zeros(1, dtype=torch.int64, device=model.device)
    model.learn(2)
    assert (model.rollout_buffer.rewards == 0).all()
    assert not model.rollout_buffer.action_masks.any()
    assert np.allclose(model.rollout_buffer.log_probs, 0)
    assert all(ep['r'] == 0 for ep in model.ep_info_buffer)
    raw = env.envs[0].unwrapped
    assert np.array_equal(raw.Move_out_origin, raw.Move_out)
    raw.move_amount = torch.zeros(1, dtype=torch.int64, device=raw.device)
    raw.action_mask_left = 0
    _, reward, done, _, _ = raw.step(0)
    assert reward == 0 and done
    env.close()
