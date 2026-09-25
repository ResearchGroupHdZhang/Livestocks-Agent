"""Run with PYTHONPATH=.:livestock_rl .venv/bin/python -m pytest <this file>."""
import numpy as np
import torch
from gymnasium import spaces

from livestock_rl.AttentionPolicy import CustomAttentionPolicy


def test_rollout_update_log_probs_are_batch_and_mode_invariant():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    obs_space = spaces.Dict({
        f'{name}_Move_{side}': spaces.Box(-1e9, 1e9, shape=(n, width), dtype=np.float32)
        for side, n in [('in', 3), ('out', 2)]
        for name, width in [('Amount', 2), ('N_demand', 1), ('Ammonia', 1),
                            ('sensitivity', 1), ('relative_pm25', 1)]
    })
    policy = CustomAttentionPolicy(obs_space, spaces.Discrete(6), lambda _: 2e-5)
    observations = {key: torch.randn((4,) + space.shape) for key, space in obs_space.spaces.items()}
    mask = torch.tensor([[False, True, False, True, False, False]] * 4)
    policy.set_training_mode(False)
    with torch.no_grad():
        singles = [policy({key: value[i:i+1] for key, value in observations.items()}, mask[i:i+1])
                   for i in range(4)]
        actions = torch.cat([s[0] for s in singles])
        old_log_probs = torch.cat([s[2] for s in singles])
        policy.set_training_mode(True)
        _, new_log_probs, _ = policy.evaluate_actions(observations, actions, mask)
    ratio = (new_log_probs - old_log_probs).exp()
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6), ratio
