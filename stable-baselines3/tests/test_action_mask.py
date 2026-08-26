import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import RolloutBufferSamples


def test_rollout_reuses_action_masks_for_log_probabilities():
    th.manual_seed(0)
    observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(3)
    policy = ActorCriticPolicy(observation_space, action_space, lambda _: 3e-4).to("cpu")
    buffer = RolloutBuffer(2, observation_space, action_space, device="cpu", store_action_masks=True)

    observations = np.zeros((1, 4), dtype=np.float32)
    for action_mask in (th.tensor([False, True, True]), th.tensor([True, False, True])):
        observation_tensor = th.as_tensor(observations)
        with th.no_grad():
            actions, values, log_probs = policy(observation_tensor, action_mask)
        buffer.add(
            observations,
            actions.numpy(),
            np.zeros(1),
            np.zeros(1),
            values,
            log_probs,
            action_mask,
        )

    rollout_data = next(buffer.get())
    _, new_log_probs, _ = policy.evaluate_actions(
        rollout_data.observations,
        rollout_data.actions.long().flatten(),
        rollout_data.action_masks,
    )

    assert rollout_data.action_masks.shape == (2, 3)
    assert th.allclose(new_log_probs, rollout_data.old_log_prob)
    assert th.allclose(th.exp(new_log_probs - rollout_data.old_log_prob), th.ones(2))


def test_unmasked_rollout_buffer_remains_compatible():
    observation_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    buffer = RolloutBuffer(1, observation_space, spaces.Discrete(3), device="cpu")
    buffer.add(
        np.zeros((1, 4), dtype=np.float32),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        th.zeros(1),
        th.zeros(1),
    )

    assert next(buffer.get()).action_masks is None

    legacy_sample = RolloutBufferSamples(*(th.zeros(1) for _ in range(6)))
    assert legacy_sample.action_masks is None


def test_shared_mask_is_broadcast_and_kept_aligned_across_minibatches():
    th.manual_seed(0)
    observation_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(3)
    policy = ActorCriticPolicy(observation_space, action_space, lambda _: 3e-4).to("cpu")
    buffer = RolloutBuffer(
        2, observation_space, action_space, device="cpu", n_envs=2, store_action_masks=True
    )
    transitions = (
        (np.array([[0.0], [0.1]], dtype=np.float32), th.tensor([False, True, True])),
        (
            np.array([[0.2], [0.3]], dtype=np.float32),
            th.tensor([[True, False, True], [True, True, False]]),
        ),
    )

    for observations, action_masks in transitions:
        observation_tensor = th.as_tensor(observations)
        with th.no_grad():
            actions, values, log_probs = policy(observation_tensor, action_masks)
        buffer.add(
            observations,
            actions.numpy(),
            np.zeros(2),
            np.zeros(2),
            values,
            log_probs,
            action_masks,
        )

    batches = list(buffer.get(batch_size=2))
    assert len(batches) == 2
    for rollout_data in batches:
        assert rollout_data.action_masks is not None
        _, new_log_probs, _ = policy.evaluate_actions(
            rollout_data.observations,
            rollout_data.actions.long().flatten(),
            rollout_data.action_masks,
        )
        assert th.allclose(new_log_probs, rollout_data.old_log_prob)

    action_masks = th.cat([batch.action_masks for batch in batches])
    assert sorted(action_masks.tolist()) == sorted(
        [[False, True, True], [False, True, True], [True, False, True], [True, True, False]]
    )


def test_dict_rollout_buffer_round_trips_action_masks():
    observation_space = spaces.Dict(
        {"observation": spaces.Box(-1, 1, shape=(1,), dtype=np.float32)}
    )
    buffer = DictRolloutBuffer(
        1, observation_space, spaces.Discrete(3), device="cpu", store_action_masks=True
    )
    action_mask = th.tensor([False, True, True])
    buffer.add(
        {"observation": np.zeros((1, 1), dtype=np.float32)},
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        th.zeros(1),
        th.zeros(1),
        action_mask,
    )

    assert th.equal(next(buffer.get()).action_masks, action_mask.unsqueeze(0))