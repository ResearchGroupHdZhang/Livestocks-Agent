from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))


def fake_agent():
    return SimpleNamespace(
        action_mask=torch.tensor([[False, True], [False, False]]),
        num_move_in_counties=2,
        amount_adapt=lambda out_idx, in_idx: torch.tensor([out_idx + in_idx + 1]),
        detect_violation=lambda out_idx, in_idx, amount: [False],
    )


def test_random_is_uniform_over_allowed_action_ids():
    from pilot import RandomSelector
    selector = RandomSelector(np.random.default_rng(7))
    draws = [selector.choose(fake_agent(), None)[0] for _ in range(6000)]
    counts = np.bincount(draws, minlength=4)
    assert counts[1] == 0
    assert max(counts[[0, 2, 3]]) - min(counts[[0, 2, 3]]) < 180


def test_greedy_uses_exact_immediate_reward_and_smallest_action_tie():
    from pilot import GreedySelector
    agent = fake_agent()
    env = SimpleNamespace(score=lambda action, amount: {0: 5, 2: 9, 3: 9}[action])
    assert GreedySelector().choose(agent, env)[0] == 2


def test_greedy_skips_empty_and_physically_invalid_candidates():
    from pilot import GreedySelector
    agent = fake_agent()
    agent.amount_adapt = lambda out_idx, in_idx: torch.tensor({(0, 0): 0, (1, 0): 2, (1, 1): 1}[out_idx, in_idx])
    agent.detect_violation = lambda out_idx, in_idx, amount: [out_idx == 1 and in_idx == 0]
    env = SimpleNamespace(score=lambda action, amount: {0: 100, 2: 90, 3: 1}[action])
    assert GreedySelector().choose(agent, env)[0] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='mixed-device regression needs CUDA')
def test_immediate_reward_accepts_cpu_amount_with_cuda_environment():
    from pilot import immediate_reward
    env = SimpleNamespace(
        Move_in_tensor_Coef_Ammonia=torch.tensor([[2.]], device='cuda'),
        thresholds=[0], Ammonia_move_in_origin=np.array([[10.]]),
        reward_priority=[4, 2, 1],
        reward_func=lambda x, x0, sigma: x,
        evaluate_sensitivity_and_pm25relative=lambda index: (1, 1),
    )
    assert immediate_reward(env, 0, torch.tensor([3])) == 27


def test_priority_ranks_vectorized_features_then_allocates_only_chosen_route():
    from pilot import PrioritySelector
    calls = []
    agent = fake_agent()
    agent.amount_adapt = lambda out_idx, in_idx: calls.append((out_idx, in_idx)) or torch.tensor([1])
    features = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [2, 0, 0, 0, 0]])
    assert PrioritySelector(np.array([1, 0, 0, 0, 0]), features).choose(agent, None)[0] == 2
    assert calls == [(1, 0)]


def physical_agent():
    frame = lambda rows: pd.DataFrame(rows, dtype=float)
    return SimpleNamespace(
        action_len=2, thresholds=[0, 0],
        ID_move_out=pd.DataFrame({'ID': ['o0']}), ID_move_in=pd.DataFrame({'ID': ['i0']}),
        Move_out_origin=frame([[10, 10]]), Move_in_origin=frame([[2, 0]]),
        N_demand_move_out=frame([[20]]), N_demand_move_in=frame([[-20]]),
        Ammonia_move_out=frame([[20]]), Ammonia_move_in=frame([[20]]),
        N_demand_Coef_move_out=frame([[1, 2]]), N_demand_Coef_move_in=frame([[1, 2]]),
        Ammonia_Coef_move_out=frame([[2, 1]]), Ammonia_Coef_move_in=frame([[2, 1]]),
        sensitivity_move_in=frame([[0]]), relative_pm25_move_in=frame([[0]]),
    )


def test_independent_evaluator_replays_physics_and_rejects_support(tmp_path):
    from pilot import evaluate_transfers
    good = tmp_path / 'good.jsonl'
    good.write_text(json.dumps({'out_id': 'o0', 'in_id': 'i0', 'amounts': [3, 0]}) + '\n')
    report = evaluate_transfers(physical_agent(), good)
    assert report['passed']
    assert report['metrics']['moved_heads'] == 3
    assert report['metrics']['n_removed_kg'] == 3
    assert report['metrics']['nh3_net_change_kg'] == 0
    assert report['metrics']['moved_to_low_sensitivity_heads'] == 3
    bad = tmp_path / 'bad.jsonl'
    bad.write_text(json.dumps({'out_id': 'o0', 'in_id': 'i0', 'amounts': [0, 1]}) + '\n')
    assert not evaluate_transfers(physical_agent(), bad)['passed']


def test_priority_search_uses_locked_budget_and_population():
    from pilot import search_weights
    seen = []
    objective = lambda w: seen.append(w.copy()) or -float(np.square(w - 1).sum())
    for method in ('sa-priority', 'ga-priority'):
        seen.clear()
        weights, diagnostics = search_weights(method, np.random.default_rng(2), objective)
        assert len(seen) == 24
        assert weights.shape == (5,)
        assert diagnostics['candidate_trajectories'] == 24
        assert diagnostics['population'] == 6
        assert diagnostics['candidate_max_steps'] == 64
        assert diagnostics['search_seconds'] >= 0


def test_legacy_extractor_forces_checkpoint_transformer_axes():
    from pilot import LegacyAttentionExtractor
    from gymnasium import spaces
    shape = lambda n, m: spaces.Box(0, 1, (n, m), dtype=np.float64)
    observation = spaces.Dict({
        **{f'{field}_Move_{side}': shape(2, 2 if field == 'Amount' else 1)
           for side in ('in', 'out') for field in ('Amount', 'N_demand', 'Ammonia')},
        **{f'{field}_Move_{side}': shape(2, 1)
           for side in ('in', 'out') for field in ('sensitivity', 'relative_pm25')},
    })
    extractor = LegacyAttentionExtractor(observation)
    for layer in (extractor.transformer_layer_in, extractor.transformer_layer_out):
        assert not layer.self_attn.batch_first
        assert layer.self_attn.dropout == .1


def test_rollout_writes_required_artifacts(tmp_path):
    from pilot import RandomSelector, run_rollout
    agent = fake_agent()
    agent.action_mask = torch.tensor([[False]])
    agent.num_move_in_counties = 1
    agent.ID_move_out = pd.DataFrame({'ID': ['o']})
    agent.ID_move_in = pd.DataFrame({'ID': ['i']})
    agent.detect_violation = lambda o, i, a: [False]
    agent.update_Move_df = lambda i, o, a: None
    agent.update_action_mask = lambda o=None, i=None, violation=None: agent.action_mask.fill_(True) if o is not None else None
    env = SimpleNamespace(
        action_mask_left=1, move_amount=None,
        reset=lambda seed=None: ({}, {}),
        step=lambda action: ({}, 2.5, True, False, {}),
    )
    agent.Move_out_origin = pd.DataFrame([[1]])
    agent.Move_in_origin = pd.DataFrame([[1]])
    agent.Move_out_tensor_N_demand_origin = agent.Move_in_tensor_N_demand_origin = torch.tensor([0.])
    agent.Move_out_tensor_Ammonia_origin = agent.Move_in_tensor_Ammonia_origin = torch.tensor([0.])
    agent.reset_action_mask = lambda: torch.tensor([[False]])
    validation = {'passed': True, 'metrics': {'moved_heads': 1}}
    result = run_rollout(agent, env, RandomSelector(np.random.default_rng(1)), tmp_path, 1,
                         {'method': 'random'}, evaluator=lambda *_: validation)
    assert result['status'] == 'mask_exhausted'
    assert result['steps'] == 1 and result['reward'] == 2.5
    assert {p.name for p in tmp_path.iterdir()} == {'config.json', 'transfers.jsonl', 'result.json', 'validation.json'}


def test_rollout_persists_a_valid_time_budget_result(tmp_path):
    from pilot import RandomSelector, run_rollout
    agent = fake_agent()
    agent.action_mask = torch.tensor([[False]])
    agent.num_move_in_counties = 1
    agent.ID_move_out = pd.DataFrame({'ID': ['o']})
    agent.ID_move_in = pd.DataFrame({'ID': ['i']})
    agent.Move_out_origin = agent.Move_in_origin = pd.DataFrame([[1]])
    agent.Move_out_tensor_N_demand_origin = agent.Move_in_tensor_N_demand_origin = torch.tensor([0.])
    agent.Move_out_tensor_Ammonia_origin = agent.Move_in_tensor_Ammonia_origin = torch.tensor([0.])
    agent.reset_action_mask = lambda: torch.tensor([[False]])
    agent.update_action_mask = lambda *args: None
    env = SimpleNamespace(reset=lambda seed=None: ({}, {}))
    validation = {'passed': True, 'metrics': {'moved_heads': 0}}
    result = run_rollout(agent, env, RandomSelector(np.random.default_rng(1)), tmp_path, 16,
                         {'method': 'random'}, evaluator=lambda *_: validation, max_seconds=0)
    assert result['status'] == 'time_budget' and result['steps'] == 0
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'time_budget'


def test_cli_defaults_and_locked_choices():
    from pilot import parse_args
    args = parse_args(['--method', 'frozen-ppo', '--sensitivity', '30', '--output', '/tmp/x'])
    assert args.max_steps == 256 and args.seed == 42
    assert args.method == 'frozen-ppo' and args.sensitivity == 30
    with pytest.raises(SystemExit):
        parse_args(['--method', 'milp', '--sensitivity', '30', '--output', '/tmp/x'])


def test_build_selector_covers_nonsearch_methods():
    from pilot import GreedySelector, PPOSelector, RandomSelector, build_selector
    policy = object()
    assert isinstance(build_selector('random', np.random.default_rng(1), policy)[0], RandomSelector)
    assert isinstance(build_selector('greedy', np.random.default_rng(1), policy)[0], GreedySelector)
    assert isinstance(build_selector('frozen-ppo', np.random.default_rng(1), policy)[0], PPOSelector)


def test_create_real_australia_agent_initializes_locked_problem():
    from pilot import create_agent
    agent, env = create_agent(30, 'cpu')
    try:
        assert agent.country == 'aus'
        assert agent.mobility_ratio == .1 and agent.thresholds == [0, 0]
        assert (agent.num_move_out_counties, agent.num_move_in_counties, agent.action_len) == (513, 426, 6)
        assert agent.action_mask.shape == (513, 426)
        assert 0 < int((~agent.action_mask).sum()) < 513 * 426
        assert env.current_step == 0
    finally:
        env.close()


def test_real_frozen_ppo_loads_legacy_policy_strictly():
    from pilot import LegacyAttentionExtractor, create_agent
    agent, env = create_agent(30, 'cpu', load_policy=True)
    try:
        assert isinstance(agent.policy.features_extractor, LegacyAttentionExtractor)
        assert all(not layer.self_attn.batch_first for layer in (
            agent.policy.features_extractor.transformer_layer_in,
            agent.policy.features_extractor.transformer_layer_out,
        ))
        assert not any(parameter.requires_grad for parameter in agent.policy.parameters())
        assert agent.num_timesteps == 0
    finally:
        env.close()
