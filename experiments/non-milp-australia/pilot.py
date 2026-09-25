"""Locked Australia non-MILP pilot runner."""
from __future__ import annotations

import sys
import argparse
import copy
import hashlib
import json
import importlib.util
import resource
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from livestock_rl.AttentionPolicy import CustomAttentionExtractor

INPUTS = ROOT / "experiments/rl-frozen-generalization/inputs"
CHECKPOINT = ROOT / "logs/v8/aus/best_model.zip"
FROZEN_RUNNER = ROOT / "experiments/rl-frozen-generalization/frozen_eval.py"


class LegacyAttentionExtractor(CustomAttentionExtractor):
    """Checkpoint-era Transformer semantics; current class changed in place."""
    def forward(self, observations):
        for layer in (self.transformer_layer_in, self.transformer_layer_out):
            layer.self_attn.batch_first = False
        return super().forward(observations)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in (self.transformer_layer_in, self.transformer_layer_out):
            layer.self_attn.batch_first = False
            layer.self_attn.dropout = .1
            for module in layer.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = .1


def _frozen_runner():
    """Load the audited frozen-only helper without making it a package."""
    directory = str(FROZEN_RUNNER.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("livestock_frozen_eval", FROZEN_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen runner: {FROZEN_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_agent(sensitivity, device, load_policy=False):
    """Create the locked Australia problem from new input capacities."""
    input_path = INPUTS / f"aus-{sensitivity}.xlsx"
    hash_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    job = {
        "name": f"aus-{sensitivity}-national",
        "country": "aus",
        "sensitivity": sensitivity,
        "scope": "national",
        "province": None,
        "input": str(input_path),
        "input_sha256": hash_file(input_path),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": hash_file(CHECKPOINT),
        "status": "ready",
        "mobility_ratio": .1,
        "thresholds": [0, 0],
    }
    runner = _frozen_runner()
    agent, env = runner.make_agent(job, device)
    # The audited helper is loaded dynamically; keep this module's stable type identity.
    agent.policy.features_extractor.__class__ = LegacyAttentionExtractor
    agent.action_mask = agent.reset_action_mask()
    agent.update_action_mask()
    if not load_policy:
        del agent.policy
    return agent, env


def reset_problem(agent, env, seed):
    observation, _ = env.reset(seed=seed)
    agent.Move_out = agent.Move_out_origin.to_numpy().copy()
    agent.Move_in = agent.Move_in_origin.to_numpy().copy()
    agent.Move_in_tensor_N_demand = agent.Move_in_tensor_N_demand_origin.clone()
    agent.Move_out_tensor_N_demand = agent.Move_out_tensor_N_demand_origin.clone()
    agent.Move_in_tensor_Ammonia = agent.Move_in_tensor_Ammonia_origin.clone()
    agent.Move_out_tensor_Ammonia = agent.Move_out_tensor_Ammonia_origin.clone()
    agent.action_mask = agent.reset_action_mask()
    agent.update_action_mask()
    return observation


def allowed_ids(agent):
    return torch.nonzero(~agent.action_mask.reshape(-1), as_tuple=False).cpu().numpy().ravel()


class RandomSelector:
    def __init__(self, rng):
        self.rng = rng

    def choose(self, agent, env):
        action = int(self.rng.choice(allowed_ids(agent)))
        out_idx, in_idx = divmod(action, agent.num_move_in_counties)
        return action, agent.amount_adapt(out_idx, in_idx)


class GreedySelector:
    def choose(self, agent, env):
        best = None
        for action in allowed_ids(agent):
            action = int(action)
            out_idx, in_idx = divmod(action, agent.num_move_in_counties)
            amount = agent.amount_adapt(out_idx, in_idx)
            if not bool(amount.any()) or any(agent.detect_violation(out_idx, in_idx, amount)):
                continue
            score = float(env.score(action, amount)) if hasattr(env, "score") else immediate_reward(env, in_idx, amount)
            candidate = (-score, action, amount)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("No physically feasible greedy candidate")
        return best[1], best[2]


class PrioritySelector:
    def __init__(self, weights, features=None):
        self.weights = np.asarray(weights, dtype=float)
        self.features = features

    def choose(self, agent, env):
        ids = allowed_ids(agent)
        features = self.features if self.features is not None else route_features(agent)
        scores = features[ids] @ self.weights
        action = int(ids[np.lexsort((ids, -scores))[0]])
        out_idx, in_idx = divmod(action, agent.num_move_in_counties)
        return action, agent.amount_adapt(out_idx, in_idx)


class PPOSelector:
    def __init__(self, policy, deterministic=True):
        self.policy, self.deterministic = policy, deterministic

    def choose(self, agent, env):
        from stable_baselines3.common.utils import obs_as_tensor
        observation = obs_as_tensor({k: v[None] for k, v in env.state.items()}, agent.device)
        with torch.inference_mode():
            action = int(self.policy(observation, agent.action_mask, deterministic=self.deterministic)[0].item())
        oi, ii = divmod(action, agent.num_move_in_counties)
        return action, agent.amount_adapt(oi, ii)


def build_selector(method, rng, policy=None, weights=None):
    if method == "random":
        return RandomSelector(rng), {}
    if method == "greedy":
        return GreedySelector(), {}
    if method == "frozen-ppo":
        if policy is None:
            raise ValueError("Frozen PPO requires a policy")
        return PPOSelector(policy), {"decoding": "deterministic"}
    if method in ("sa-priority", "ga-priority") and weights is not None:
        return PrioritySelector(weights), {"priority_weights": list(map(float, weights))}
    raise ValueError(f"Unsupported or unsearched selector: {method}")


def immediate_reward(env, in_idx, amount):
    coefficients = env.Move_in_tensor_Coef_Ammonia[in_idx]
    delta = (amount.to(device=coefficients.device, dtype=coefficients.dtype) @ coefficients).item()
    ammonia = env.reward_func(torch.tensor(delta), env.thresholds[0], env.Ammonia_move_in_origin[in_idx])
    sensitivity, pm25 = env.evaluate_sensitivity_and_pm25relative(in_idx)
    return float(ammonia * env.reward_priority[0] + sensitivity * env.reward_priority[1] + pm25 * env.reward_priority[2])


def route_features(agent):
    """Five normalized route features, materializing only ~9 MB for Australia."""
    out_n = agent.Move_out_tensor_N_demand.detach().cpu().numpy().ravel()
    in_a = agent.Move_in_tensor_Ammonia.detach().cpu().numpy().ravel()
    sensitivity = np.asarray(agent.sensitivity_move_in).ravel()
    pm25 = np.asarray(agent.relative_pm25_move_in).ravel()
    out = np.asarray(agent.Move_out) > 0
    dest = np.asarray(agent.Move_in_origin) > 0
    compatibility = out.astype(np.uint8) @ dest.T.astype(np.uint8) / agent.action_len
    def norm(x):
        span = np.ptp(x)
        return (x - np.min(x)) / span if span else np.zeros_like(x, dtype=float)
    return np.column_stack((
        np.repeat(norm(out_n), agent.num_move_in_counties),
        np.tile(norm(in_a), agent.num_move_out_counties),
        np.tile(1 - norm(sensitivity), agent.num_move_out_counties),
        np.tile(1 - norm(pm25), agent.num_move_out_counties),
        compatibility.ravel(),
    ))


def evaluate_transfers(agent, path):
    """Replay the ledger from immutable input tables, independent of rollout state."""
    try:
        out0, in0 = np.asarray(agent.Move_out_origin, dtype=np.int64), np.asarray(agent.Move_in_origin, dtype=np.int64)
        moved, added = np.zeros_like(out0), np.zeros_like(in0)
        out_ids = {str(v): i for i, v in enumerate(agent.ID_move_out.ID)}
        in_ids = {str(v): i for i, v in enumerate(agent.ID_move_in.ID)}
        destinations = np.zeros(len(in0), dtype=bool)
        sources = np.zeros(len(out0), dtype=bool)
        steps = 0
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                row = __import__("json").loads(line)
                oi, ii = out_ids[row["out_id"]], in_ids[row["in_id"]]
                raw = row["amounts"]
                if len(raw) != agent.action_len or any(type(x) is not int or x < 0 for x in raw) or not any(raw):
                    raise ValueError("nonnegative nonempty integer quantities required")
                amount = np.asarray(raw, dtype=np.int64)
                if (amount > out0[oi] - moved[oi]).any() or ((in0[ii] == 0) & (amount != 0)).any():
                    raise ValueError("inventory or destination support violation")
                moved[oi] += amount
                added[ii] += amount
                sources[oi] = destinations[ii] = True
                steps += 1
        np.testing.assert_array_equal(moved.sum(0), added.sum(0))
        nout = np.asarray(agent.N_demand_move_out).ravel() - np.sum(moved * np.asarray(agent.N_demand_Coef_move_out), axis=1)
        nin = np.asarray(agent.N_demand_move_in).ravel() + np.sum(added * np.asarray(agent.N_demand_Coef_move_in), axis=1)
        ain = np.asarray(agent.Ammonia_move_in).ravel() - np.sum(added * np.asarray(agent.Ammonia_Coef_move_in), axis=1)
        shortfalls = {
            "source_n_shortfall": float(np.maximum(agent.thresholds[0] - nout[sources], 0).max(initial=0)),
            "destination_n_excess": float(np.maximum(nin[destinations] - agent.thresholds[0], 0).max(initial=0)),
            "destination_nh3_shortfall": float(np.maximum(agent.thresholds[1] - ain[destinations], 0).max(initial=0)),
        }
        if shortfalls["source_n_shortfall"] > 1.000001 or max(shortfalls["destination_n_excess"], shortfalls["destination_nh3_shortfall"]) > .010001:
            raise ValueError(f"capacity violation: {shortfalls}")
        n_removed = float(np.sum(moved * np.asarray(agent.N_demand_Coef_move_out)))
        nh3_removed = float(np.sum(moved * np.asarray(agent.Ammonia_Coef_move_out)))
        nh3_added = float(np.sum(added * np.asarray(agent.Ammonia_Coef_move_in)))
        low_s = np.asarray(agent.sensitivity_move_in).ravel() == 0
        low_p = np.asarray(agent.relative_pm25_move_in).ravel() == 0
        total = int(moved.sum())
        initial_share = out0 / np.maximum(out0.sum(1, keepdims=True), 1)
        final = out0 - moved
        final_share = final / np.maximum(final.sum(1, keepdims=True), 1)
        metrics = {
            "steps": steps, "feasible_rate": 1.0, "conservation_error_heads": int(np.abs(moved.sum(0) - added.sum(0)).max(initial=0)),
            "moved_heads": total, "moved_by_species": moved.sum(0).tolist(), "affected_sources": int(sources.sum()), "affected_destinations": int(destinations.sum()),
            "n_removed_kg": n_removed, "n_removed_ratio": n_removed / max(float(np.sum(np.maximum(np.asarray(agent.N_demand_move_out).ravel(), 0))), 1e-12),
            "destination_n_capacity_used_kg": float(np.sum(added * np.asarray(agent.N_demand_Coef_move_in))),
            "destination_nh3_capacity_used_kg": nh3_added, "nh3_net_change_kg": nh3_added - nh3_removed,
            "moved_to_low_sensitivity_heads": int(added[low_s].sum()), "moved_to_low_sensitivity_ratio": float(added[low_s].sum() / total) if total else 0.0,
            "moved_to_low_pm25_heads": int(added[low_p].sum()), "moved_to_low_pm25_ratio": float(added[low_p].sum() / total) if total else 0.0,
            "source_remaining_share_l1": float(np.sum(np.abs(final_share[sources] - initial_share[sources]), axis=1).mean()) if sources.any() else 0.0,
        }
        return {"passed": True, "metrics": metrics, "capacity_shortfalls": shortfalls,
                "tolerances": {"destination_n": .01, "destination_nh3": .01, "source_n": 1.0}}
    except (ValueError, KeyError, TypeError, AssertionError, OSError) as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def search_weights(method, rng, objective):
    """Locked 24-evaluation pilot search over five ranking weights."""
    started = time.perf_counter()
    population = rng.normal(size=(6, 5))
    scored = [(float(objective(w)), w.copy()) for w in population]
    if method == "sa-priority":
        current_score, current = max(scored, key=lambda x: x[0])
        best_score, best = current_score, current.copy()
        for i in range(18):
            temperature = 1 - i / 18
            candidate = current + rng.normal(scale=max(temperature, .05), size=5)
            score = float(objective(candidate))
            if score >= current_score or rng.random() < np.exp((score - current_score) / max(temperature, .05)):
                current_score, current = score, candidate
            if score > best_score:
                best_score, best = score, candidate.copy()
    elif method == "ga-priority":
        for _ in range(3):
            scored.sort(key=lambda x: x[0], reverse=True)
            elite = [w for _, w in scored[:3]]
            population = np.array([(elite[i % 3] + elite[(i + 1) % 3]) / 2 + rng.normal(scale=.2, size=5) for i in range(6)])
            scored = [(float(objective(w)), w.copy()) for w in population]
        best_score, best = max(scored, key=lambda x: x[0])
    else:
        raise ValueError(method)
    return best, {"candidate_trajectories": 24, "population": 6, "candidate_max_steps": 64,
                  "best_candidate_reward": best_score,
                  "search_seconds": time.perf_counter() - started}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_rollout(agent, env, selector, output, max_steps, config, evaluator=evaluate_transfers,
                max_seconds=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("config.json", "transfers.jsonl", "result.json", "validation.json")):
        raise FileExistsError("refusing to overwrite pilot artifacts")
    write_json(output / "config.json", config)
    reset_problem(agent, env, config.get("seed", 42))
    started = time.perf_counter()
    selector_seconds = allocation_seconds = reward = 0.0
    rejected = steps = 0
    first_feasible = None
    ledger_path = output / "transfers.jsonl"
    with ledger_path.open("x", encoding="utf-8") as ledger:
        while (not bool(agent.action_mask.all()) and steps < max_steps
               and (max_seconds is None or time.perf_counter() - started < max_seconds)):
            tick = time.perf_counter()
            action, amount = selector.choose(agent, env)
            selector_seconds += time.perf_counter() - tick
            oi, ii = divmod(action, agent.num_move_in_counties)
            tick = time.perf_counter()
            violation = any(agent.detect_violation(oi, ii, amount))
            allocation_seconds += time.perf_counter() - tick
            if violation:
                agent.update_action_mask(oi, ii, True)
                rejected += 1
                continue
            agent.update_Move_df(ii, oi, amount)
            agent.update_action_mask(oi, ii, False)
            env.action_mask_left = int((~agent.action_mask).sum())
            env.move_amount = amount
            _, step_reward, _, truncated, _ = env.step(action)
            if truncated:
                raise ValueError("unexpected environment truncation")
            steps += 1
            reward += float(step_reward)
            first_feasible = first_feasible or time.perf_counter() - started
            ledger.write(json.dumps({"step": steps, "action": action,
                         "out_id": str(agent.ID_move_out.ID.iloc[oi]), "in_id": str(agent.ID_move_in.ID.iloc[ii]),
                         "amounts": [int(x) for x in amount.cpu().tolist()]}, allow_nan=False) + "\n")
    validation = evaluator(agent, ledger_path)
    write_json(output / "validation.json", validation)
    status = ("mask_exhausted" if bool(agent.action_mask.all()) else
              "time_budget" if max_seconds is not None and time.perf_counter() - started >= max_seconds else
              "step_limit")
    elapsed = time.perf_counter() - started
    result = {"status": status if validation["passed"] else "failed_validation", "steps": steps,
              "rejected_routes": rejected, "reward": reward, "allowed_routes": int((~agent.action_mask).sum()),
              "first_feasible_seconds": first_feasible, "selector_seconds": selector_seconds,
              "local_allocation_seconds": allocation_seconds, "total_seconds": elapsed,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "validation": validation}
    write_json(output / "result.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=("random", "greedy", "sa-priority", "ga-priority", "frozen-ppo"))
    parser.add_argument("--sensitivity", required=True, type=int, choices=(30, 50))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    load_policy = args.method == "frozen-ppo"
    agent, env = create_agent(args.sensitivity, args.device, load_policy=load_policy)
    try:
        if args.method in ("sa-priority", "ga-priority"):
            def objective(weights):
                candidate_agent, candidate_env = copy.deepcopy(agent), copy.deepcopy(env)
                with __import__("tempfile").TemporaryDirectory() as directory:
                    result = run_rollout(candidate_agent, candidate_env, PrioritySelector(weights),
                                         directory, 64, {"method": args.method, "seed": args.seed},
                                         evaluator=lambda *_: {"passed": True})
                candidate_env.close()
                return result["reward"]
            weights, diagnostics = search_weights(args.method, rng, objective)
            selector, selector_config = build_selector(args.method, rng, weights=weights)
            selector_config.update(diagnostics)
        else:
            selector, selector_config = build_selector(
                args.method, rng, getattr(agent, "policy", None))
        config = {
            "method": args.method,
            "sensitivity": args.sensitivity,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "max_seconds": args.max_seconds,
            "input": str(INPUTS / f"aus-{args.sensitivity}.xlsx"),
            "input_sha256": hashlib.sha256((INPUTS / f"aus-{args.sensitivity}.xlsx").read_bytes()).hexdigest(),
            "checkpoint": str(CHECKPOINT) if load_policy else None,
            "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() if load_policy else None,
            "mobility_ratio": .1,
            "thresholds": [0, 0],
            **selector_config,
        }
        result = run_rollout(agent, env, selector, args.output, args.max_steps, config,
                             max_seconds=args.max_seconds)
        print(json.dumps({k: result[k] for k in ("status", "steps", "total_seconds")}, allow_nan=False))
        return 0 if result["validation"]["passed"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
