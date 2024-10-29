from re import T
from tkinter import N
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from livestockEnvV2 import load_datas
from scipy.optimize import linprog
import copy

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    action_mask = None,
    save_path = None
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

   .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :param action_mask: The action mask to be considered during evaluation.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    import pandas as pd
    
    log = []
    # ID_move_in, ID_move_out, Move_in, Move_out, Target_move_in, Coef_move_in, Target_move_out, Coef_move_out = load_datas(model.country)
    Move_in, Move_out = model.Move_in_origin.copy(), model.Move_out_origin.copy()
    Target_move_in, Target_move_out = copy.deepcopy(model.Target_move_in_origin), copy.deepcopy(model.Target_move_out_origin)
    
    Move_in_tensor_N_demand = torch.tensor(Target_move_in['N_demand'], dtype=torch.float64).to(model.device)
    Move_in_tensor_ammonia_density = torch.tensor(Target_move_in['ammonia_density'], dtype=torch.float64).to(model.device)
    Move_in_tensor_livestock_PB = torch.tensor(Target_move_in['livestock_PB'], dtype=torch.float64).to(model.device)

    Move_out_tensor_N_demand = torch.tensor(Target_move_out['N_demand'], dtype=torch.float64).to(model.device)
    Move_out_tensor_ammonia_density = torch.tensor(Target_move_out['ammonia_density'], dtype=torch.float64).to(model.device)
    Move_out_tensor_livestock_PB = torch.tensor(Target_move_out['livestock_PB'], dtype=torch.float64).to(model.device)

    num_move_out_counties, num_move_in_counties = Move_out.shape[0], Move_in.shape[0]

    if action_mask is None:
        action_mask = torch.zeros([model.num_move_out_counties, model.num_move_in_counties], dtype=bool).to(model.device)

    def update_action_mask(move_out_idx=None, move_in_idx=None, violation=None):
        '''
        action mask update function
        '''
        if move_out_idx is None: # action mask initialization
            for out_idx in range(model.num_move_out_counties):
                action_mask[out_idx, :] = True if (Move_out.iloc[out_idx, :] <= 0).all() else False
        else:
            action_mask[move_out_idx, :] = torch.logical_or(action_mask[move_out_idx, :].data,
                                                            detect_violation_move_out(move_out_idx).to(action_mask.device))
            action_mask[:, move_in_idx] = torch.logical_or(action_mask[:, move_in_idx].data,
                                                            detect_violation_move_in(move_in_idx))
            if violation:
                action_mask[move_out_idx, move_in_idx] = True
                            
    def detect_violation_move_out(move_out_idx):

        empty_violation =(Move_out.iloc[move_out_idx, :] == 0).all()
        N_demand_violation = Move_out_tensor_N_demand[move_out_idx] < model.thresholds[0] + model.N_demand_coef_move_out[move_out_idx].min()
        ammonia_density_violation = Move_out_tensor_ammonia_density[move_out_idx] < model.thresholds[1] + model.ammonia_density_coef_move_out[move_out_idx].min()
        livestock_PB_violation = Move_out_tensor_livestock_PB[move_out_idx] < model.thresholds[2] + model.livestock_PB_coef_move_out[move_out_idx].min()

        if empty_violation:
            return torch.ones(model.num_move_in_counties) 
        elif model.ammonia_density_case(model.Target_move_out_origin['ammonia_density'][move_out_idx]):
            return torch.zeros(model.num_move_in_counties)
        elif N_demand_violation and ammonia_density_violation and livestock_PB_violation:
            return torch.ones(model.num_move_in_counties) 
        else: 
            return torch.zeros(model.num_move_in_counties)

    def detect_violation_move_in(move_in_idx):
        N_demand_violation = Move_in_tensor_N_demand[move_in_idx] > model.thresholds[0] - model.N_demand_coef_move_in[move_in_idx].min()
        ammonia_density_violation = Move_in_tensor_ammonia_density[move_in_idx] > model.thresholds[1] - model.ammonia_density_coef_move_in[move_in_idx].min()
        livestock_PB_violation = Move_in_tensor_livestock_PB[move_in_idx] > model.thresholds[2] - model.livestock_PB_coef_move_in[move_in_idx].min()

        return torch.ones(num_move_out_counties).to(action_mask.device) if (N_demand_violation or ammonia_density_violation or livestock_PB_violation) else torch.zeros(num_move_out_counties).to(action_mask.device)

    def detect_violation(move_out_idx, move_in_idx, amounts):
        cur_amounts = Move_out.iloc[move_out_idx, :]
        Amounts_violation = (cur_amounts.values < amounts.cpu().numpy()).any()
        empty_violation = (amounts == 0).all() or (cur_amounts == 0).all()
        N_demand_violation_move_out = Move_out_tensor_N_demand[move_out_idx] < model.thresholds[0] + amounts.double() @ model.N_demand_coef_move_out[move_out_idx]
        ammonia_density_violation_move_out = Move_out_tensor_ammonia_density[move_out_idx] < model.thresholds[1] + amounts.double() @ model.ammonia_density_coef_move_out[move_out_idx]
        livestock_PB_violation_move_out = Move_out_tensor_livestock_PB[move_out_idx] < model.thresholds[2] + amounts.double() @ model.livestock_PB_coef_move_out[move_out_idx]
        
        N_demand_violation_move_in = Move_in_tensor_N_demand[move_in_idx] > model.thresholds[0] - amounts.double() @ model.N_demand_coef_move_in[move_in_idx]
        ammonia_density_violation_move_in = Move_in_tensor_ammonia_density[move_in_idx] > model.thresholds[1] - amounts.double() @ model.ammonia_density_coef_move_in[move_in_idx]
        livestock_PB_violation_move_in = Move_in_tensor_livestock_PB[move_in_idx] > model.thresholds[2] - amounts.double() @ model.livestock_PB_coef_move_in[move_in_idx]

        return empty_violation, Amounts_violation,\
              (N_demand_violation_move_out and ammonia_density_violation_move_out and livestock_PB_violation_move_out),\
              (N_demand_violation_move_in or ammonia_density_violation_move_in or livestock_PB_violation_move_in)

    
    def update_Move_df(move_out_idx, move_in_idx, amounts):
        Move_in.iloc[move_in_idx, :] += amounts.cpu().numpy()
        Move_out.iloc[move_out_idx, :] -= amounts.cpu().numpy()

        Move_in_tensor_N_demand[move_in_idx] += (amounts.double() @ model.N_demand_coef_move_in[move_in_idx]).item()
        Move_out_tensor_N_demand[move_out_idx] -= (amounts.double() @ model.N_demand_coef_move_out[move_out_idx]).item()
        
        Move_in_tensor_ammonia_density[move_in_idx] += (amounts.double() @ model.ammonia_density_coef_move_in[move_in_idx]).item()
        Move_out_tensor_ammonia_density[move_out_idx] -=(amounts.double() @ model.ammonia_density_coef_move_out[move_out_idx]).item()
        
        Move_in_tensor_livestock_PB[move_in_idx] += (amounts.double() @ model.livestock_PB_coef_move_in[move_in_idx]).item()
        Move_out_tensor_livestock_PB[move_out_idx] -= (amounts.double() @ model.livestock_PB_coef_move_out[move_out_idx]).item()
    

    def amount_adapt(move_out_idx, move_in_idx):
        amounts = model.amounts[move_out_idx, :].clone()
        cur_amounts = Move_out.iloc[move_out_idx, :].values

        if not True in detect_violation(move_out_idx, move_in_idx, amounts):
            return amounts

        N_demand_coef_in = model.N_demand_coef_move_in[move_in_idx].cpu().numpy()
        ammonia_density_coef_in = model.ammonia_density_coef_move_in[move_in_idx].cpu().numpy()
        livestock_PB_coef_in = model.livestock_PB_coef_move_in[move_in_idx].cpu().numpy()

        N_demand_coef_out = model.N_demand_coef_move_out[move_out_idx].cpu().numpy()
        ammonia_density_coef_out = model.ammonia_density_coef_move_out[move_out_idx].cpu().numpy()
        livestock_PB_coef_out = model.livestock_PB_coef_move_out[move_out_idx].cpu().numpy()
        
        cur = torch.tensor([model.Move_in_tensor_N_demand[move_in_idx],model.Move_in_tensor_ammonia_density[move_in_idx],model.Move_in_tensor_livestock_PB[move_in_idx]]).cpu().numpy().reshape(-1, 1)

        c = -np.ones(amounts.shape[0])  # Coefficients for the linear objective function

        bounds = [(0, min(amounts[i].item(), cur_amounts[i])) for i in range(len(cur_amounts))]

        cur = torch.tensor([Move_in_tensor_N_demand[move_in_idx],
                         Move_in_tensor_ammonia_density[move_in_idx],
                         Move_in_tensor_livestock_PB[move_in_idx],
                         Move_out_tensor_N_demand[move_out_idx],
                         Move_out_tensor_ammonia_density[move_out_idx],
                         Move_out_tensor_livestock_PB[move_out_idx]]).cpu().numpy().reshape(-1, 1)

        A_ub = np.array([N_demand_coef_in, ammonia_density_coef_in, livestock_PB_coef_in,
                         N_demand_coef_out, ammonia_density_coef_out, livestock_PB_coef_out])
        
        b_ub_in = np.array([model.thresholds]).reshape(-1, 1) - cur[:3]
        b_ub_out = cur[3:] - np.array([model.thresholds]).reshape(-1, 1)
        b_ub = np.concatenate((b_ub_in, b_ub_out), axis=0).reshape(-1, 1)

        idxs = b_ub < 0
        idxs = np.where(idxs)[0]
        if len(idxs) > 0:
            A_ub = np.delete(A_ub, idxs, axis=0)
            b_ub = np.delete(b_ub, idxs, axis=0)

        res = linprog(c, bounds=bounds, A_ub=A_ub, b_ub=b_ub, method='highs', integrality=1)
        if res.success:
            amounts = torch.tensor(res.x, dtype=torch.int64).to(amounts.device)
        return amounts

    episode_starts = np.ones((env.num_envs,), dtype=bool)

    with tqdm(total=100000, desc="进度") as pbar:  # 假设总进度是 100000
        while (episode_counts < episode_count_targets).any():
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
                mask=action_mask  # 传递 action_mask
            )
            move_out_idx, move_in_idx = model.action_proj(actions[0])
            amount = amount_adapt(move_out_idx, move_in_idx)
            violation = detect_violation(move_out_idx, move_in_idx, amount)
            counter = 0

            while True in violation and action_mask.sum() < num_move_in_counties* num_move_out_counties and counter < 500: # if available amounts search round > 500, break
                update_action_mask(move_out_idx, move_in_idx, True)  # 更新 action_mask
                actions, states = model.predict(
                    observations,  # type: ignore[arg-type]
                    state=states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                    mask=action_mask  # 传递 action_mask
                )
                
                move_out_idx, move_in_idx = model.action_proj(actions[0])
                amount = amount_adapt(move_out_idx, move_in_idx)
                violation = detect_violation(move_out_idx, move_in_idx, amount)
                counter += 1

            update_Move_df(move_out_idx, move_in_idx, amount)  # 更新 Move_in 和 Move_out
            update_action_mask(move_out_idx, move_in_idx, False)  # 更新 action_mask
            if counter >= 500:
                env.unwrapped.envs[0].env.env.env.action_mask_left = 0
            else:
                env.unwrapped.envs[0].env.env.env.action_mask_left = model.action_mask_sum_origin - action_mask.sum()
            env.unwrapped.envs[0].env.env.env.move_amount = amount
            
            new_observations, rewards, dones, infos = env.step(actions)
            
            if action_mask.sum() == num_move_in_counties* num_move_out_counties:
                dones = [True] * n_envs
                
            current_rewards += rewards
            current_lengths += 1
            
            pbar.set_postfix({"reward":rewards[0], "mask":action_mask.sum()})
            pbar.update(1)
            if save_path:
                log.append([move_out_idx, move_in_idx, amount, rewards[0], action_mask.sum().item()])
            
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    # unpack values so that the callback can access the local variables
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done

                    if callback is not None:
                        callback(locals(), globals())

                    if dones[i]:
                        if is_monitor_wrapped:
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards.append(info["episode"]["r"])
                                episode_lengths.append(info["episode"]["l"])
                                # Only increment at the real end of an episode
                                episode_counts[i] += 1
                            # reset
                            action_mask = torch.zeros([num_move_out_counties, num_move_in_counties], dtype=bool).to(action_mask.device)
                            Move_in, Move_out = model.Move_in_origin.copy(), model.Move_out_origin.copy()
                            Target_move_in, Target_move_out = copy.deepcopy(model.Target_move_in_origin), copy.deepcopy(model.Target_move_out_origin)
    
                            Move_in_tensor_N_demand = torch.tensor(Target_move_in['N_demand'], dtype=torch.float64).to(model.device)
                            Move_in_tensor_ammonia_density = torch.tensor(Target_move_in['ammonia_density'], dtype=torch.float64).to(model.device)
                            Move_in_tensor_livestock_PB = torch.tensor(Target_move_in['livestock_PB'], dtype=torch.float64).to(model.device)

                            Move_out_tensor_N_demand = torch.tensor(Target_move_out['N_demand'], dtype=torch.float64).to(model.device)
                            Move_out_tensor_ammonia_density = torch.tensor(Target_move_out['ammonia_density'], dtype=torch.float64).to(model.device)
                            Move_out_tensor_livestock_PB = torch.tensor(Target_move_out['livestock_PB'], dtype=torch.float64).to(model.device)

                            update_action_mask()
                            env.unwrapped.envs[0].env.env.env.action_mask_left  = model.action_mask_sum_origin - action_mask.sum()
                            
                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0

            observations = new_observations

            if render:
                env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    if save_path:
        import os
        if not os.path.exists(os.path.join(save_path, model.country)):
            os.makedirs(os.path.join(save_path, model.country))
        pd.DataFrame(log, columns=["move_out_idx", "move_in_idx", "amount", "reward", "action_mask.sum()"], index=None).to_excel(f"{save_path}/{model.country}/PPO1.xlsx")
    return mean_reward, std_reward