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
# from livestockEnv import load_datas
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


    Move_in, Move_out = model.Move_in_origin.values.copy(), model.Move_out_origin.values.copy()
    
    
    Move_in_tensor_N_demand = model.Move_in_tensor_N_demand_origin.clone()
    Move_out_tensor_N_demand = model.Move_out_tensor_N_demand_origin.clone()
    Move_in_tensor_Ammonia = model.Move_in_tensor_Ammonia_origin.clone()
    Move_out_tensor_Ammonia = model.Move_out_tensor_Ammonia_origin.clone()
   
    num_move_out_counties, num_move_in_counties = Move_out.shape[0], Move_in.shape[0]

    if action_mask is None:
        action_mask = torch.zeros([model.num_move_out_counties, model.num_move_in_counties], dtype=bool).to(model.device)

    def update_action_mask(move_out_idx=None, move_in_idx=None, violation=None):
        '''
        action mask update function
        '''
        if move_out_idx==None:   
            for out_idx in range(model.num_move_out_counties):
                if model.Move_out_origin.iloc[out_idx, :].sum() == 0:
                    action_mask[out_idx, :] = True
                else:
                    action_mask[out_idx, :] = detect_violation_move_out(out_idx).to(model.device)
            for in_idx in range(model.num_move_in_counties):
                if model.Move_in_origin.iloc[in_idx, :].sum() == 0:
                    action_mask[:, in_idx] = True
                else:
                    action_mask[:, in_idx] = torch.logical_or(action_mask[:, in_idx], detect_violation_move_in(in_idx).to(model.device))
        else:
            action_mask[move_out_idx, :] = torch.logical_or(action_mask[move_out_idx, :].data,
                                                            detect_violation_move_out(move_out_idx).to(action_mask.device))
            action_mask[:, move_in_idx] = torch.logical_or(action_mask[:, move_in_idx].data,
                                                            detect_violation_move_in(move_in_idx).to(action_mask.device))
            if violation:
                action_mask[move_out_idx, move_in_idx] = True
                            
    def detect_violation_move_out(move_out_idx):
        cur_amounts = Move_out[move_out_idx, :]

        empty_violation =(cur_amounts == 0).all()
        # NH3_violation = Move_out_tensor_Ammonia[move_out_idx] <= model.thresholds[0] - 0.1
        if not empty_violation:
            N_violation = Move_out_tensor_N_demand[move_out_idx] < (model.thresholds[0] + model.Move_out_tensor_Coef_N_demand[move_out_idx][cur_amounts != 0].min())
        else:
            N_violation = False
        if empty_violation or N_violation:
            return torch.ones(model.num_move_in_counties) 
        else: 
            ammonia_bounds = torch.min(model.Move_in_tensor_Coef_Ammonia[:, cur_amounts != 0], axis=1).values.reshape(-1, 1)
            N_bounds = torch.min(model.Move_in_tensor_Coef_N_demand[:, cur_amounts != 0], axis=1).values.reshape(-1, 1)
            amount_constraint = torch.tensor((Move_in > 0) @ (cur_amounts > 0)).to(model.device).unsqueeze(1)
            
            return torch.logical_or(
                        torch.logical_or(
                            torch.logical_or(
                                Move_in_tensor_Ammonia - ammonia_bounds < model.thresholds[1], # 氨剩余排放量小于当前最小氨排放变化量
                                Move_in_tensor_N_demand + N_bounds > model.thresholds[0] # 氮剩余需求量小于当前最小氮需求变化量
                            ),
                            torch.tensor(((cur_amounts == 0) + (Move_in == 0)).sum(axis=1) == model.action_len).reshape(-1, 1).to(model.device)) # 移入县结构约束交移出县剩余牲畜数量
                            ,~amount_constraint
                    ).flatten() 

    def detect_violation_move_in(move_in_idx):
        cur_amounts = Move_in[move_in_idx, :]
        N_violation = Move_in_tensor_N_demand[move_in_idx] > model.thresholds[0] - model.Move_in_tensor_Coef_N_demand[move_in_idx][cur_amounts != 0].min()
        NH3_violation = Move_in_tensor_Ammonia[move_in_idx] <  model.thresholds[1] + model.Move_in_tensor_Coef_Ammonia[move_in_idx][cur_amounts != 0].min()

        if NH3_violation or N_violation:
            return torch.ones(model.num_move_out_counties)
        else:
            return (Move_out_tensor_N_demand < model.thresholds[0] - torch.min(model.Move_out_tensor_Coef_N_demand[:, cur_amounts != 0], axis=1).values.reshape(-1, 1)).flatten()

    def detect_violation(move_out_idx, move_in_idx, amounts):

        cur_amounts_out = Move_out[move_out_idx, :]
        Amounts_violation = (cur_amounts_out < amounts.cpu().numpy()).any()
        empty_violation = (amounts == 0).all()
        Ammonia_violation = Move_in_tensor_Ammonia[move_in_idx] < model.thresholds[1] + amounts.double() @ model.Move_in_tensor_Coef_Ammonia[move_in_idx] - 1e-2
        N_violation_out = Move_out_tensor_N_demand[move_out_idx] < model.thresholds[0] + amounts.double() @ model.Move_out_tensor_Coef_N_demand[move_out_idx] - 1
        N_violation_in = Move_in_tensor_N_demand[move_in_idx] > model.thresholds[0] - amounts.double() @ model.Move_in_tensor_Coef_N_demand[move_in_idx] + 1e-2
        
        return empty_violation, Amounts_violation, Ammonia_violation, N_violation_out, N_violation_in

    
    def update_Move_df(move_out_idx, move_in_idx, amounts):
        Move_in[move_in_idx, :] += amounts.cpu().numpy()
        Move_out[move_out_idx, :] -= amounts.cpu().numpy()
        
        Move_in_tensor_N_demand[move_in_idx] += (amounts.double() @ model.Move_in_tensor_Coef_N_demand[move_in_idx]).item()
        Move_out_tensor_N_demand[move_out_idx] -= (amounts.double() @ model.Move_out_tensor_Coef_N_demand[move_out_idx]).item()
        Move_in_tensor_Ammonia[move_in_idx] -= (amounts.double() @ model.Move_in_tensor_Coef_Ammonia[move_in_idx, :]).item()
        Move_out_tensor_Ammonia[move_out_idx] -= (amounts.double() @ model.Move_out_tensor_Coef_Ammonia[move_out_idx, :]).item()
    

    def amount_adapt(move_out_idx, move_in_idx):
        amounts = model.amounts[move_out_idx, :].clone()
        # 移入县结构性约束
        amounts[model.Move_in_origin.iloc[move_in_idx, :].values == 0] = 0
        if not True in detect_violation(move_out_idx, move_in_idx, amounts):
            return amounts
        
        cur_amounts = Move_out[move_out_idx, :]
        cur = torch.tensor([Move_in_tensor_Ammonia[move_in_idx],
                            Move_in_tensor_N_demand[move_in_idx],
                            Move_out_tensor_N_demand[move_out_idx]]).cpu().numpy().reshape(-1, 1)
        if (cur_amounts == 0).all() or cur[0] <= model.thresholds[1] or cur[1] >= model.thresholds[0] or cur[2] <= model.thresholds[0]:
            return torch.zeros(model.action_len, dtype=torch.int64).to(amounts.device)

        Move_in_amounts_original = model.Move_in_origin.iloc[move_in_idx, :].values
        Move_out_amounts_original = model.Move_out_origin.iloc[move_out_idx, :].values

        bounds = [(0, min(amounts[i].item()+1, cur_amounts[i])) for i in range(len(cur_amounts))]
        _, upbounds = zip(*bounds)
        if (np.array(upbounds) == 0).all():
            return torch.zeros(model.action_len, dtype=torch.int64).to(amounts.device)

        Ammonia_coef_in = model.Move_in_tensor_Coef_Ammonia[move_in_idx].cpu().numpy()
        N_demand_coef_out = model.Move_out_tensor_Coef_N_demand[move_out_idx].cpu().numpy()
        N_demand_coef_in = model.Move_in_tensor_Coef_N_demand[move_in_idx].cpu().numpy()
           
        # 等比例移入目标
        c = -(Move_in_amounts_original / Move_in_amounts_original.max()) * (Move_out_amounts_original / max(1, Move_out_amounts_original.max())) # Coefficients for the linear objective function, 保证除数不为0

        A_ub = np.array([Ammonia_coef_in,
                         N_demand_coef_in,
                         N_demand_coef_out,
                         ])
        
        b_ub_NH3_in = cur[0] - np.array([model.thresholds[1] - 1e-2]).reshape(-1, 1)
        b_ub_N_in = np.array([model.thresholds[0] + 1e-2]).reshape(-1, 1) - cur[1]
        b_ub_N_out = cur[2] - np.array([model.thresholds[0] - 1]).reshape(-1, 1)
        b_ub = np.concatenate((b_ub_NH3_in, b_ub_N_in, b_ub_N_out), axis=0).reshape(-1, 1)


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
    update_action_mask()
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

            while True in violation and action_mask.sum() < num_move_in_counties* num_move_out_counties and counter < 500000: # if available amounts search round > 500, break
                update_action_mask(move_out_idx, move_in_idx, True)  # 更新 action_mask
                pbar.set_postfix({"reward":-1, "mask":action_mask.sum()})
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
            if counter >= 500000:
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
                log.append([move_out_idx, move_in_idx, *model.ID_move_out.iloc[move_out_idx], *model.ID_move_in.iloc[move_in_idx],amount, rewards[0], action_mask.sum().item()])
            
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
                            Move_in, Move_out = model.Move_in_origin.values.copy(), model.Move_out_origin.values.copy()
                                
                            Move_in_tensor_N_demand = model.Move_in_tensor_N_demand_origin.clone()
                            Move_out_tensor_N_demand = model.Move_out_tensor_N_demand_origin.clone()
                            Move_in_tensor_Ammonia = model.Move_in_tensor_Ammonia_origin.clone()
                            Move_out_tensor_Ammonia = model.Move_out_tensor_Ammonia_origin.clone()

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
        pd.DataFrame(log, columns=["move_out_idx", "move_in_idx", "ID_move_out", "city_move_out", "county_move_out", "ID_move_in", "city_move_in", "county_move_in", "amount", "reward", "action_mask.sum()"], index=None).to_excel(f"{save_path}/PPO.xlsx")
    return mean_reward, std_reward