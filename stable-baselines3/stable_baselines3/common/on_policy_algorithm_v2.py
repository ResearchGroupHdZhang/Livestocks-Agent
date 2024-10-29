import sys
import time
from tkinter import N
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from tqdm import tqdm
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from 强化学习 import load_datas
from scipy.optimize import linprog
import copy

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        kwargs = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps        # 步长
        self.gamma = gamma            # 折扣因子
        self.learning_rate = learning_rate        # 学习率

        self.gae_lambda = gae_lambda           
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        
        # Yanis
        self.country = env.get_attr('country',0)[0]
        self.ID_move_in, self.ID_move_out, self.Move_in, self.Move_out, self.Target_move_in, self.Coef_move_in, self.Target_move_out, self.Coef_move_out = load_datas(self.country)

        self.Move_out_origin = self.Move_out.copy()
        self.Move_in_origin = self.Move_in.copy()
        self.Target_move_out_origin = copy.deepcopy(self.Target_move_out)
        self.Target_move_in_origin = copy.deepcopy(self.Target_move_in)

        self.Move_in_tensor_ammonia_density = th.tensor(self.Target_move_in['ammonia_density'], dtype=th.float64).to(self.device)
        self.Move_in_tensor_livestock_PB = th.tensor(self.Target_move_in['livestock_PB'], dtype=th.float64).to(self.device)

        self.Move_out_tensor_ammonia_density = th.tensor(self.Target_move_out['ammonia_density'], dtype=th.float64).to(self.device)
        self.Move_out_tensor_livestock_PB = th.tensor(self.Target_move_out['livestock_PB'], dtype=th.float64).to(self.device)

        self.ammonia_density_coef_move_in = self.env.get_attr('Move_in_tensor_Coef_ammonia_density',0)[0]
        self.livestock_PB_coef_move_in = self.env.get_attr('Move_in_tensor_Coef_livestock_PB',0)[0]
        
        self.ammonia_density_coef_move_out = self.env.get_attr('Move_out_tensor_Coef_ammonia_density',0)[0]
        self.livestock_PB_coef_move_out = self.env.get_attr('Move_out_tensor_Coef_livestock_PB',0)[0]

        self.thresholds = self.env.get_attr('thresholds',0)[0]
        self.num_move_out_counties, self.num_move_in_counties = self.Move_out.shape[0], self.Move_in.shape[0]
        
        self.mobility_ratio = self.env.get_attr('mobility_ratio',0)[0]

        self.action_len = self.env.get_attr("action_len", 0)[0]
        self.action_mask = self.reset_action_mask()
        self.action_mask_sum_origin = self.num_move_in_counties* self.num_move_out_counties
        
        self.amounts = self.env.get_attr('amounts',0)[0]
        self.ammonia_density_case = self.env.get_attr('ammonia_density_case',0)[0]
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
    
    def update_action_mask(self, move_out_idx=None, move_in_idx=None, violation:bool=None):
        # action mask initialization
        if move_out_idx==None:   
            for out_idx in range(self.num_move_out_counties):
                self.action_mask[out_idx, :] = True if (self.Move_out.iloc[out_idx, :] <= 0).all() else False
        else:
            self.action_mask[move_out_idx, :] = th.logical_or(self.action_mask[move_out_idx, :].data,
                                                                        self.detect_violation_move_out(move_out_idx).to(self.device))
            self.action_mask[:, move_in_idx] = th.logical_or(self.action_mask[:, move_in_idx].data,
                                                                    self.detect_violation_move_in(move_in_idx).to(self.device))
            if violation:
                self.action_mask[move_out_idx, move_in_idx] = True
                            
    def detect_violation_move_out(self, move_out_idx):
        # 移出县三者均需小于等于阈值 最大满足
        empty_violation = (self.Move_out.iloc[move_out_idx, :] == 0).all()

        ammonia_density_violation = self.Move_out_tensor_ammonia_density[move_out_idx] < self.thresholds[0] + self.ammonia_density_coef_move_out[move_out_idx].min()
        livestock_PB_violation = self.Move_out_tensor_livestock_PB[move_out_idx] < self.thresholds[1] + self.livestock_PB_coef_move_out[move_out_idx].min()
        
        if empty_violation:
            return th.ones(self.num_move_in_counties) 
        elif self.ammonia_density_case(self.Target_move_out_origin['ammonia_density'][move_out_idx]):
            return th.zeros(self.num_move_in_counties)
        elif ammonia_density_violation and livestock_PB_violation:
            return th.ones(self.num_move_in_counties) 
        else: 
            return th.zeros(self.num_move_in_counties)

    def detect_violation_move_in(self, move_in_idx):
        # 移入县三者满足其一 最小满足
        ammonia_density_violation = self.Move_in_tensor_ammonia_density[move_in_idx] > self.thresholds[0] - self.ammonia_density_coef_move_in[move_in_idx].min()
        livestock_PB_violation = self.Move_in_tensor_livestock_PB[move_in_idx] > self.thresholds[1] - self.livestock_PB_coef_move_in[move_in_idx].min()
        
        return th.ones(self.num_move_out_counties) if (ammonia_density_violation or livestock_PB_violation) else th.zeros(self.num_move_out_counties)

    def update_Move_df(self, move_in_idx, move_out_idx, amounts):
            
        self.Move_in.iloc[move_in_idx, :] += amounts.cpu().numpy()
        self.Move_out.iloc[move_out_idx, :] -= amounts.cpu().numpy()
      
        self.Move_in_tensor_ammonia_density[move_in_idx] += (amounts.double() @ self.ammonia_density_coef_move_in[move_in_idx]).item()
        self.Move_out_tensor_ammonia_density[move_out_idx] -=(amounts.double() @ self.ammonia_density_coef_move_out[move_out_idx]).item()
        
        self.Move_in_tensor_livestock_PB[move_in_idx] += (amounts.double() @ self.livestock_PB_coef_move_in[move_in_idx]).item()
        self.Move_out_tensor_livestock_PB[move_out_idx] -= (amounts.double() @ self.livestock_PB_coef_move_out[move_out_idx]).item()
    
    def detect_violation(self, move_out_idx, move_in_idx, amounts):
        
        cur_amounts = self.Move_out.iloc[move_out_idx, :]
        Amounts_violation = (cur_amounts < amounts.cpu().numpy()).any()
        empty_violation = (cur_amounts == 0).all() or (amounts == 0).all()

        ammonia_density_violation_move_out = self.Move_out_tensor_ammonia_density[move_out_idx] < self.thresholds[0] + amounts.double() @ self.ammonia_density_coef_move_out[move_out_idx]
        livestock_PB_violation_move_out = self.Move_out_tensor_livestock_PB[move_out_idx] < self.thresholds[1] + amounts.double() @ self.livestock_PB_coef_move_out[move_out_idx]
        
        ammonia_density_violation_move_in = self.Move_in_tensor_ammonia_density[move_in_idx] > self.thresholds[0] - amounts.double() @ self.ammonia_density_coef_move_in[move_in_idx]
        livestock_PB_violation_move_in = self.Move_in_tensor_livestock_PB[move_in_idx] > self.thresholds[1] - amounts.double() @ self.livestock_PB_coef_move_in[move_in_idx]

        
        return empty_violation,\
                Amounts_violation,\
                (ammonia_density_violation_move_in or livestock_PB_violation_move_in),\
                (ammonia_density_violation_move_out and livestock_PB_violation_move_out)

    def reset_action_mask(self):
        return th.zeros((self.num_move_out_counties, self.num_move_in_counties),dtype=bool).to(self.device)
    
    def action_proj(self, actions):
        move_out_idx, move_in_idx = actions // self.num_move_in_counties, actions % self.num_move_in_counties
        return move_out_idx.item(), move_in_idx.item()
    
    def amount_adapt(self, move_out_idx, move_in_idx):
        amounts = self.amounts[move_out_idx, :].clone()
      
        if not True in self.detect_violation(move_out_idx, move_in_idx, amounts):
            return amounts

        ammonia_density_coef_in = self.ammonia_density_coef_move_in[move_in_idx].cpu().numpy()
        livestock_PB_coef_in = self.livestock_PB_coef_move_in[move_in_idx].cpu().numpy()

        ammonia_density_coef_out = self.ammonia_density_coef_move_out[move_out_idx].cpu().numpy()
        livestock_PB_coef_out = self.livestock_PB_coef_move_out[move_out_idx].cpu().numpy()

        c = -np.ones(amounts.shape[0])  # Coefficients for the linear objective function
        cur_amounts = self.Move_out.iloc[move_out_idx, :]
        bounds = [(0, min(amounts[i].item(), cur_amounts.iloc[i])) for i in range(len(cur_amounts))]
        cur = th.tensor([self.Move_in_tensor_ammonia_density[move_in_idx],
                         self.Move_in_tensor_livestock_PB[move_in_idx],
                         self.Move_out_tensor_ammonia_density[move_out_idx],
                         self.Move_out_tensor_livestock_PB[move_out_idx]]).cpu().numpy().reshape(-1, 1)

        A_ub = np.array([ammonia_density_coef_in, livestock_PB_coef_in,
                         ammonia_density_coef_out, livestock_PB_coef_out])
        
        b_ub_in = np.array([self.thresholds]).reshape(-1, 1) - cur[:2]
        b_ub_out = cur[2:] - np.array([self.thresholds]).reshape(-1, 1)
        b_ub = np.concatenate((b_ub_in, b_ub_out), axis=0).reshape(-1, 1)

        idxs = b_ub < 0
        idxs = np.where(idxs)[0]
        if len(idxs) > 0:
            A_ub = np.delete(A_ub, idxs, axis=0)
            b_ub = np.delete(b_ub, idxs, axis=0)
        res = linprog(c, bounds=bounds, A_ub=A_ub, b_ub=b_ub, method='highs', integrality=1)
        if res.success:
            amounts = th.tensor(res.x, dtype=th.int64).to(amounts.device)
        return amounts
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        total_timesteps: int,
        # heat_rate:float,

    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # self.origin_reward_tables = self.init_reward() # reward_table初始化
        # self.reward_table = self.origin_reward_tables.clone()
        self.update_action_mask() # action mask初始化

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # policy action
                actions, values, log_probs = self.policy(obs_tensor, self.action_mask)

                move_out_idx, move_in_idx = self.action_proj(actions[0])
                amount = self.amount_adapt(move_out_idx, move_in_idx)
                violation = self.detect_violation(move_out_idx, move_in_idx, amount)
                counter = 0

                while True in violation and self.action_mask.sum() < self.action_mask_sum_origin and counter < 500:
                    self.update_action_mask(move_out_idx, move_in_idx, True)
                    print(f"step:{n_steps}, mask_sum_ratio:{self.action_mask.sum()/self.action_mask_sum_origin:.4f} | amount:{amount}")
                    
                    actions, values, log_probs = self.policy(obs_tensor, self.action_mask)

                    move_out_idx, move_in_idx = self.action_proj(actions[0])
                    amount = self.amount_adapt(move_out_idx, move_in_idx)
                    violation = self.detect_violation(move_out_idx, move_in_idx, amount)
                    counter += 1

            # Rescale and perform action
            if isinstance(actions, th.Tensor):
                actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
            self.update_Move_df(move_in_idx, move_out_idx, amount)

            self.update_action_mask(move_out_idx, move_in_idx, False)

            if counter >= 500:
                env.unwrapped.envs[0].env.env.env.action_mask_left = 0
            else:
                env.unwrapped.envs[0].env.env.env.action_mask_left = self.action_mask_sum_origin - self.action_mask.sum()
            
            self.env.unwrapped.envs[0].env.env.env.move_amount = amount
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1
            print(f"n_steps: {n_steps} || mask_sum_ratio:{self.action_mask.sum()/self.action_mask_sum_origin} || actions: {actions} || rewards: {rewards} || dones: {dones} || infos: {infos}")
            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
                if done:
                    self.action_mask = self.reset_action_mask()
                    self.Move_out = self.Move_out_origin.copy()
                    self.Move_in = self.Move_in_origin.copy()


                    self.Target_move_in, self.Target_move_out = copy.deepcopy(self.Target_move_in_origin), copy.deepcopy(self.Target_move_out_origin)
                    self.Move_in_tensor_ammonia_density = th.tensor(self.Target_move_in['ammonia_density'], dtype=th.float64).to(self.device)
                    self.Move_in_tensor_livestock_PB = th.tensor(self.Target_move_in['livestock_PB'], dtype=th.float64).to(self.device)

                    self.Move_out_tensor_ammonia_density = th.tensor(self.Target_move_out['ammonia_density'], dtype=th.float64).to(self.device)
                    self.Move_out_tensor_livestock_PB = th.tensor(self.Target_move_out['livestock_PB'], dtype=th.float64).to(self.device)

                    self.update_action_mask()
                    self.env.unwrapped.envs[0].env.env.env.action_mask_left  = self.action_mask_sum_origin
                    

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,

    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        # while self.num_timesteps < total_timesteps:
        #     continue_training = self.collect_rollouts(self.env, 
        #                                             callback, 
        #                                             self.rollout_buffer, 
        #                                             n_rollout_steps=self.n_steps, 
        #                                             total_timesteps=total_timesteps,
        #                                             heat_rate=heat_rate)
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, 
                                                    callback, 
                                                    self.rollout_buffer, 
                                                    n_rollout_steps=self.n_steps, 
                                                    total_timesteps=total_timesteps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
