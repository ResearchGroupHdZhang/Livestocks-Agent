from ast import arg
import os
import sys
sys.path.append("../stable-baselines3")
sys.path.append("..")
from torch.nn.modules.activation import F
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gymnasium as gym
from logger import setup_logger
from livestockEnvV2 import load_datas,LivestockEnvConfig
from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from AttentionPolicy import CustomAttentionPolicy
import argparse
from typing import Callable



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', type=str, default='cn')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total timesteps for training')
    parser.add_argument('--mobility_ratio', type=float, default=0.1, help='Mobility ratio for training')
    parser.add_argument('--learning_rate', type=float, default=3e-3, help='Learning rate for training')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--Reward_priority', type=list, default=[4, 4, 3, 2, 1])
    parser.add_argument('--thresholds', type=list, default=[0, 31, 0])
    parser.add_argument('--max_steps', type=int, default=12000)
    
    args = parser.parse_args()
    register(
        id='LivestockEnv-v2',
        entry_point='livestockEnvV2:LivestockEnv',
    )
    config = LivestockEnvConfig(args.country, 
                                Reward_priority=args.Reward_priority, 
                                thresholds=args.thresholds, 
                                mobility_ratio=args.mobility_ratio,
                                max_steps=args.max_steps)

    # 创建并包装环境
    env = make_vec_env('LivestockEnv-v2', n_envs=1, env_kwargs={'config': config})
    env = VecCheckNan(env, raise_exception=True)
    eval_callback = EvalCallback(env, best_model_save_path=f'../logs/v2/{args.country}/',
                                log_path='./logs/', eval_freq=config.max_steps,
                                deterministic=False, render=False)

    model = PPO_action_mask_v2(CustomAttentionPolicy, 
                        env, 
                        batch_size=4, 
                        verbose=1, 
                        tensorboard_log='./board/',
                        seed=42,
                        kwargs={'country':args.country},
                        learning_rate=linear_schedule(2e-2),
                        n_steps=2**13,
                        )
    # model.learn(total_timesteps=args.total_timesteps,tb_log_name = f"{country}PPO928",callback=eval_callback, heat_rate=0.2)
    model.learn(total_timesteps=args.total_timesteps,tb_log_name = f"{args.country}PPO_v2",callback=eval_callback)




