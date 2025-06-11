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

register(
    id='LivestockEnv-v2',
    entry_point='livestockEnvV2:LivestockEnv',
)
country = 'eu'
config = LivestockEnvConfig(country, 
                            Reward_priority=[4, 2, 1], 
                            thresholds=[0, 0], 
                            mobility_ratio=0.02,
                            max_steps=50000,
                            df_path='欧盟更新PB后第一步.xlsx')

version = 'v10'
env = make_vec_env('LivestockEnv-v2', n_envs=1, env_kwargs={'config': config})
env = VecCheckNan(env, raise_exception=True)
eval_callback = EvalCallback(env, best_model_save_path=f'../logs/{version}/{country}/',
                             log_path='./logs/', eval_freq=2**15+1,
                             deterministic=False, render=False)
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
        Progress will decrease from 1 (beginning) to 0./

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
model = PPO_action_mask_v2(CustomAttentionPolicy, 
                        env, 
                        batch_size=256, 
                        verbose=1, 
                        tensorboard_log='./board/',
                        seed=42,
                        kwargs={'country':country},
                        learning_rate=linear_schedule(2e-5),
                        n_steps=2**15,
                        )
model.learn(total_timesteps=200000,tb_log_name = f"{country}PPO_{version}",callback=eval_callback)

