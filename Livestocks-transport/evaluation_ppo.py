import os
import sys
sys.path.append("../pythstable-baselines3")
sys.path.append("..")
from torch.nn.modules.activation import F
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import gymnasium as gym
from logger import setup_logger
from livestockEnvV2 import load_datas,LivestockEnvConfig
from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from AttentionPolicy import CustomAttentionPolicy
from stable_baselines3.common.evaluation_action_mask_v2 import evaluate_policy

register(
    id='LivestockEnv-v2',
    entry_point='livestockEnvV2:LivestockEnv',
)
country = 'cn'
config = LivestockEnvConfig(country, 
                            Reward_priority=[4, 4, 3, 2, 1], 
                            thresholds=[0, 31, 0], 
                            mobility_ratio=0.25,
                            max_steps=8000)

# 创建并包装环境
env = make_vec_env('LivestockEnv-v2', n_envs=1, env_kwargs={'config': config})
env = VecCheckNan(env, raise_exception=True)
model = PPO_action_mask_v2.load(rf'../logs/v2/{country}/best_model.zip', env=env)
evaluate_policy(model, env, n_eval_episodes=1, deterministic=True, render=False, action_mask=None, save_path="../results/v2")
