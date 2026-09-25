import os
from pathlib import Path
from typing import Callable

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pandas as pd
from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecCheckNan
from livestock_rl import LivestockEnv, LivestockEnvConfig
from livestock_rl.AttentionPolicy import CustomAttentionPolicy
from livestock_rl.data_loader import country_mapping

country = 'usa'
FilePath = '美国数据分省尺度第二步1224.xlsx'
version = 'v9-2'

# 加载数据
df_out = pd.read_excel(Path(__file__).resolve().parents[1] / 'data' / country_mapping[country] / FilePath, sheet_name='移出')

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
for province in df_out['province'].unique():
    try:
        print("-------------------",province,"-------------------")
        config = LivestockEnvConfig(country,
                                    Reward_priority=[4, 2, 1],
                                    thresholds=[0, 0],
                                    mobility_ratio=0.1,
                                    max_steps=30000,
                                    df_path=FilePath,
                                    province=province,)

        # 创建并包装环境
        env = make_vec_env(lambda: LivestockEnv(config), n_envs=1)
        env = VecCheckNan(env, raise_exception=True)
        eval_env = make_vec_env(lambda: LivestockEnv(config), n_envs=1)
        eval_env = VecCheckNan(eval_env, raise_exception=True)
        eval_callback = EvalCallback(eval_env, best_model_save_path=f'../logs/{version}/{country}/{province}',
                                    log_path='../logs/', eval_freq=2**13+1,
                                    deterministic=False, render=False)

        model = PPO_action_mask_v2(CustomAttentionPolicy,
                                env,
                                batch_size=512,
                                verbose=1,
                                tensorboard_log='./board/',
                                seed=42,
                                kwargs={'country':country},
                                learning_rate=linear_schedule(5e-5),
                                n_steps=2**13,
                                )
        model.learn(total_timesteps=20000,tb_log_name = f"{country}PPO_{version}_{province}",callback=eval_callback)
        del model, env, config
    except Exception as e:
        print("-------------------",province,"failed-------------------")
        print(e)
    # break
