from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan
from stable_baselines3.common.evaluation_action_mask_v2 import evaluate_policy
from livestock_rl import LivestockEnv, LivestockEnvConfig


def main():
    country = 'cn'
    config = LivestockEnvConfig(
        country,
        Reward_priority=[4, 2, 1],
        thresholds=[0, 0],
        mobility_ratio=0.25,
        max_steps=8000,
        df_path='中国优化N优先v2.xlsx',
    )
    env = VecCheckNan(make_vec_env(lambda: LivestockEnv(config), n_envs=1), raise_exception=True)
    model = PPO_action_mask_v2.load(rf'../logs/v2/{country}/best_model.zip', env=env)
    evaluate_policy(
        model, env, n_eval_episodes=1, deterministic=True, render=False,
        action_mask=None, save_path="../results/v2",
    )


if __name__ == "__main__":
    main()
