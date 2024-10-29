import os
import sys
sys.path.append("stable-baselines3")
sys.path.append("强化学习")
sys.path.append("..")
from torch.nn.modules.activation import F
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gymnasium as gym
from stable_baselines3 import PPO_action_mask
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from livestockEnv import load_datas, country_mapping
from logger import setup_logger
import argparse
import numpy as np
import torch
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', type=str, default='aus')
   
    args = parser.parse_args()
    register(
        id='LivestockEnv-v2',
        entry_point='livestockEnvV2:LivestockEnv',
    )
    country = args.country
    if country_mapping[country] in ['巴西', '欧盟']:
        k = 2
    elif country_mapping[country]=='中国':
        k = 4
    else:
        k = 3

    def detect_violation(move_out_idx, move_in_idx):
        global Nh3_meta, Carrying_meta, amounts,Move_in, Move_out
        amount = amounts[move_out_idx]
        
        Move_in_values = Move_in.iloc[move_in_idx]
        NH3_violation = Move_in_values['氨排放差距'] < NH3_meta[move_out_idx, move_in_idx].item()
        Carrying_violation = Move_in_values['承载力差距'] < Carrying_meta[move_out_idx, move_in_idx].item()
        Amounts_violation = (Move_out.iloc[move_out_idx, k:].values < amount).any() 
        empty_violation = Move_out.iloc[move_out_idx, k:].sum() == 0
        
        return NH3_violation, Carrying_violation, Amounts_violation, empty_violation
    
    # 创建并包装环境
    env = make_vec_env('LivestockEnv-v2', n_envs=1, env_kwargs={'country': country})
    env = VecCheckNan(env, raise_exception=True)


    # 测试训练好的模型
    obs = env.reset()
    done = False
    i = 1

    Move_in, Move_out,  sensibility, manure = load_datas(country)
    Move_out_origin = Move_out.copy()
    action_len = Move_out.columns[k:].shape[0]
    mobility_ratio = 0.01

    logger = setup_logger(name=f'{country}PPO6')
    obs = env.reset()
    done = False
    i = 1

    model = PPO_action_mask.load(rf'D:\Researchs\畜牧业空间规划\logs\{country}\V1\best_model.zip')
    
    amounts = torch.tensor((Move_out_origin.iloc[:, k:].values * mobility_ratio), dtype=torch.float32).to(device)
    Move_in_tensor = torch.tensor(Move_in.iloc[:, k + 1:k + 1 + action_len].values,dtype=torch.float32).to(device)
    NH3_meta = torch.mm(amounts, Move_in_tensor.t()).cuda()

    Move_in_tensor_carrying = torch.tensor(Move_in.iloc[:, -action_len:].values,dtype=torch.float32).to(device)
    Carrying_meta = torch.mm(amounts, Move_in_tensor_carrying.t())

    amounts = amounts.cpu().numpy().astype(np.int32)
    Move_in_tensor = Move_in_tensor.cpu().numpy()
    NH3_meta = NH3_meta.cpu().numpy()
    Carrying_meta = Carrying_meta.cpu().numpy()
    Move_in_tensor_carrying = Move_in_tensor_carrying.cpu().numpy()
    num_move_out_counties, num_move_in_counties = Move_out.shape[0], Move_in.shape[0]
    # action_mask = torch.load(f'D:\Researchs\畜牧业空间规划\强化学习\{country}action_mask.pth')
    action_mask = torch.zeros(num_move_out_counties, num_move_in_counties, dtype=bool).to(device)
    def update_action_mask(move_out_idx=None, move_in_idx=None):
        action_mask[move_out_idx] = torch.logical_or(action_mask[move_out_idx].data,torch.from_numpy(detect_violation_move_out(move_out_idx)).to(device))
        action_mask[:, move_in_idx] = torch.logical_or(action_mask[:, move_in_idx].data, torch.from_numpy(detect_violation_move_in(move_in_idx)).to(device))
        
                
    def detect_violation_move_out(move_out_idx):
        global Move_in, Move_out
        amount = amounts[move_out_idx]
        NH3_violation = Move_in['氨排放差距'].values < NH3_meta[move_out_idx, :]
        Carrying_violation = Move_in['承载力差距'].values < Carrying_meta[move_out_idx, :]
        Amounts_violation = (Move_out.iloc[move_out_idx, model.k:] < amount).any()
        empty_violation = amount.sum() == 0
        if Amounts_violation or empty_violation:
            return np.ones(action_mask.shape[1])
        else:
            return np.logical_or(NH3_violation, Carrying_violation)
        
    def detect_violation_move_in(move_in_idx):
        NH3_violation = Move_in.iloc[move_in_idx]['氨排放差距'] < NH3_meta[:, move_in_idx]
        Carrying_violation = Move_in.iloc[move_in_idx]['承载力差距'] < Carrying_meta[:, move_in_idx]
        return np.logical_or(NH3_violation, Carrying_violation)
    
    def update_Move_df(move_out_idx, move_in_idx):
        # Move_in_values = Move_in.iloc[move_in_idx]
        global Move_in, Move_out
        Move_in.at[move_in_idx,'氨排放差距'] -= NH3_meta[move_out_idx, move_in_idx].item()
        Move_in.at[move_in_idx,'承载力差距'] -= Carrying_meta[move_out_idx, move_in_idx].item()
        Move_out.iloc[move_out_idx, model.k:] -= amounts[move_out_idx]

    while not done:
        action, _states = model.predict(obs,deterministic=False, mask=action_mask)
        move_out_idx, move_in_idx = action.item() // num_move_in_counties, action.item()  % num_move_in_counties
        violation = detect_violation(move_out_idx, move_in_idx)
        while True in violation and action_mask.shape[0] * action_mask.shape[1] - action_mask.sum() > 0:
            action_mask[move_out_idx, move_in_idx] = True
            action, _states = model.predict(obs,deterministic=False, mask=action_mask)
            move_out_idx, move_in_idx = action.item() // num_move_in_counties, action.item()  % num_move_in_counties
            violation = detect_violation(move_out_idx, move_in_idx)
            # print(move_out_idx, move_in_idx,action_mask[move_out_idx, move_in_idx])
        obs, reward, done, info = env.step(action)
        update_Move_df(move_out_idx, move_in_idx)
        update_action_mask(move_out_idx, move_in_idx)
        logger.info(f"step:{i}||action:{action} {info} reward:{reward}")
        i+=1
        # env.render()
    # logger.info(action_mask.shape,action_mask.sum())
    logger.info(action_mask.shape[0] * action_mask.shape[1] - action_mask.sum())
    logger.info(1 - info[0]['livestock_left'] / Move_out_origin.iloc[:, k:].sum().sum())