from operator import is_
from re import L
import stat
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import pandas as pd
import sys
sys.path.append('..') 
np.random.seed(0) # 为了保证每次运行结果一致，设置随机种子
import torch
from scipy.optimize import linprog
from data_loader import load_datas, country_mapping
import copy

class LivestockEnvConfig:
    def __init__(self, country, Reward_priority: list, thresholds: list, mobility_ratio=0.25, max_steps=5000):
        self.country = country
        self.mobility_ratio = mobility_ratio
        self.Reward_priority = Reward_priority
        self.thresholds = thresholds
        self.max_steps = max_steps


class LivestockEnv(gym.Env):
    def __init__(self, config: LivestockEnvConfig):
        super(LivestockEnv, self).__init__()
        
        self.country = config.country
        self.reward_priority = config.Reward_priority # 初始化优先级权重
        self.mobility_ratio = config.mobility_ratio
        self.thresholds = config.thresholds
        self.max_steps = config.max_steps

        # load data
        self.ID_move_in, self.ID_move_out, self.Move_in, self.Move_out, self.Target_move_in, self.Coef_move_in, self.Target_move_out, self.Coef_move_out = load_datas(self.country)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # original data
        self.Move_in_origin = self.Move_in.copy()
        self.Move_out_origin = self.Move_out.copy()
        self.Target_move_in_origin = copy.deepcopy(self.Target_move_in)
        self.Target_move_out_origin = copy.deepcopy(self.Target_move_out)

        self.num_move_out_counties = self.Move_out.shape[0]
        self.num_move_in_counties = self.Move_in.shape[0]
        self.current_step = 0
      
        self.action_len = self.Move_out.shape[1]

        self.action_space = spaces.Discrete(
            self.num_move_out_counties * self.num_move_in_counties       
            )

        self.observation_space = spaces.Dict({
            'Amount_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, self.Move_in.shape[1]), dtype=np.int64),
            'Amount_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, self.Move_out.shape[1]), dtype=np.int64),
            # 'N_demand_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            # 'N_demand_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, 1), dtype=np.float64),
            'ammonia_density_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            'ammonia_density_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, 1), dtype=np.float64),
            'livestock_PB_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            'livestock_PB_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, 1), dtype=np.float64),
            'sensitivity_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            'sensitivity_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, 1), dtype=np.float64),
            'relative_pm25_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            'relative_pm25_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, 1), dtype=np.float64),
        })
                
        # self.Move_in_tensor_N_demand = torch.tensor(self.Target_move_in['N_demand'], dtype=torch.float64).to(self.device)
        self.Move_in_tensor_ammonia_density = torch.tensor(self.Target_move_in['ammonia_density'], dtype=torch.float64).to(self.device)
        self.Move_in_tensor_livestock_PB = torch.tensor(self.Target_move_in['livestock_PB'], dtype=torch.float64).to(self.device)
        self.Move_in_tensor_sensitivity = torch.tensor(self.Target_move_in['sensitivity'], dtype=torch.float64).to(self.device)
        self.Move_in_tensor_relative_pm25 = torch.tensor(self.Target_move_in['relative_pm25'], dtype=torch.float64).to(self.device)

        # self.Move_out_tensor_N_demand = torch.tensor(self.Target_move_out['N_demand'], dtype=torch.float64).to(self.device)
        self.Move_out_tensor_ammonia_density = torch.tensor(self.Target_move_out['ammonia_density'], dtype=torch.float64).to(self.device)
        self.Move_out_tensor_livestock_PB = torch.tensor(self.Target_move_out['livestock_PB'], dtype=torch.float64).to(self.device)
        self.Move_out_tensor_sensitivity = torch.tensor(self.Target_move_out['sensitivity'], dtype=torch.float64).to(self.device)
        self.Move_out_tensor_relative_pm25 = torch.tensor(self.Target_move_out['relative_pm25'], dtype=torch.float64).to(self.device)

        # self.Move_in_tensor_Coef_N_demand = torch.tensor(self.Coef_move_in['N_demand'].values, dtype=torch.float64).to(self.device)
        self.Move_in_tensor_Coef_ammonia_density = torch.tensor(self.Coef_move_in['ammonia_density'].values, dtype=torch.float64).to(self.device)
        self.Move_in_tensor_Coef_livestock_PB = torch.tensor(self.Coef_move_in['livestock_PB'].values, dtype=torch.float64).to(self.device)

        # self.Move_out_tensor_Coef_N_demand = torch.tensor(self.Coef_move_out['N_demand'].values, dtype=torch.float64).to(self.device)
        self.Move_out_tensor_Coef_ammonia_density = torch.tensor(self.Coef_move_out['ammonia_density'].values, dtype=torch.float64).to(self.device)
        self.Move_out_tensor_Coef_livestock_PB = torch.tensor(self.Coef_move_out['livestock_PB'].values, dtype=torch.float64).to(self.device)
        
        self.amounts = torch.tensor((self.Move_out * self.mobility_ratio).values, dtype=torch.int64).to(self.device)
           
        self.action_mask_left = None
        self.reset()
    
    @staticmethod
    def ammonia_density_case(ammonia_density):
        return ammonia_density>=99999


    def update_state(self, move_in_idx, move_out_idx, amounts):

        self.state['Amount_Move_in'][move_in_idx, :] += amounts.cpu().numpy()
        self.state['Amount_Move_out'][move_out_idx, :] -= amounts.cpu().numpy()
        # self.state['N_demand_Move_in'][move_in_idx] += (amounts.double() @ self.Move_in_tensor_Coef_N_demand[move_in_idx, :]).item()
        # self.state['N_demand_Move_out'][move_out_idx] -= (amounts.double() @ self.Move_out_tensor_Coef_N_demand[move_out_idx, :]).item()
        self.state['ammonia_density_Move_in'][move_in_idx] += (amounts.double() @ self.Move_in_tensor_Coef_ammonia_density[move_in_idx, :]).item()
        self.state['ammonia_density_Move_out'][move_out_idx] -= (amounts.double() @ self.Move_out_tensor_Coef_ammonia_density[move_out_idx, :]).item()
        self.state['livestock_PB_Move_in'][move_in_idx] += (amounts.double() @ self.Move_in_tensor_Coef_livestock_PB[move_in_idx, :]).item()
        self.state['livestock_PB_Move_out'][move_out_idx] -= (amounts.double() @ self.Move_out_tensor_Coef_livestock_PB[move_out_idx, :]).item()
        
        return self.state
    
    def detect_violation(self, move_out_idx, move_in_idx):
        
        # 移入和移出地区的牲畜数量是否小于0
        Amounts_violation = (self.state['Amount_Move_out'][move_out_idx] < 0).any() or (self.state['Amount_Move_in'][move_in_idx] < 0).any()
        # TODO
        return Amounts_violation
    
    def get_total_reward(self, move_in_index, move_out_index):
        """计算移入和移出县的奖励，并确保全局达标。"""
        # 奖励初始化    
        reward = 0
        # 移入县的奖励
        reward += self._evaluate_county(move_in_index, is_move_in=True)
        # 移出县的奖励
        reward += self._evaluate_county(move_out_index, is_move_in=False)

        if self.ammonia_density_case(self.Target_move_out_origin['ammonia_density'][move_out_index]):    
            reward += 50  # 给予奖励
        return reward

    def _evaluate_county(self, idx, is_move_in):
        """根据县的状态计算奖励，移入和移出县分别处理。"""
        reward = 0
        if is_move_in:
            name = '_Move_in'
        else:
            name = '_Move_out'
        if is_move_in:
            sensitivity, pm25relative = self.evaluate_sensitivity_and_pm25relative(idx)
        ammonia_density_reward = self.get_reward(f'ammonia_density{name}', idx, self.thresholds[0], self.reward_func)
        livestock_PB_reward= self.get_reward(f'livestock_PB{name}', idx, self.thresholds[1], self.reward_func)
        
        if is_move_in:
            for i, _ in enumerate([sensitivity, pm25relative, ammonia_density_reward, livestock_PB_reward]):
                reward += _ * self.reward_priority[i]
        else:
            for i, _ in enumerate([ammonia_density_reward, livestock_PB_reward]):
                reward += _ * self.reward_priority[2+i]

        # # 移入县和移出县的微调系数
        # if not is_move_in:
        #     reward *= 1.1  # 移入县的奖励偏重
        # else:
        #     reward *= 0.9  # 移出县的奖励偏轻

        return reward
    
    def evaluate_sensitivity_and_pm25relative(self, move_in_idx):
        # 根据移入地区的承载力情况计算奖励
        sensitivity = self.state['sensitivity_Move_in'][move_in_idx]
        relative_pm25 = self.state['relative_pm25_Move_in'][move_in_idx]
        return torch.sigmoid(torch.tensor(sensitivity)).item(), torch.sigmoid(torch.tensor(relative_pm25)).item()
    
    def get_reward(self, attr, idx, threshold, func=None):
        return func(torch.tensor(self.state[attr][idx]), threshold, 0.5)
    
    def check_termination_condition(self):
        # 检查所有移入和移出的县是否满足终止条件
        
        ammonia_density_condition = (
            (self.state['ammonia_density_Move_in'] == self.thresholds[0]).all() and
            (self.state['ammonia_density_Move_out'] == self.thresholds[0]).all()
        )
        
        livestock_PB_condition = (
            (self.state['livestock_PB_Move_in'] == self.thresholds[1]).all() and
            (self.state['livestock_PB_Move_out'] == self.thresholds[1]).all()
        )
        
        return ammonia_density_condition and livestock_PB_condition

    @staticmethod
    def reward_func(x, x0, sigma):
        return torch.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    def step(self, action):
        move_out_idx = action // self.Move_in.shape[0]
        move_in_idx = action % self.Move_in.shape[0]
        
        amounts = self.move_amount

        if amounts is None:
            raise ValueError("Amounts is None")
                    
        self.state = self.update_state(move_in_idx, move_out_idx, amounts)
        self.current_step += 1

        # 计算奖励
        # self.total_left_amounts -= amounts.sum()  # 剩余牲畜数量

        # 检测约束满足
        amounts_violation = self.detect_violation(move_out_idx, move_in_idx)
                
        reward = self.get_total_reward(move_in_idx, move_out_idx)

        final_terminated = self.check_termination_condition()
        terminated = self.action_mask_left==0 or final_terminated or self.current_step >= self.max_steps
        
        
        if terminated:
            print("终止")

        if final_terminated:
            print("理想终止条件满足，任务完成！")
            reward += 100
        
        truncated = False

        if amounts_violation:
            reward -= -10
            truncated = True    

        info = {
            'Amounts_violation': amounts_violation,
            'Move_out_idx': move_out_idx,
            'Move_in_idx': move_in_idx,
            'Amounts': amounts,
            'Reward': reward,
        }

        return self.state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.move_amount = None
        self.current_step = 0

        self.Move_in = self.Move_in_origin.copy()
        self.Move_out = self.Move_out_origin.copy()
        self.Target_move_out = copy.deepcopy(self.Target_move_out_origin)
        self.Target_move_in = copy.deepcopy(self.Target_move_in_origin)

        self.state = {
            'Amount_Move_in': self.Move_in.values,
            'Amount_Move_out': self.Move_out.values,
            'ammonia_density_Move_in': self.Target_move_in['ammonia_density'].values.reshape(-1, 1),
            'ammonia_density_Move_out': self.Target_move_out['ammonia_density'].values.reshape(-1, 1),
            'livestock_PB_Move_in': self.Target_move_in['livestock_PB'].values.reshape(-1, 1),
            'livestock_PB_Move_out': self.Target_move_out['livestock_PB'].values.reshape(-1, 1),
            'sensitivity_Move_in': self.Target_move_in['sensitivity'].values.reshape(-1, 1),
            'sensitivity_Move_out': self.Target_move_out['sensitivity'].values.reshape(-1, 1),
            'relative_pm25_Move_in': self.Target_move_in['relative_pm25'].values.reshape(-1, 1),
            'relative_pm25_Move_out': self.Target_move_out['relative_pm25'].values.reshape(-1, 1),
        }

        # 直接返回 state 作为 observation
        return self.state, {}

    def render(self, mode='human'):
        # Render environment (optional)
        print(f"step:{env.current_step}, || Action: {action}, Reward: {reward}, terminated: {terminated}, truncated:{truncated},info:{info}")

if __name__ == "__main__":
    # for country in ['aus', 'br', 'cn', 'usa', 'us']:
    
    for country in ['cn']:
        config = LivestockEnvConfig(country, Reward_priority=[4, 4, 2, 1], thresholds=[0, 31, 0], mobility_ratio=0.25)
        env = LivestockEnv(config)
        state, _ = env.reset()
        print("Initial State:", )
        env.move_amount = torch.tensor([1] * env.action_len, dtype=torch.int64).to(env.device)
        
        terminated, truncated = False, False
        total_reward = 0
        while not terminated and not truncated:
            action = env.action_space.sample()
            state,reward,terminated,truncated,info = env.step(action)
            total_reward += reward
            print(f"<{country}> step:{env.current_step}, || Action: {action}, Reward: {reward}, terminated: {terminated}, truncated:{truncated},info:{info}")

        print("Total Reward:", total_reward)
        # print(env.action_mask.sum())