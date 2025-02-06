from re import L
import stat
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
np.random.seed(0) # 为了保证每次运行结果一致，设置随机种子
import torch
from scipy.optimize import linprog
# from data_loader import load_datas, country_mapping
from data_loader import load_datas, country_mapping
import copy

class LivestockEnvConfig:
    def __init__(self, country,
                Reward_priority: list,
                thresholds: list, 
                df_path,
                province=None,
                mobility_ratio=0.25,
                max_steps=5000):
        self.country = country
        self.mobility_ratio = mobility_ratio
        self.Reward_priority = Reward_priority
        self.thresholds = thresholds
        self.max_steps = max_steps
        self.df_path = df_path
        self.province = province


class LivestockEnv(gym.Env):
    def __init__(self, config: LivestockEnvConfig):
        super(LivestockEnv, self).__init__()
        
        self.country = config.country
        self.reward_priority = config.Reward_priority # 初始化优先级权重
        self.mobility_ratio = config.mobility_ratio
        self.thresholds = config.thresholds
        self.max_steps = config.max_steps
        self.df_path = config.df_path
        self.province = config.province

        # load data
        self.ID_move_in, \
        self.ID_move_out, \
        self.Move_out_origin,\
        self.Move_in_origin,\
        self.N_demand_move_in,\
        self.N_demand_Coef_move_in,\
        self.Ammonia_move_in,\
        self.Ammonia_Coef_move_in,\
        self.sensitivity_move_in,\
        self.relative_pm25_move_in,\
        self.N_demand_move_out,\
        self.N_demand_Coef_move_out,\
        self.Ammonia_move_out,\
        self.Ammonia_Coef_move_out,\
        self.sensitivity_move_out,\
        self.relative_pm25_move_out = load_datas(self.country, self.df_path, self.province)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # original data
        self.Move_in_origin = self.Move_in_origin.values
        self.Move_out_origin = self.Move_out_origin.values
        self.N_demand_move_in_origin = self.N_demand_move_in.values.copy()
        self.N_demand_move_out_origin = self.N_demand_move_out.values.copy()
        self.Ammonia_move_out_origin = self.Ammonia_move_out.values.copy()
        self.Ammonia_move_in_origin = self.Ammonia_move_in.values.copy()

        self.num_move_out_counties = self.Move_out_origin.shape[0]
        self.num_move_in_counties = self.Move_in_origin.shape[0]
        self.current_step = 0
    
        self.action_len = self.Move_out_origin.shape[1]

        self.action_space = spaces.Discrete(
            self.num_move_out_counties * self.num_move_in_counties       
            )

        self.observation_space = spaces.Dict({
            'Amount_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, self.Move_in_origin.shape[1]), dtype=np.int64),
            'Amount_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, self.Move_out_origin.shape[1]), dtype=np.int64),
            'N_demand_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            'N_demand_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, 1), dtype=np.float64),
            'Ammonia_Move_in': spaces.Box(low=0, high=np.inf, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            'Ammonia_Move_out': spaces.Box(low=0, high=np.inf, shape=(self.num_move_out_counties, 1), dtype=np.float64),
            'sensitivity_Move_in': spaces.Box(low=0, high=1, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            'sensitivity_Move_out': spaces.Box(low=0, high=1, shape=(self.num_move_out_counties, 1), dtype=np.float64),
            'relative_pm25_Move_in': spaces.Box(low=0, high=1, shape=(self.num_move_in_counties, 1), dtype=np.float64),
            'relative_pm25_Move_out': spaces.Box(low=0, high=1, shape=(self.num_move_out_counties, 1), dtype=np.float64),
        })
                
        self.Move_in_tensor_N_demand = torch.tensor(self.N_demand_move_in.values, dtype=torch.float64).to(self.device)
        self.Move_out_tensor_N_demand = torch.tensor(self.N_demand_move_out.values, dtype=torch.float64).to(self.device)
        self.Move_in_tensor_Coef_N_demand = torch.tensor(self.N_demand_Coef_move_in.values, dtype=torch.float64).to(self.device)
        self.Move_out_tensor_Coef_N_demand = torch.tensor(self.N_demand_Coef_move_out.values, dtype=torch.float64).to(self.device)
        self.Move_in_tensor_Ammonia = torch.tensor(self.Ammonia_move_in.values, dtype=torch.float64).to(self.device)
        self.Move_out_tensor_Ammonia = torch.tensor(self.Ammonia_move_out.values, dtype=torch.float64).to(self.device)
        self.Move_in_tensor_Coef_Ammonia = torch.tensor(self.Ammonia_Coef_move_in.values, dtype=torch.float64).to(self.device)
        self.Move_out_tensor_Coef_Ammonia = torch.tensor(self.Ammonia_Coef_move_out.values, dtype=torch.float64).to(self.device)

        self.amounts = torch.tensor((self.Move_out_origin * self.mobility_ratio), dtype=torch.int64).to(self.device)
        
        self.action_mask_left = None
        self.f = {0:2, 1:1}

        self.reset()
    
    # @staticmethod
    # def ammonia_density_case(ammonia_density):
    #     return ammonia_density>=99999
    def update_state(self, move_in_idx, move_out_idx, amounts):

        self.state['Amount_Move_in'][move_in_idx, :] += amounts.cpu().numpy()
        self.state['Amount_Move_out'][move_out_idx, :] -= amounts.cpu().numpy()
        self.state['N_demand_Move_in'][move_in_idx] += (amounts.double() @ self.Move_in_tensor_Coef_N_demand[move_in_idx, :]).item()
        self.state['N_demand_Move_out'][move_out_idx] -= (amounts.double() @ self.Move_out_tensor_Coef_N_demand[move_out_idx, :]).item()
        self.state['Ammonia_Move_in'][move_in_idx] -= (amounts.double() @ self.Move_in_tensor_Coef_Ammonia[move_in_idx, :]).item()
        self.state['Ammonia_Move_out'][move_out_idx] -= (amounts.double() @ self.Move_out_tensor_Coef_Ammonia[move_out_idx, :]).item()
        return self.state
    
    def detect_violation(self, move_out_idx, move_in_idx):
        
        # 移入和移出地区的牲畜数量是否小于0
        Amounts_violation = (self.state['Amount_Move_out'][move_out_idx] < 0).any() or (self.state['Amount_Move_in'][move_in_idx] < 0).any()
        return Amounts_violation
    
    def get_total_reward(self, move_in_index, move_out_index):
        """计算移入和移出县的奖励，并确保全局达标。"""
        # 奖励初始化    
        reward = 0
        # 移入县的奖励
        reward += self._evaluate_county(move_in_index, is_move_in=True)
        # # 移出县的奖励
        # reward += self._evaluate_county(move_out_index, is_move_in=False)

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
        
        # delta NH3 / NH3_threshold
        Ammonia_reward = self.get_reward((self.move_amount.double() @ self.Move_in_tensor_Coef_Ammonia[idx, :]).item(),
                                        self.thresholds[0],
                                        self.Ammonia_move_in_origin[idx],
                                        self.reward_func)
                
        if is_move_in:
            for i, _ in enumerate([Ammonia_reward, sensitivity, pm25relative]):
                reward += _ * self.reward_priority[i]
        else:
            for i, _ in enumerate([Ammonia_reward]):
                reward += _ * self.reward_priority[i]

        return reward
    def evaluate_sensitivity_and_pm25relative(self, move_in_idx):
        # 根据移入地区的承载力情况计算奖励
        sensitivity = self.state['sensitivity_Move_in'][move_in_idx]
        relative_pm25 = self.state['relative_pm25_Move_in'][move_in_idx]
        # 计算反向的敏感度和相关性值
        inverse_sensitivity = self.f[sensitivity[0]]
        inverse_relative_pm25 = self.f[relative_pm25[0]]
        return inverse_sensitivity, inverse_relative_pm25
    
    def get_reward(self, delta, threshold, sigma, func=None):
        return func(torch.tensor(delta), threshold, sigma)
    

    @staticmethod
    def reward_func(x, x0, sigma):
        return -((x - x0) / (sigma + 1e-5) + 1)
    
    def step(self, action):
        move_out_idx = action // self.Move_in.shape[0]
        move_in_idx = action % self.Move_in.shape[0]
        
        amounts = self.move_amount

        if amounts is None:
            raise ValueError("Amounts is None")

        reward = self.get_total_reward(move_in_idx, move_out_idx)
                    
        self.state = self.update_state(move_in_idx, move_out_idx, amounts)
        self.current_step += 1

        # 检测约束满足
        # amounts_violation = self.detect_violation(move_out_idx, move_in_idx)
        
        # final_terminated = self.check_termination_condition()
        terminated = self.action_mask_left==0 or self.current_step >= self.max_steps     
        truncated = False

        # if amounts_violation:
        #     reward -= -10
        #     truncated = True    

        info = {
            # 'Amounts_violation': amounts_violation,
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
        self.N_demand_move_in = self.N_demand_move_in_origin.copy()
        self.N_demand_move_out = self.N_demand_move_out_origin.copy()
        self.Ammonia_move_out = self.Ammonia_move_out_origin.copy()
        self.Ammonia_move_in = self.Ammonia_move_in_origin.copy()

        self.state = {
            'Amount_Move_in': self.Move_in,
            'Amount_Move_out': self.Move_out,
            'N_demand_Move_in': self.N_demand_move_in.reshape(-1, 1),
            'N_demand_Move_out': self.N_demand_move_out.reshape(-1, 1),
            'Ammonia_Move_in': self.Ammonia_move_in.reshape(-1, 1),
            'Ammonia_Move_out': self.Ammonia_move_out.reshape(-1, 1),
            'sensitivity_Move_in': self.sensitivity_move_in.values.reshape(-1, 1),
            'sensitivity_Move_out': self.sensitivity_move_out.values.reshape(-1, 1),
            'relative_pm25_Move_in': self.relative_pm25_move_in.values.reshape(-1, 1),
            'relative_pm25_Move_out': self.relative_pm25_move_out.values.reshape(-1, 1),
        }

        # 直接返回 state 作为 observation
        return self.state, {}

    def render(self, mode='human'):
        # Render environment (optional)
        # print(f"step:{env.current_step}, || Action: {action}, Reward: {reward}, terminated: {terminated}, truncated:{truncated},info:{info}")
        pass

if __name__ == "__main__":
    # for country in ['aus', 'br', 'cn', 'usa', 'us']:
    
    for country in ['eu']:
        config = LivestockEnvConfig(country, 
                                    Reward_priority=[4, 2, 1], 
                                    thresholds=[0], 
                                    mobility_ratio=0.1,
                                    df_path='欧盟更新PB后第一步.xlsx')
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