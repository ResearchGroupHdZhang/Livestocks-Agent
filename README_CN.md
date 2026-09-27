# Livestocks-Agent

[English](README.md) | [中文](README_CN.md)

Livestocks-Agent 是一个用于畜禽空间重分配的约束强化学习项目。带 action mask 的 PPO 策略选择迁出地—迁入地组合，整数数量分配模块在畜禽库存、氮和氨约束下确定各畜种迁移量。

## 论文方法

每一步中，策略根据畜禽存量、氮和氨残差、敏感度及 PM2.5 相关性指标选择地区对。Action mask 排除不可行地区对，数量分配模块为选定地区对生成整数迁移向量。Reward 由 NH3、敏感度和 PM2.5 相关性三个分量组成，权重顺序为 `[4, 2, 1]`。

主要代码：

- `livestock_rl/livestockEnvV2.py`：环境与约束数量分配
- `livestock_rl/AttentionPolicy.py`：attention 策略
- `stable-baselines3/`：带 action mask 的 PPO
- `data_preprocessing_code/`：数据预处理
- `sensitivity/`：敏感度分析

## 环境

环境要求：

- Python 3.11
- uv
- PyTorch 2.3.1
- GPU 训练使用 CUDA 12.1

```bash
git clone https://github.com/ResearchGroupHdZhang/Livestocks-Agent.git
cd Livestocks-Agent
git switch new
uv sync
```

输入 Excel 包含 `移出` 和 `移入` 两个 sheet，提供地区 ID、畜禽存量、氮和氨指标及系数、敏感度和 PM2.5 相关性指标。

## 训练

国家级训练：

```bash
uv run python -m livestock_rl.train_ppo_v2
```

省级训练：

```bash
uv run python -m livestock_rl.train_ppo_v2_province
```

`configs/train/` 下分别提供澳大利亚、巴西、中国、欧盟和美国的国家级及省级训练配置。

## 许可

本项目使用 MIT License，详见 [LICENSE](LICENSE)。
