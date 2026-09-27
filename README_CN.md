# Livestocks-Agent

[English](README.md) | [中文](README_CN.md)

Livestocks-Agent 是一个用于畜禽空间重分配的领域约束强化学习实现。PPO 策略负责选择迁出地区—迁入地区组合，action mask 排除不可行动作，带整数约束的数量分配模块再确定各畜种的迁移数量。

当前分支仅保留可复用的模型与数据处理代码。内部 `experiments/`、优化 `baselines/`、私人研究笔记、checkpoint、日志和生成结果均不在本公开分支中。

## 方法概览

环境每一步执行以下流程：

1. 观测包括畜禽存量、氮需求残差、氨容量，以及迁入地区的两个社会/环境指标。
2. 策略从 `迁出地区数 × 迁入地区数` 个离散动作中选择一个地区对。
3. action mask 中 `True` 表示该动作不可用。
4. 数量分配模块生成各畜种的整数迁移向量，并检查库存、迁入地物种支持、氮约束和氨约束。
5. 环境更新迁出地和迁入地状态，并返回标量 reward。

三个 reward 权重的固定顺序是：

```text
[NH3 分量，敏感度分量，PM2.5 相关性分量]
```

`LivestockEnvConfig` 强制要求恰好三个权重。示例训练配置使用 `[4, 2, 1]`；多余权重会直接报错，不会再被静默忽略。

## 当前分支包含的正确性修复

- 采样时使用的 action mask 会随每条 rollout transition 保存，并在 PPO 更新时原样复用。
- minibatch 打乱后，逐样本 mask 仍与 observation 和 action 对齐。
- 全部动作无效时显式报错，不再退化成均匀分类分布。
- Transformer 使用 `[batch, token, feature]` 布局并设置 `batch_first=True`，不同 minibatch 样本不再互相做 attention。
- Transformer dropout 已关闭，使参数不变时 rollout 和 update 模式能够复算相同动作概率。
- 数量求解失败或结果无效时返回零迁移；环境会在修改状态前拒绝不可行迁移。
- 训练环境与评估环境相互独立。
- 氮和氨残差的 observation space 允许负值，与环境实际返回值一致。

这些修复保证的是新训练流程的一致性，不能追溯修复由旧实现训练出的 checkpoint。

## 目录结构

```text
livestock_rl/                  环境、策略、训练与评估代码
stable-baselines3/             项目定制的 Stable-Baselines3 与 masked PPO
data_preprocessing_code/       数据预处理 notebook 和脚本
sensitivity/                   社会敏感度分析代码及配套数据
pyproject.toml                 Python 依赖
uv.lock                        锁定环境
```

当前公开分支明确不包含：

```text
experiments/
baselines/
knowledge/
checkpoint、日志、运行状态和生成结果包
```

## 环境要求

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- PyTorch 2.3.1
- Gymnasium 0.29.1
- NumPy 1.24.4
- SciPy 1.11 或更高版本

锁文件使用 PyTorch CUDA 12.1 软件源。GPU 训练需要兼容的 NVIDIA 驱动；小规模检查可以使用 CPU，但完整实例可能计算开销较大。

## 安装

```bash
git clone https://github.com/ResearchGroupHdZhang/Livestocks-Agent.git
cd Livestocks-Agent
git switch new
uv sync
```

验证核心包导入：

```bash
uv run python -c "import livestock_rl, stable_baselines3; print('imports ok')"
```

## 输入工作簿约定

`LivestockEnv` 读取一个包含两个 sheet 的 Excel 工作簿：

- `移出`：迁出地区；
- `移入`：迁入地区。

加载器依赖以下字段或列名标记：

| 含义 | 必需名称/标记 |
|---|---|
| 地区标识 | `ID`、`city`、`county` |
| 可选的省级筛选 | `province` |
| 各畜种存量 | 列名包含 `num` |
| 氮需求残差 | 列名包含 `最优氮需求` |
| 单头畜禽 manure-N 系数 | 列名包含 `manure变化量` |
| 氨容量/残差 | 列名包含 `ammonia指标` |
| 单头畜禽氨系数 | 列名包含 `氨变化量` |
| 敏感度指标 | 列名包含 `敏感度` |
| PM2.5 相关性指标 | 列名包含 `PM2.5相关性` |

存量列、manure-N 系数列和氨系数列必须采用相同的畜种顺序。旧加载器按列名标记和位置提取数据，不能安全推断任意物种映射；训练前必须先核对 schema、单位和畜种顺序。

当前加载器还会将工作簿中的缺失单元格转成零。因此，科学数据中的未知值必须在进入加载器之前完成核实或显式编码，不能在没有依据时把“未知”当作物理零值。

本分支不分发论文正式运行使用的输入工作簿和训练 checkpoint。

## 环境配置

```python
from pathlib import Path
from livestock_rl import LivestockEnv, LivestockEnvConfig

input_path = Path("/absolute/path/to/input.xlsx")
config = LivestockEnvConfig(
    country="eu",                 # eu、aus、cn、br 或 usa
    Reward_priority=[4, 2, 1],   # NH3、敏感度、PM2.5 相关性
    thresholds=[0, 0],           # N 阈值、NH3 阈值
    df_path=str(input_path),
    province=None,                # 省/州/分组实验时填写
    mobility_ratio=0.02,
    max_steps=50_000,
)

env = LivestockEnv(config)
observation, info = env.reset(seed=42)
assert env.observation_space.contains(observation)
print(env.action_space.n)
```

建议使用绝对输入路径。使用相对路径时，旧加载器会在 `data/<国家中文名>/` 下解析文件。

## 训练

下面的示例会创建彼此独立的训练环境和评估环境：

```python
from pathlib import Path
from stable_baselines3 import PPO_action_mask_v2
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan
from livestock_rl import LivestockEnv, LivestockEnvConfig
from livestock_rl.AttentionPolicy import CustomAttentionPolicy

input_path = Path("/absolute/path/to/input.xlsx")
config = LivestockEnvConfig(
    "eu", [4, 2, 1], [0, 0], str(input_path),
    province=None, mobility_ratio=0.02, max_steps=50_000,
)

train_env = VecCheckNan(
    make_vec_env(lambda: LivestockEnv(config), n_envs=1, seed=42),
    raise_exception=True,
)
eval_env = VecCheckNan(
    make_vec_env(lambda: LivestockEnv(config), n_envs=1, seed=1042),
    raise_exception=True,
)

callback = EvalCallback(
    eval_env,
    best_model_save_path="checkpoints/eu",
    log_path="logs/eu",
    eval_freq=32_769,
    deterministic=False,
)

model = PPO_action_mask_v2(
    CustomAttentionPolicy,
    train_env,
    n_steps=32_768,
    batch_size=256,
    learning_rate=2e-5,
    seed=42,
    tensorboard_log="board/eu",
    verbose=1,
)
model.learn(total_timesteps=200_000, callback=callback)
model.save("checkpoints/eu/final_model")
```

昂贵训练开始前，应固定并记录输入哈希、国家/省级范围、seed 集合、训练预算、checkpoint 选择规则和评估协议。加载 checkpoint 时，地区顺序、畜种顺序、观测/动作维度、阈值和 mobility 语义必须与训练时一致。

`livestock_rl/train_ppo_v2.py` 和 `livestock_rl/train_ppo_v2_province.py` 是带项目默认值的参考入口。运行前必须检查国家、输入路径、GPU、输出路径和训练预算。

## 评估

`livestock_rl/evaluation_ppo.py` 是可安全导入的参考入口。其中数据和 checkpoint 路径仍是原项目布局的占位值，运行前必须修改。

冻结 checkpoint 应使用与训练完全一致的 schema 构造环境，再加载模型：

```python
model = PPO_action_mask_v2.load("checkpoints/eu/best_model.zip", env=eval_env)
```

结果报告应同时给出最终物理指标和约束残差，而不应只比较累计 reward。累计 reward 会受到步长和轨迹长度影响，本身不是与方法无关的最终方案质量指标。

## 验证

运行 action-mask 回归测试：

```bash
uv run --with pytest python -m pytest -q stable-baselines3/tests/test_action_mask.py
```

编译公开 Python 代码：

```bash
uv run python -m compileall -q livestock_rl stable-baselines3/stable_baselines3
```

## 复现边界

- 当前分支提供代码，不是全部论文结果的一键复现包。
- 正式输入工作簿、checkpoint、实验运行账本、优化基线和生成图表不在此分支中。
- 修改 action mask、Transformer 布局、reward 权重、畜种顺序或物理约束，都构成新的训练条件。
- 修复前训练的 checkpoint 不能表述为使用当前修复实现训练所得。
- 精确求解器、启发式方法和学习策略之间的比较，必须采用相同输入、可行性规则、目标定义和独立物理验证。

## 许可证与引用

仓库目前没有提供许可证或引用元数据。在重新分发代码或作为正式归档版本使用之前，应补充适用的项目许可证和论文引用。
