# Livestocks-Agent

[English](README.md) | [中文](README_CN.md)

Livestocks-Agent implements constrained reinforcement learning for livestock spatial reallocation. A masked PPO policy selects source–destination pairs, and an integer allocator determines species-level transfer quantities subject to livestock inventory, nitrogen, and ammonia constraints.

## Paper method

At each step, the policy observes livestock inventories, nitrogen and ammonia residuals, sensitivity, and PM2.5-correlation indicators. The action mask removes infeasible region pairs. The selected pair is passed to the quantity allocator, which returns an integer transfer vector. The reward combines NH3, sensitivity, and PM2.5-correlation components with weights ordered as `[4, 2, 1]`.

Main code:

- `livestock_rl/livestockEnvV2.py`: environment and constrained quantity allocation
- `livestock_rl/AttentionPolicy.py`: attention policy
- `stable-baselines3/`: masked PPO implementation
- `data_preprocessing_code/`: data preprocessing
- `sensitivity/`: sensitivity analysis

## Environment

Requirements:

- Python 3.11
- uv
- PyTorch 2.3.1
- CUDA 12.1 for GPU training

```bash
git clone https://github.com/ResearchGroupHdZhang/Livestocks-Agent.git
cd Livestocks-Agent
git switch new
uv sync
```

The input Excel workbook contains `移出` and `移入` sheets. It provides region IDs, livestock inventories, nitrogen and ammonia values and coefficients, sensitivity, and PM2.5-correlation indicators.

## Training

Country-level training:

```bash
uv run python -m livestock_rl.train_ppo_v2
```

Province-level training:

```bash
uv run python -m livestock_rl.train_ppo_v2_province
```

Training configurations are stored under `configs/train/`, with separate national and province files for Australia, Brazil, China, the European Union, and the United States.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
