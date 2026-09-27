# Livestocks-Agent

[English](README.md) | [中文](README_CN.md)

Livestocks-Agent is a domain-constrained reinforcement-learning implementation for livestock spatial reallocation. A PPO policy selects a source–destination region pair, an action mask removes inadmissible pairs, and a constrained integer quantity allocator determines the species-specific transfer amounts.

This branch contains the reusable model and data-processing code. Internal `experiments/`, optimization `baselines/`, private research notes, checkpoints, logs, and generated results are intentionally excluded.

## Method overview

At each environment step:

1. The observation contains livestock inventories, nitrogen-demand residuals, ammonia capacities, and two destination-side social/environmental indicators.
2. The policy selects one discrete source–destination pair from `n_source × n_destination` actions.
3. `True` entries in the action mask mean that an action is invalid.
4. The quantity allocator proposes an integer transfer vector across livestock species and checks inventory, destination species support, nitrogen, and ammonia constraints.
5. The environment updates source and destination state and returns a scalar reward.

The three reward weights are ordered exactly as:

```text
[NH3 component, sensitivity component, PM2.5-correlation component]
```

`LivestockEnvConfig` requires exactly three values. The default configuration used by the provided training examples is `[4, 2, 1]`; extra values are rejected instead of being silently ignored.

## Correctness fixes in this branch

The public implementation includes the following fixes:

- The action mask used during sampling is stored with every rollout transition and reused during PPO updates.
- Per-sample masks remain aligned with observations and actions after minibatch shuffling.
- An all-invalid mask raises an explicit error instead of becoming a uniform categorical distribution.
- Transformer layers use `[batch, token, feature]` layout with `batch_first=True`; minibatch samples no longer attend to one another.
- Transformer dropout is disabled so unchanged parameters reproduce the same action probabilities between rollout and update modes.
- Failed or invalid quantity solves return zero transfer, and the environment rejects infeasible transfers before state mutation.
- Training and evaluation use separate environment instances.
- Signed nitrogen and ammonia residuals are represented by observation spaces that allow negative values.

These fixes make new training runs internally consistent. They do not retroactively repair checkpoints trained by an older implementation.

## Repository layout

```text
livestock_rl/                  Core environment, policy, training and evaluation code
stable-baselines3/             Project-specific Stable-Baselines3 fork with masked PPO
data_preprocessing_code/       Data-preparation notebooks and scripts
sensitivity/                   Social-sensitivity analysis code and supporting data
pyproject.toml                 Python dependencies
uv.lock                        Locked environment
```

The current public branch does not include:

```text
experiments/
baselines/
knowledge/
checkpoints, logs, runtime ledgers, or generated result packages
```

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- PyTorch 2.3.1
- Gymnasium 0.29.1
- NumPy 1.24.4
- SciPy 1.11 or later

The lock file uses the PyTorch CUDA 12.1 package index. A compatible NVIDIA driver is required for GPU training. The code can use CPU for small validation runs, but full instances may be expensive.

## Installation

```bash
git clone https://github.com/ResearchGroupHdZhang/Livestocks-Agent.git
cd Livestocks-Agent
git switch new
uv sync
```

Verify the core imports:

```bash
uv run python -c "import livestock_rl, stable_baselines3; print('imports ok')"
```

## Input workbook contract

`LivestockEnv` reads an Excel workbook with two sheets:

- `移出`: source regions;
- `移入`: destination regions.

The loader expects the following fields or column-name markers:

| Meaning | Required name/marker |
|---|---|
| Region identity | `ID`, `city`, `county` |
| Optional subregional filter | `province` |
| Species inventories | columns containing `num` |
| Nitrogen-demand residual | column containing `最优氮需求` |
| Per-animal manure-N coefficient | columns containing `manure变化量` |
| Ammonia capacity/residual | column containing `ammonia指标` |
| Per-animal ammonia coefficient | columns containing `氨变化量` |
| Sensitivity indicator | column containing `敏感度` |
| PM2.5-correlation indicator | column containing `PM2.5相关性` |

Inventory columns and both coefficient groups must use the same species order. The legacy loader selects columns by marker and position; it does not safely infer arbitrary species mappings. Validate the schema and units before training.

The current loader also converts missing workbook cells to zero. Missing scientific measurements should therefore be resolved or explicitly encoded before the workbook reaches the loader; an unknown value must not be treated as a physical zero without justification.

Paper-production input workbooks and trained checkpoints are not distributed in this branch.

## Environment configuration

```python
from pathlib import Path
from livestock_rl import LivestockEnv, LivestockEnvConfig

input_path = Path("/absolute/path/to/input.xlsx")
config = LivestockEnvConfig(
    country="eu",                 # eu, aus, cn, br, or usa
    Reward_priority=[4, 2, 1],   # NH3, sensitivity, PM2.5 correlation
    thresholds=[0, 0],           # N threshold, NH3 threshold
    df_path=str(input_path),
    province=None,                # set a province/state/group for a subregional run
    mobility_ratio=0.02,
    max_steps=50_000,
)

env = LivestockEnv(config)
observation, info = env.reset(seed=42)
assert env.observation_space.contains(observation)
print(env.action_space.n)
```

Use an absolute input path when possible. Relative paths are resolved under `data/<country-label>/` by the legacy loader.

## Training

The following example creates independent training and evaluation environments:

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

Before an expensive run, lock and record the input hash, scope, seed set, training budget, checkpoint-selection rule, and evaluation protocol. A checkpoint must be evaluated with the same region ordering, species ordering, observation/action dimensions, thresholds, and mobility semantics used during training.

The files `livestock_rl/train_ppo_v2.py` and `livestock_rl/train_ppo_v2_province.py` are reference entry points with project-specific defaults. Review their country, input path, GPU selection, output paths, and budget before running them.

## Evaluation

`livestock_rl/evaluation_ppo.py` is an import-safe reference entry point. Its data and checkpoint paths are placeholders for the original project layout and must be updated before execution.

For a frozen checkpoint, construct an environment with the exact training schema and then load the model:

```python
model = PPO_action_mask_v2.load("checkpoints/eu/best_model.zip", env=eval_env)
```

Report final physical outcomes and constraint residuals in addition to cumulative reward. Cumulative reward depends on step size and trajectory length and is not, by itself, a method-independent measure of final allocation quality.

## Verification

Run the action-mask regression tests:

```bash
uv run --with pytest python -m pytest -q stable-baselines3/tests/test_action_mask.py
```

Compile the public Python code:

```bash
uv run python -m compileall -q livestock_rl stable-baselines3/stable_baselines3
```

## Reproducibility boundaries

- This branch provides code, not a complete reproduction package for every paper result.
- Production workbooks, checkpoints, experiment ledgers, optimization baselines, and generated figures are not included.
- Changing action-mask semantics, Transformer layout, reward weights, species order, or physical constraints defines a different training condition.
- Checkpoints trained before the correctness fixes should not be described as models trained with the repaired implementation.
- Exact solver, heuristic, and learned-policy comparisons require identical inputs, admissibility rules, objective definitions, and independent physical validation.

## License and citation

No license or citation metadata is currently included in the repository. Add the appropriate project license and paper citation before redistributing the code or using it as a formal archival release.
