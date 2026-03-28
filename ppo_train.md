# PPO Training Setup (policy_reusability/data_generation/deeprl/train_ppo.py)

This document explains the full training setup implemented in `policy_reusability/data_generation/deeprl/train_ppo.py` and the LeverGrid environment used by it (`policy_reusability/env/lever_minigrid.py`). It covers the algorithm, model, optimizer, hyperparameters, environment, observation processing, rewards, evaluation, and outputs. It also summarizes how the multi-reward training mode works.

**Scope note**: Everything described here is derived from the training script and the LeverGrid environment implementation. No external assumptions are required except for the default SB3 optimizer (not overridden in the script).

**Algorithm and Model**
- **Algorithm**: Stable-Baselines3 `PPO` (on-policy actor-critic with clipped surrogate objective and GAE advantage estimation).
- **Policy**: `CnnPolicy` with a custom feature extractor `MinigridFeaturesExtractor`.
- **Optimizer**: SB3 default (Adam), because `optimizer_class` and `optimizer_kwargs` are not set in the script.
- **Exploration**: Controlled by entropy coefficient (`ent_coef`); no epsilon-greedy schedule. Higher `ent_coef` encourages more stochastic action selection during training.

**Feature Extractor Architecture (MinigridFeaturesExtractor)**
- Input: image-like observation tensor (channel-first for CNN).
- Convolution stack: up to three Conv2D layers with kernel size `2` and stride `1`, channel sizes 16 -> 32 -> 64. The number of conv layers used depends on spatial size; if the spatial size is too small (< 2), fewer layers are used.
- Flatten.
- MLP head: `Linear(n_flatten, features_dim)` + ReLU, then `Linear(features_dim, features_dim)` + ReLU.
- Default `features_dim`: 128 (configurable via `--features-dim`).

**Environment (LeverGridEnv)**
- Base environment: custom MiniGrid variant `LeverGridEnv`.
- Grid elements: walls (impassable), lava (hazard, terminates on contact), balls (gold, collectible), key (lever objective), exit (goal).
- Default max steps: `2 * size * size` (MiniGrid default).
- Action mode for training: **cardinal** (4 actions: right, down, left, up). Internally, each cardinal action sets the agent direction and performs a forward step.
- Reward system is configurable via `reward_system` tokens (see Rewards section).

**Observation Pipeline**
The training script wraps the base environment with observation-specific wrappers.

Observation mode selection (`--obs-mode`):
- `partial`: `ImgObsWrapper` (MiniGrid default 7x7 egocentric view) + `ScaleObsWrapper`.
- `full`: `FullyObsWrapper` + `ImgObsWrapper` + `ScaleObsWrapper`.
- `local`: `LocalObsWrapper` (top-down patch centered on agent) + `ScaleObsWrapper`.

`LocalObsWrapper` details:
- Produces channel-first `(3, size, size)` patch (object, color, state channels) centered on the agent.
- Pads out-of-bounds with wall encoding.
- **Sees through walls**: the patch is extracted directly from the full grid encoding (`grid.encode()`), with no line-of-sight or wall-occlusion logic. This is a deliberate design choice -- the trained policies are inputs to the downstream policy composition pipeline, so we prioritize training competent policies over realistic perception. The `partial` obs mode (MiniGrid default) does use wall occlusion if needed for comparison.

`ScaleObsWrapper` details:
- MiniGrid encodes (object_type, color, state) in small integer ranges.
- The wrapper scales each channel to use the full `[0, 255]` range so SB3's CNN preprocessing does not squash values toward zero.

Vectorized env handling:
- Uses `SubprocVecEnv` by default, falls back to `DummyVecEnv` if necessary.
- Each environment is wrapped with `Monitor` for episode logging.
- Uses `VecTransposeImage` when SB3 determines the observation space is not already channel-first.
- Uses `VecNormalize` with `norm_obs=False` and `norm_reward=True`, `clip_reward=10.0`, and `gamma` passed through. **Key difference from DQN**: PPO normalizes rewards to stabilise advantage estimates; DQN does not normalize rewards.

**Rewards and Reward Systems**
Reward systems are configured by string tokens separated by `-` (e.g., `path-gold-hazard-lever`). The training script defines the valid systems used by `--all-rewards`:

- `path`
- `gold`
- `hazard`
- `lever`
- `hazard-lever`
- `path-gold`
- `path-gold-hazard`
- `path-gold-hazard-lever`

Any token combination is parsed by the environment, but the script's `--all-rewards` cycles through the list above.
LeverGridEnv also treats `reward_system` values of `None`, empty string, `combined`, or `all` as meaning all components are active.

Reward parameter values (all configurable via CLI):
| Parameter | Default | CLI flag | Notes |
| --- | --- | --- | --- |
| `ball_reward` | 15.0 | `--ball-reward` | Reward for collecting a ball when `gold` is active. |
| `key_reward` | 20.0 | `--key-reward` | Reward for collecting the key when `lever` is active. |
| `exit_reward` | 50.0 (200.0 for `path`) | `--exit-reward` | Reward for reaching exit. When unset, defaults to 200 for `path`, 50 otherwise. |
| `exit_with_key_reward` | 100.0 | `--exit-with-key-reward` | Reward for exiting with key when `lever` is active. |
| `lava_penalty` | -1.0 | `--lava-penalty` | Penalty and termination when stepping on lava. |
| `step_penalty` | -0.05 | `--step-penalty` | Per-step penalty applied only when `path` is active. |
| `path_progress_scale` | 1.0 | `--path-progress-scale` | Scale for shaping rewards in path/gold/lever/hazard. |

Reward components and shaping (implemented in `LeverGridEnv.step`):

Path objective (`path`):
- **Step penalty**: applied every step only if `path` is active.
- **Exit reward**: `exit_reward` when reaching the goal.
- **Shaping**: BFS distance delta to exit multiplied by `path_progress_scale`.

Gold objective (`gold`):
- **Ball reward**: `ball_reward` on collecting a ball.
- **Exit reward**: `exit_reward` when reaching the goal.
- **Shaping**: BFS distance delta to nearest remaining ball; after all balls are collected, BFS distance delta to exit. Uses `path_progress_scale`.

Lever objective (`lever`):
- **Key reward**: `key_reward` on collecting the key.
- **Exit reward**: if `lever` is the only active objective and you reach the goal without a key, you still get `exit_reward`. If you reach the goal with a key, you get `exit_with_key_reward` when `path` is not active; if `path` is active, you get only the bonus `exit_with_key_reward - exit_reward` (so the path exit total becomes `exit_with_key_reward`).
- **Shaping**: BFS distance delta to key until the key is collected, then BFS distance delta to exit. Uses `path_progress_scale`.

Hazard objective (`hazard`):
- **Lava penalty**: `lava_penalty` and episode termination when stepping on lava.
- **Exit reward**: `exit_reward` when reaching the goal.
- **Shaping**: Manhattan distance progress to goal (`prev_dist - curr_dist`), additional scaled progress fraction toward the goal (`path_progress_scale * progress`), and hazard proximity shaping `2.0 * (prev_hazard_neighbors - curr_hazard_neighbors)` to encourage moving away from adjacent lava.

Important interaction detail for combined rewards:
- When the goal is reached, `exit_reward` is added once for each active component among `path`, `gold`, and `hazard`. This means combined reward systems can grant multiple `exit_reward` additions. The lever bonus is added on top as described above.

**Training Loop and Evaluation**
- The training loop is controlled by `train_ppo(...)`.
- Evaluation seeds are generated once and stored in `eval_env_seeds.json` per size. They are reused across runs for consistent evaluation.
- Callback: `EpisodeSnapshotCallback`.
- Callback behavior: saves models periodically by **episode count** (`--snapshot-interval`) and optionally by **timesteps** (`--snapshot-steps`), runs evaluation on fixed seeds twice (stochastic and deterministic), writes metrics to `episode_rewards.csv`, and saves evaluation transitions to `eval_transitions.pkl` for the stochastic evaluation (state, next_state, action, etc.).

Logging and artifacts:
- SB3 logger configured for `stdout`, `csv`, and `tensorboard` under `logs/`.
- Optional loss plot (`--loss-plot`) generated from `progress.csv` and saved as `loss_plot.png`. PPO loss plot includes `policy_loss`, `value_loss`, and `entropy_loss`.
- MiniGrid ID mappings are written once at startup to `output_root/minigrid_ids.json` (unless `--minigrid-ids-json` is set to an empty string).
- Final model saved as `final_model` (SB3 saves a `.zip` file at that path).
- VecNormalize stats saved to `vecnormalize.pkl` (important for PPO since reward normalization is enabled).

**Default Hyperparameters (CLI)**
These defaults are exactly those defined in `train_ppo.py` and used unless overridden on the command line.

Training and evaluation:
| Parameter | Default | Description |
| --- | --- | --- |
| `--timesteps` | 1_000_000 | Total training timesteps. |
| `--snapshot-interval` | 100 | Snapshot/eval interval in episodes. |
| `--snapshot-steps` | None | Snapshot/eval interval in timesteps. |
| `--n-envs` | 8 | Number of parallel training envs. |
| `--n-eval-envs` | 100 | Number of evaluation seeds/envs. |
| `--seed` | None | Random seed; also used to derive eval seeds. |

Environment:
| Parameter | Default | Description |
| --- | --- | --- |
| `--size` | 16 | Grid size (size x size). |
| `--num-balls` | 24 | Number of balls in the grid. |
| `--num-walls` | 16 | Number of walls. |
| `--num-lava` | 16 | Number of lava tiles. |
| `--grid-preset` | None | Preset object counts: `grid8` or `grid16_scaled`. |

PPO hyperparameters:
| Parameter | Default | Description |
| --- | --- | --- |
| `--lr` | 2.5e-4 | Learning rate. |
| `--n-steps` | None | Steps per env per update (default: `2 * size^2`). |
| `--batch-size` | None | Minibatch size (default: `n_steps / 2`). |
| `--n-epochs` | 4 | Number of epochs per PPO update. |
| `--gamma` | 0.99 | Discount factor. |
| `--ent-coef` | 0.01 | Entropy coefficient (encourages exploration). |
| `--features-dim` | 128 | Feature dimension for CNN extractor. |

Note: PPO also uses `gae_lambda=0.95` and `clip_range=0.2` internally (not exposed as CLI flags but hardcoded in `train_ppo()`).

Reward shaping:
| Parameter | Default | Description |
| --- | --- | --- |
| `--step-penalty` | -0.05 | Per-step penalty (applies only when `path` is active). |
| `--path-progress-scale` | 1.0 | Scale for shaping rewards (BFS/goal progress). |

Observation:
| Parameter | Default | Description |
| --- | --- | --- |
| `--obs-mode` | `partial` | Observation mode: `partial`, `local`, `full`. |
| `--local-size` | 7 | Patch size for `local` mode (odd). |

Saving and logging:
| Parameter | Default | Description |
| --- | --- | --- |
| `--output-root` | `deeprl_runs` | Root directory for training outputs. |
| `--minigrid-ids-json` | `minigrid_ids.json` | Writes MiniGrid ID mappings under output root. |
| `--overwrite` | False | Allow training into non-empty output dirs. |
| `--tensorboard` | None | Optional TensorBoard root. |
| `--verbose` | 1 | Verbosity level. |
| `--loss-plot` | False | Save a loss plot after training. |

Reward system selection:
| Parameter | Default | Description |
| --- | --- | --- |
| `--reward-system` | `path-gold-hazard-lever` | Reward system string. |
| `--all-rewards` | False | Train all reward systems in the script list. |
| `--sizes` | None | Comma-separated sizes (e.g., `16,32`). |

Evaluation mode:
| Parameter | Default | Description |
| --- | --- | --- |
| `--eval` | None | If set, run evaluation instead of training. |
| `--eval-episodes` | 10 | Number of eval episodes. |

**Grid Presets**
Grid presets override `--size`, `--num-balls`, `--num-walls`, and `--num-lava` and clear `--sizes`.

| Preset | Size | Balls | Walls | Lava |
| --- | --- | --- | --- | --- |
| `grid8` | 8 | 6 | 4 | 4 |
| `grid16_scaled` | 16 | 24 | 16 | 16 |

**Output Layout**
For each size and reward system, the script writes to:
- `output_root/{size}/{reward_system}/`
- `episodes/episode_XXXXXX/` for periodic snapshots (each contains `model` and `eval_transitions.pkl`).
- `logs/` for `progress.csv` and TensorBoard logs.
- `episode_rewards.csv` for evaluation metrics.
- `final_model` and `vecnormalize.pkl` after training ends.

**Special Case: `path` Reward System**
In `main()`, when training with `reward_system == "path"`, the script forces `exit_reward=200.0` instead of the default 50.0. This is done only in training (and in `evaluate_model` if `reward_system == "path"` and `exit_reward` was left at the default).

**Key Differences from DQN Training**

| Aspect | DQN (`train_dqn.py`) | PPO (`train_ppo.py`) |
| --- | --- | --- |
| Algorithm | Off-policy, value-based (Q-learning) | On-policy, actor-critic |
| Exploration | Epsilon-greedy with linear annealing | Entropy bonus (`ent_coef`) |
| Replay buffer | Yes (`buffer_size=1_000_000`) | No (on-policy, rollout buffer) |
| Target network | Yes (`tau`, `target_update_interval`) | No |
| Reward normalization | No (`norm_reward=False`) | Yes (`VecNormalize`, `norm_reward=True`, `clip_reward=10.0`) |
| `n_steps` default | 1 (single-step Q-learning) | `2 * size^2` (multi-step rollouts) |
| `batch_size` default | 32 | `n_steps / 2` |
| Learning rate default | 1e-4 | 2.5e-4 |
| Additional hyperparams | `learning_starts`, `exploration_fraction`, `exploration_final_eps`, `train_freq`, `gradient_steps`, `target_update_interval`, `tau`, `max_grad_norm` | `n_epochs`, `gae_lambda`, `clip_range`, `ent_coef` |
| Loss curves | `train/loss` (TD error) | `policy_loss`, `value_loss`, `entropy_loss` |
| Sample efficiency | Lower (needs more timesteps) | Higher (fewer timesteps typically needed) |
