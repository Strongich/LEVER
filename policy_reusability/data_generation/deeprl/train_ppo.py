"""
PPO Training for LeverGrid Environment.

Based on:
- MiniGrid Training Guide: https://minigrid.farama.org/content/training/
- StableBaselines3 Docs: https://stable-baselines3.readthedocs.io/en/master/

Usage:
    python -m policy_reusability.data_generation.deeprl.train_ppo --help
    python -m policy_reusability.data_generation.deeprl.train_ppo --timesteps 1000000
    python -m policy_reusability.data_generation.deeprl.train_ppo --all-rewards --sizes 16,32
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecTransposeImage,
)

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from stable_baselines3.common.vec_env.vec_transpose import is_image_space_channels_first

# Import our custom environment (registers it with gymnasium)
import policy_reusability.env.lever_minigrid  # noqa: F401
from policy_reusability.env.lever_minigrid import (
    LeverGridEnv,
    LocalObsWrapper,
    ScaleObsWrapper,
)

REWARD_SYSTEMS = [
    "path",
    "gold",
    "hazard",
    "lever",
    "hazard-lever",
    "path-gold",
    "path-gold-hazard",
    "path-gold-hazard-lever",
]

GRID_PRESETS = {
    "grid8": {"size": 8, "num_balls": 6, "num_walls": 4, "num_lava": 4},
    "grid16_scaled": {"size": 16, "num_balls": 24, "num_walls": 16, "num_lava": 16},
}


def parse_sizes(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(token) for token in value.split(",") if token.strip()]


def write_minigrid_ids_json(path: Path) -> None:
    """Write MiniGrid object/color/state ID mappings to JSON."""
    data = {
        "object_to_idx": OBJECT_TO_IDX,
        "color_to_idx": COLOR_TO_IDX,
        "state_to_idx": STATE_TO_IDX,
        "aliases": {
            "gold": OBJECT_TO_IDX.get("ball"),
            "ball": OBJECT_TO_IDX.get("ball"),
            "hazard": OBJECT_TO_IDX.get("lava"),
            "lava": OBJECT_TO_IDX.get("lava"),
            "wall": OBJECT_TO_IDX.get("wall"),
            "lever": OBJECT_TO_IDX.get("key"),
            "key": OBJECT_TO_IDX.get("key"),
            "exit": OBJECT_TO_IDX.get("goal"),
            "goal": OBJECT_TO_IDX.get("goal"),
            "agent": OBJECT_TO_IDX.get("agent"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    print(f"Wrote MiniGrid ID mapping to {path}")


def _plot_loss_curves(progress_csv: Path, output_path: Path) -> None:
    """Plot PPO loss curves from SB3 progress.csv if available."""
    if not progress_csv.exists():
        print(f"Warning: no progress.csv found at {progress_csv}; skipping loss plot.")
        return

    import csv as _csv

    rows = []
    with progress_csv.open(newline="") as handle:
        reader = _csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"Warning: progress.csv is empty at {progress_csv}; skipping loss plot.")
        return

    def _get_series(key: str):
        series = []
        for r in rows:
            val = r.get(key)
            if val in (None, ""):
                continue
            try:
                series.append(float(val))
            except ValueError:
                continue
        return series

    timesteps = _get_series("time/total_timesteps") or list(range(len(rows)))
    policy_loss = _get_series("train/policy_loss")
    value_loss = _get_series("train/value_loss")
    entropy_loss = _get_series("train/entropy_loss")

    if not (policy_loss or value_loss or entropy_loss):
        print("Warning: no loss metrics found in progress.csv; skipping loss plot.")
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 4.0))
    if policy_loss:
        plt.plot(timesteps[: len(policy_loss)], policy_loss, label="policy_loss")
    if value_loss:
        plt.plot(timesteps[: len(value_loss)], value_loss, label="value_loss")
    if entropy_loss:
        plt.plot(timesteps[: len(entropy_loss)], entropy_loss, label="entropy_loss")

    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.title("PPO Loss Curves")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved loss plot to {output_path}")


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for MiniGrid environments.

    Architecture from Lucas Willems' rl-starter-files, as recommended
    in the MiniGrid documentation.

    The default SB3 CNN doesn't work well with MiniGrid's small 7x7x3
    observation space, so we use smaller kernels.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        spatial_size = observation_space.shape[1]  # height (== width for square obs)

        # Build conv layers dynamically so the extractor works for any
        # spatial size >= 2 (e.g. 3x3 local, 7x7 partial, 8x8 full).
        channels = [16, 32, 64]
        layers: list[nn.Module] = []
        in_ch = n_input_channels
        for ch in channels:
            if spatial_size < 2:
                break
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=2, stride=1))
            layers.append(nn.ReLU())
            spatial_size -= 1  # kernel_size=2, stride=1 => size - 1
            in_ch = ch
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Compute the flattened size by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def load_or_create_eval_seeds(
    seeds_path: str,
    n_eval_envs: int,
    seed: int | None,
) -> list[int]:
    """Load eval seeds from disk or create and persist them if missing."""
    path = Path(seeds_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            seeds = json.load(handle)
        if len(seeds) != n_eval_envs:
            print(
                "Eval seeds length mismatch; regenerating to match "
                f"{n_eval_envs} envs."
            )
        else:
            return [int(s) for s in seeds]

    rng = random.Random(seed)
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(n_eval_envs)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(seeds, handle, indent=2)
    return seeds


def _maybe_transpose_obs(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 3 and obs.shape[-1] in (1, 3) and obs.shape[0] not in (1, 3):
        return obs.transpose(2, 0, 1)
    return obs


def _wrap_obs(env: gym.Env, obs_mode: str, local_size: int = 3) -> gym.Env:
    """Apply the appropriate observation wrapper based on *obs_mode*.

    - ``"partial"`` (default): 7x7 egocentric forward cone (ImgObsWrapper).
    - ``"local"``: direction-independent top-down patch (LocalObsWrapper).
      *local_size* controls the patch side length (default 3).
    - ``"full"``: full grid observation (FullyObsWrapper + ImgObsWrapper).

    All modes are followed by ``ScaleObsWrapper`` which rescales the raw
    MiniGrid encoding (values 0-10) to span the full [0, 255] uint8 range.
    Without this, SB3's /255 preprocessing squishes everything to near-zero.
    """
    if obs_mode == "local":
        env = LocalObsWrapper(env, size=local_size)
    elif obs_mode == "full":
        env = ImgObsWrapper(FullyObsWrapper(env))
    else:  # "partial"
        env = ImgObsWrapper(env)
    return ScaleObsWrapper(env)


def evaluate_on_seeds(
    model: PPO,
    seeds: list[int],
    size: int,
    num_balls: int,
    num_walls: int,
    num_lava: int,
    reward_system: str,
    ball_reward: float,
    key_reward: float,
    exit_reward: float,
    exit_with_key_reward: float,
    lava_penalty: float,
    step_penalty: float,
    path_progress_scale: float,
    deterministic_eval: bool = True,
    transitions_path: str | None = None,
    obs_mode: str = "partial",
    local_size: int = 3,
) -> tuple[float, float, float, float, float]:
    """Evaluate the model for one episode per seed and return metrics."""
    episode_rewards = []
    episode_lengths = []
    successes = []
    keys_collected = []
    balls_collected = []
    transitions = [] if transitions_path else None

    for seed in seeds:
        env = LeverGridEnv(
            size=size,
            num_balls=num_balls,
            num_walls=num_walls,
            num_lava=num_lava,
            reward_system=reward_system,
            ball_reward=ball_reward,
            key_reward=key_reward,
            exit_reward=exit_reward,
            exit_with_key_reward=exit_with_key_reward,
            lava_penalty=lava_penalty,
            step_penalty=step_penalty,
            path_progress_scale=path_progress_scale,
            action_mode="cardinal",
            render_mode="rgb_array",
        )
        env = _wrap_obs(env, obs_mode, local_size=local_size)

        obs, info = env.reset(seed=seed)
        obs = _maybe_transpose_obs(obs)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            if transitions is not None:
                prev_state = _extract_full_state(env)
            action, _ = model.predict(obs, deterministic=deterministic_eval)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            if transitions is not None:
                transitions.append(
                    {
                        "seed": seed,
                        "step": steps,
                        "action": int(action),
                        "state": prev_state,
                        "next_state": _extract_full_state(env),
                    }
                )
            steps += 1
            obs = _maybe_transpose_obs(next_obs)

        env.close()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(bool(info.get("success", False)))
        keys_collected.append(bool(info.get("has_key", False)))
        balls_collected.append(int(info.get("balls_collected", 0)))

    if not episode_rewards:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    mean_length = sum(episode_lengths) / len(episode_lengths)
    success_rate = sum(1 for s in successes if s) / len(successes)
    key_rate = sum(1 for k in keys_collected if k) / len(keys_collected)
    mean_balls = sum(balls_collected) / len(balls_collected)

    if transitions_path and transitions is not None:
        path = Path(transitions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(transitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return mean_reward, success_rate, mean_length, key_rate, mean_balls


def append_eval_result(
    results_csv: str,
    episode_count: int,
    timesteps: int | None,
    mean_reward: float,
    elapsed_seconds: float,
    success_rate: float,
    mean_length: float,
    key_rate: float,
    mean_balls: float,
    mean_reward_det: float | None = None,
    success_rate_det: float | None = None,
    mean_length_det: float | None = None,
    key_rate_det: float | None = None,
    mean_balls_det: float | None = None,
) -> None:
    """Append a single evaluation row to the CSV, adding a header if needed."""
    path = Path(results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "episode",
                    "timesteps",
                    "reward",
                    "reward_det",
                    "time_seconds",
                    "success_rate",
                    "success_rate_det",
                    "mean_length",
                    "mean_length_det",
                    "key_rate",
                    "key_rate_det",
                    "mean_balls_collected",
                    "mean_balls_collected_det",
                ]
            )
        writer.writerow(
            [
                episode_count,
                "" if timesteps is None else timesteps,
                f"{mean_reward:.6f}",
                "" if mean_reward_det is None else f"{mean_reward_det:.6f}",
                f"{elapsed_seconds:.2f}",
                f"{success_rate:.4f}",
                "" if success_rate_det is None else f"{success_rate_det:.4f}",
                f"{mean_length:.2f}",
                "" if mean_length_det is None else f"{mean_length_det:.2f}",
                f"{key_rate:.4f}",
                "" if key_rate_det is None else f"{key_rate_det:.4f}",
                f"{mean_balls:.2f}",
                "" if mean_balls_det is None else f"{mean_balls_det:.2f}",
            ]
        )


def _extract_full_state(env) -> dict:
    base = env.unwrapped
    grid = base.grid.encode()
    grid_copy = grid.copy() if hasattr(grid, "copy") else grid
    agent_pos = tuple(int(x) for x in base.agent_pos)
    agent_dir = int(base.agent_dir)
    return {
        "grid": grid_copy,
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
    }


class EpisodeSnapshotCallback(BaseCallback):
    """Tracks episode count, saves snapshots, evaluates, and logs results."""

    def __init__(
        self,
        snapshot_interval: int,
        snapshot_interval_steps: int | None,
        snapshot_dir: str,
        eval_seeds: list[int],
        eval_env_kwargs: dict,
        results_csv: str,
        start_time: float,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.snapshot_interval = snapshot_interval if snapshot_interval > 0 else None
        self.snapshot_interval_steps = snapshot_interval_steps
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.eval_seeds = eval_seeds
        self.eval_env_kwargs = eval_env_kwargs
        self.results_csv = results_csv
        self.start_time = start_time
        self.completed_episodes = 0
        self.next_snapshot = snapshot_interval if snapshot_interval > 0 else None
        self.next_snapshot_steps = snapshot_interval_steps
        self.eval_counter = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            return True

        episodes_finished = sum(1 for done in dones if done)
        if episodes_finished and self.snapshot_interval is not None:
            self.completed_episodes += episodes_finished

            while self.completed_episodes >= self.next_snapshot:
                self._save_and_evaluate(self.next_snapshot)
                self.next_snapshot += self.snapshot_interval

        if self.snapshot_interval_steps is not None:
            while self.num_timesteps >= self.next_snapshot_steps:
                self._save_and_evaluate(
                    self.completed_episodes,
                    timesteps=self.next_snapshot_steps,
                )
                self.next_snapshot_steps += self.snapshot_interval_steps

        return True

    def _save_and_evaluate(self, episode_count: int, timesteps: int | None = None) -> None:
        tag = episode_count if timesteps is None else timesteps
        episode_dir = self.snapshot_dir / f"episode_{tag:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(episode_dir / "model"))

        mean_reward, success_rate, mean_length, key_rate, mean_balls = evaluate_on_seeds(
            self.model,
            self.eval_seeds,
            size=self.eval_env_kwargs["size"],
            num_balls=self.eval_env_kwargs["num_balls"],
            num_walls=self.eval_env_kwargs["num_walls"],
            num_lava=self.eval_env_kwargs["num_lava"],
            reward_system=self.eval_env_kwargs["reward_system"],
            ball_reward=self.eval_env_kwargs["ball_reward"],
            key_reward=self.eval_env_kwargs["key_reward"],
            exit_reward=self.eval_env_kwargs["exit_reward"],
            exit_with_key_reward=self.eval_env_kwargs["exit_with_key_reward"],
            lava_penalty=self.eval_env_kwargs["lava_penalty"],
            step_penalty=self.eval_env_kwargs["step_penalty"],
            path_progress_scale=self.eval_env_kwargs["path_progress_scale"],
            deterministic_eval=False,
            transitions_path=str(episode_dir / "eval_transitions.pkl"),
            obs_mode=self.eval_env_kwargs.get("obs_mode", "partial"),
            local_size=self.eval_env_kwargs.get("local_size", 3),
        )

        (
            mean_reward_det,
            success_rate_det,
            mean_length_det,
            key_rate_det,
            mean_balls_det,
        ) = evaluate_on_seeds(
            self.model,
            self.eval_seeds,
            size=self.eval_env_kwargs["size"],
            num_balls=self.eval_env_kwargs["num_balls"],
            num_walls=self.eval_env_kwargs["num_walls"],
            num_lava=self.eval_env_kwargs["num_lava"],
            reward_system=self.eval_env_kwargs["reward_system"],
            ball_reward=self.eval_env_kwargs["ball_reward"],
            key_reward=self.eval_env_kwargs["key_reward"],
            exit_reward=self.eval_env_kwargs["exit_reward"],
            exit_with_key_reward=self.eval_env_kwargs["exit_with_key_reward"],
            lava_penalty=self.eval_env_kwargs["lava_penalty"],
            step_penalty=self.eval_env_kwargs["step_penalty"],
            path_progress_scale=self.eval_env_kwargs["path_progress_scale"],
            deterministic_eval=True,
            transitions_path=None,
            obs_mode=self.eval_env_kwargs.get("obs_mode", "partial"),
            local_size=self.eval_env_kwargs.get("local_size", 3),
        )
        elapsed = time.time() - self.start_time

        self.eval_counter += 1
        episode_value = self.eval_counter if timesteps is not None else episode_count
        append_eval_result(
            self.results_csv,
            episode_count=episode_value,
            timesteps=timesteps,
            mean_reward=mean_reward,
            elapsed_seconds=elapsed,
            success_rate=success_rate,
            mean_length=mean_length,
            key_rate=key_rate,
            mean_balls=mean_balls,
            mean_reward_det=mean_reward_det,
            success_rate_det=success_rate_det,
            mean_length_det=mean_length_det,
            key_rate_det=key_rate_det,
            mean_balls_det=mean_balls_det,
        )

        if self.verbose > 0:
            print(
                f"Snapshot @ {episode_count} episodes | "
                f"avg reward {mean_reward:.4f} | "
                f"success {success_rate:.2%} | "
                f"mean len {mean_length:.1f} | "
                f"key rate {key_rate:.2%} | "
                f"mean balls {mean_balls:.1f} | "
                f"elapsed {elapsed:.1f}s"
            )


def make_env(
    size: int = 16,
    num_balls: int = 24,
    num_walls: int = 16,
    num_lava: int = 16,
    reward_system: str = "path-gold-hazard-lever",
    ball_reward: float = 15.0,
    key_reward: float = 20.0,
    exit_reward: float = 50.0,
    exit_with_key_reward: float = 100.0,
    lava_penalty: float = -1.0,
    step_penalty: float = -0.05,
    path_progress_scale: float = 1.0,
    seed: int | None = None,
    rank: int = 0,
    obs_mode: str = "partial",
    local_size: int = 3,
) -> callable:
    """
    Create a function that returns a wrapped LeverGrid environment.

    Args:
        size: Grid size
        num_balls: Number of collectible balls
        num_walls: Number of wall blocks
        num_lava: Number of lava hazards
        reward_system: Reward components (path/gold/hazard/lever)
        ball_reward: Reward for collecting a ball
        key_reward: Reward for collecting the key
        exit_reward: Reward for reaching exit (path objective)
        exit_with_key_reward: Reward for exiting with key (lever objective)
        lava_penalty: Penalty for stepping on lava
        step_penalty: Step penalty when path objective is active
        reward_system: Reward components (path/gold/hazard/lever)
        ball_reward: Reward for collecting a ball
        key_reward: Reward for collecting the key
        exit_reward: Reward for reaching exit (path objective)
        exit_with_key_reward: Reward for exiting with key (lever objective)
        lava_penalty: Penalty for stepping on lava
        step_penalty: Step penalty when path objective is active
        path_progress_scale: Scale for path proximity shaping
        seed: Random seed (None for random)
        rank: Environment rank for vectorized envs

    Returns:
        Function that creates the environment
    """

    def _init() -> gym.Env:
        env = LeverGridEnv(
            size=size,
            num_balls=num_balls,
            num_walls=num_walls,
            num_lava=num_lava,
            reward_system=reward_system,
            ball_reward=ball_reward,
            key_reward=key_reward,
            exit_reward=exit_reward,
            exit_with_key_reward=exit_with_key_reward,
            lava_penalty=lava_penalty,
            step_penalty=step_penalty,
            path_progress_scale=path_progress_scale,
            action_mode="cardinal",
            render_mode="rgb_array",
        )
        # Wrap to convert dict obs to image obs for CNN policy
        env = _wrap_obs(env, obs_mode, local_size=local_size)
        # Wrap with Monitor for logging
        env = Monitor(env)

        if seed is not None:
            env.reset(seed=seed + rank)

        return env

    return _init


def train_ppo(
    total_timesteps: int = 1_000_000,
    snapshot_interval: int = 100,
    snapshot_interval_steps: int | None = None,
    n_envs: int = 8,
    n_eval_envs: int = 100,
    size: int = 16,
    num_balls: int = 24,
    num_walls: int = 16,
    num_lava: int = 16,
    reward_system: str = "path-gold-hazard-lever",
    ball_reward: float = 15.0,
    key_reward: float = 20.0,
    exit_reward: float = 50.0,
    exit_with_key_reward: float = 100.0,
    lava_penalty: float = -1.0,
    step_penalty: float = -0.05,
    path_progress_scale: float = 1.0,
    obs_mode: str = "partial",
    local_size: int = 3,
    learning_rate: float = 2.5e-4,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    norm_reward: bool = True,
    clip_reward: float = 10.0,
    features_dim: int = 128,
    seed: int | None = None,
    output_dir: str = "deeprl_runs/16/path-gold-hazard-lever",
    eval_seeds_path: str = "deeprl_runs/16/eval_env_seeds.json",
    results_csv: str = "deeprl_runs/16/path-gold-hazard-lever/episode_rewards.csv",
    verbose: int = 1,
    tensorboard_log: str | None = None,
    save_loss_plot: bool = False,
) -> PPO:
    """
    Train a PPO agent on the LeverGrid environment.

    Args:
        total_timesteps: Total training timesteps
        snapshot_interval: Snapshot/eval interval in episodes
        snapshot_interval_steps: Snapshot/eval interval in timesteps (optional)
        n_envs: Number of parallel environments
        n_eval_envs: Number of evaluation environments (seeds)
        size: Grid size
        num_balls: Number of collectible balls
        num_walls: Number of wall blocks
        num_lava: Number of lava hazards
        learning_rate: Learning rate
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient (higher = more exploration)
        features_dim: CNN feature dimension
        seed: Random seed
        output_dir: Directory to save models/results
        eval_seeds_path: Path to persist evaluation seeds
        results_csv: CSV path to log episode/reward/time
        verbose: Verbosity level
        tensorboard_log: TensorBoard log directory

    Returns:
        Trained PPO model
    """
    # Default n_steps to 2 * size^2 (scales with grid area).
    if n_steps is None:
        n_steps = 2 * size * size
    if batch_size is None:
        batch_size = n_steps // 2

    # Create output directory
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    eval_seeds = load_or_create_eval_seeds(
        seeds_path=eval_seeds_path,
        n_eval_envs=n_eval_envs,
        seed=seed,
    )

    # Create vectorized training environment
    print(f"Creating {n_envs} parallel environments...")
    env_fns = [
        make_env(
            size=size,
            num_balls=num_balls,
            num_walls=num_walls,
            num_lava=num_lava,
            reward_system=reward_system,
            ball_reward=ball_reward,
            key_reward=key_reward,
            exit_reward=exit_reward,
            exit_with_key_reward=exit_with_key_reward,
            lava_penalty=lava_penalty,
            step_penalty=step_penalty,
            path_progress_scale=path_progress_scale,
            seed=seed,
            rank=i,
            obs_mode=obs_mode,
            local_size=local_size,
        )
        for i in range(n_envs)
    ]

    # Use SubprocVecEnv for true parallelism (faster but uses more memory)
    # Use DummyVecEnv for debugging or if SubprocVecEnv causes issues
    try:
        train_env = SubprocVecEnv(env_fns)
    except Exception as e:
        print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        train_env = DummyVecEnv(env_fns)

    # Transpose observations to channel-first for CNN policies.
    # Skip when the obs is already (or ambiguously) channel-first — e.g.
    # LocalObsWrapper produces (3, 3, 3) which SB3 reads as channels-first.
    if not is_image_space_channels_first(train_env.observation_space):
        train_env = VecTransposeImage(train_env)

    # Optionally normalize rewards to stabilise PPO advantage estimates.
    # Observations are left raw (already scaled by ScaleObsWrapper).
    train_env = VecNormalize(
        train_env,
        norm_obs=False,
        norm_reward=norm_reward,
        clip_reward=clip_reward,
        gamma=gamma,
    )

    if verbose > 0:
        try:
            sample_obs = train_env.reset()
            print(f"Vec env obs shape: {sample_obs.shape}")
        except Exception as exc:
            print(f"Obs shape debug failed: {exc}")

    # Custom policy kwargs with MiniGrid feature extractor
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
    )

    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )

    # Setup episode-driven snapshot + eval callback
    eval_env_kwargs = dict(
        size=size,
        num_balls=num_balls,
        num_walls=num_walls,
        num_lava=num_lava,
        reward_system=reward_system,
        ball_reward=ball_reward,
        key_reward=key_reward,
        exit_reward=exit_reward,
        exit_with_key_reward=exit_with_key_reward,
        lava_penalty=lava_penalty,
        step_penalty=step_penalty,
        path_progress_scale=path_progress_scale,
        obs_mode=obs_mode,
        local_size=local_size,
    )

    # Train!
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"  Grid: {size}x{size}")
    print(f"  Balls: {num_balls}, Walls: {num_walls}, Lava: {num_lava}")
    print(f"  Reward system: {reward_system}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Output dir: {save_path}")
    print(f"  Eval seeds: {eval_seeds_path}")
    print(f"  Results CSV: {results_csv}")

    start_time = time.time()

    # Configure CSV/tensorboard logging for loss plots.
    log_dir = save_path / "logs"
    logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    callback = EpisodeSnapshotCallback(
        snapshot_interval=snapshot_interval,
        snapshot_interval_steps=snapshot_interval_steps,
        snapshot_dir=str(save_path / "episodes"),
        eval_seeds=eval_seeds,
        eval_env_kwargs=eval_env_kwargs,
        results_csv=results_csv,
        start_time=start_time,
        verbose=verbose,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s")

    if save_loss_plot:
        plot_path = save_path / "loss_plot.png"
        _plot_loss_curves(log_dir / "progress.csv", plot_path)

    # Save final model and VecNormalize stats
    final_path = save_path / "final_model"
    model.save(str(final_path))
    print(f"Final model saved to: {final_path}")
    vecnorm_path = save_path / "vecnormalize.pkl"
    train_env.save(str(vecnorm_path))
    print(f"VecNormalize stats saved to: {vecnorm_path}")

    # Cleanup
    train_env.close()

    return model


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    size: int = 16,
    num_balls: int = 25,
    num_walls: int = 8,
    num_lava: int = 8,
    reward_system: str = "path-gold-hazard-lever",
    ball_reward: float = 15.0,
    key_reward: float = 20.0,
    exit_reward: float = 50.0,
    exit_with_key_reward: float = 100.0,
    lava_penalty: float = -1.0,
    step_penalty: float = -0.05,
    path_progress_scale: float = 1.0,
    render: bool = False,
    seed: int | None = None,
    obs_mode: str = "partial",
    local_size: int = 3,
) -> dict:
    """
    Evaluate a trained PPO model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        size: Grid size
        num_balls: Number of balls
        num_walls: Number of walls
        num_lava: Number of lava cells
        reward_system: Reward components (path/gold/hazard/lever)
        ball_reward: Reward for collecting a ball
        key_reward: Reward for collecting the key
        exit_reward: Reward for reaching exit (path objective)
        exit_with_key_reward: Reward for exiting with key (lever objective)
        lava_penalty: Penalty for stepping on lava
        step_penalty: Step penalty when path objective is active
        path_progress_scale: Scale for path proximity shaping
        render: Whether to render (saves video if True)
        seed: Random seed

    Returns:
        Dictionary with evaluation metrics
    """
    if reward_system == "path" and exit_reward == 50.0:
        exit_reward = 200.0

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = LeverGridEnv(
        size=size,
        num_balls=num_balls,
        num_walls=num_walls,
        num_lava=num_lava,
        reward_system=reward_system,
        ball_reward=ball_reward,
        key_reward=key_reward,
        exit_reward=exit_reward,
        exit_with_key_reward=exit_with_key_reward,
        lava_penalty=lava_penalty,
        step_penalty=step_penalty,
        path_progress_scale=path_progress_scale,
        action_mode="cardinal",
        render_mode="rgb_array" if render else None,
    )
    env = _wrap_obs(env, obs_mode, local_size=local_size)

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    successes = []
    keys_collected = []
    balls_collected = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep if seed else None)
        obs = _maybe_transpose_obs(obs)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            obs = _maybe_transpose_obs(obs)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(info.get("success", False))
        keys_collected.append(info.get("has_key", False))
        balls_collected.append(info.get("balls_collected", 0))

    env.close()

    results = {
        "mean_reward": sum(episode_rewards) / n_episodes,
        "std_reward": (
            sum((r - sum(episode_rewards) / n_episodes) ** 2 for r in episode_rewards)
            / n_episodes
        )
        ** 0.5,
        "mean_length": sum(episode_lengths) / n_episodes,
        "success_rate": sum(successes) / n_episodes,
        "key_collection_rate": sum(keys_collected) / n_episodes,
        "mean_balls_collected": sum(balls_collected) / n_episodes,
    }

    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_length']:.1f}")
    print(f"  Success Rate: {results['success_rate']*100:.1f}%")
    print(f"  Key Collection Rate: {results['key_collection_rate']*100:.1f}%")
    print(f"  Mean Balls Collected: {results['mean_balls_collected']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on LeverGrid environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=100,
        help="Snapshot/eval interval in episodes",
    )
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        default=None,
        help="Snapshot/eval interval in timesteps (optional)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=8, help="Number of parallel environments"
    )
    parser.add_argument(
        "--n-eval-envs",
        type=int,
        default=100,
        help="Number of evaluation environments (seeds)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    parser.add_argument(
        "--reward-system",
        type=str,
        default="path-gold-hazard-lever",
        help="Reward system to train",
    )
    parser.add_argument(
        "--all-rewards",
        action="store_true",
        help="Train all reward systems",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated sizes to train (e.g. 16,32)",
    )
    parser.add_argument(
        "--grid-preset",
        type=str,
        choices=sorted(GRID_PRESETS),
        default=None,
        help="Use preset object counts (grid8 or grid16_scaled).",
    )

    # Environment parameters
    parser.add_argument("--size", type=int, default=16, help="Grid size")
    parser.add_argument(
        "--num-balls", type=int, default=24, help="Number of collectible balls"
    )
    parser.add_argument(
        "--num-walls", type=int, default=16, help="Number of wall blocks"
    )
    parser.add_argument(
        "--num-lava", type=int, default=16, help="Number of lava hazards"
    )

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--n-steps", type=int, default=None, help="Steps per env per update (default: 2*size^2)"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Minibatch size (default: n_steps/2)")
    parser.add_argument("--n-epochs", type=int, default=4, help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--features-dim", type=int, default=128, help="CNN feature dimension"
    )

    # Reward shaping
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=-0.05,
        help="Penalty per step (negative value encourages efficiency)",
    )
    parser.add_argument(
        "--path-progress-scale",
        type=float,
        default=1.0,
        help="Scale for BFS distance-based path shaping reward",
    )
    parser.add_argument(
        "--exit-reward",
        type=float,
        default=None,
        help="Reward for reaching exit (default: 200 for path, 50 otherwise)",
    )
    parser.add_argument(
        "--ball-reward", type=float, default=15.0, help="Reward for collecting a ball"
    )
    parser.add_argument(
        "--key-reward", type=float, default=20.0, help="Reward for collecting the key"
    )
    parser.add_argument(
        "--exit-with-key-reward",
        type=float,
        default=100.0,
        help="Reward for exiting with key (lever objective)",
    )
    parser.add_argument(
        "--lava-penalty",
        type=float,
        default=-1.0,
        help="Penalty for stepping on lava",
    )

    # Saving/logging
    parser.add_argument(
        "--output-root",
        type=str,
        default="deeprl_runs",
        help="Root directory for generated runs",
    )
    parser.add_argument(
        "--minigrid-ids-json",
        type=str,
        default="minigrid_ids.json",
        help="Path (relative to output-root if not absolute) to write MiniGrid ID mappings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow training into non-empty output directories",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="TensorBoard log directory (None to disable)",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--loss-plot",
        action="store_true",
        help="Save a loss plot after training.",
    )
    parser.add_argument(
        "--no-norm-reward",
        action="store_true",
        help="Disable VecNormalize reward normalization.",
    )
    parser.add_argument(
        "--clip-reward",
        type=float,
        default=10.0,
        help="VecNormalize reward clipping threshold (default: 10.0).",
    )

    # Observation mode
    parser.add_argument(
        "--obs-mode",
        type=str,
        choices=["partial", "local", "full"],
        default="partial",
        help="Observation mode: partial (7x7 cone), local (NxN top-down), full (full grid)",
    )
    parser.add_argument(
        "--local-size",
        type=int,
        default=7,
        help="Patch side length for --obs-mode local (must be odd)",
    )

    # Evaluation mode
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Evaluate model at this path instead of training",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Number of evaluation episodes"
    )

    args = parser.parse_args()

    if args.grid_preset:
        preset = GRID_PRESETS[args.grid_preset]
        args.size = preset["size"]
        args.num_balls = preset["num_balls"]
        args.num_walls = preset["num_walls"]
        args.num_lava = preset["num_lava"]
        args.sizes = None

    if args.minigrid_ids_json:
        ids_path = Path(args.minigrid_ids_json)
        if not ids_path.is_absolute():
            ids_path = Path(args.output_root) / ids_path
        write_minigrid_ids_json(ids_path)

    if args.eval:
        # Evaluation mode
        evaluate_model(
            model_path=args.eval,
            n_episodes=args.eval_episodes,
            size=args.size,
            num_balls=args.num_balls,
            num_walls=args.num_walls,
            num_lava=args.num_lava,
            reward_system=args.reward_system,
            seed=args.seed,
            obs_mode=args.obs_mode,
            local_size=args.local_size,
        )
        return

    # Training mode
    sizes = parse_sizes(args.sizes) or [args.size]
    reward_systems = REWARD_SYSTEMS if args.all_rewards else [args.reward_system]

    for size in sizes:
        size_root = Path(args.output_root) / str(size)
        eval_seeds_path = size_root / "eval_env_seeds.json"

        for reward_system in reward_systems:
            output_dir = size_root / reward_system
            if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
                print(
                    f"Skipping {output_dir} (directory not empty). "
                    "Use --overwrite to run anyway."
                )
                continue

            results_csv = output_dir / "episode_rewards.csv"
            tensorboard_log = None
            if args.tensorboard:
                tensorboard_log = str(Path(args.tensorboard) / str(size) / reward_system)

            train_ppo(
                total_timesteps=args.timesteps,
                snapshot_interval=args.snapshot_interval,
                snapshot_interval_steps=args.snapshot_steps,
                n_envs=args.n_envs,
                n_eval_envs=args.n_eval_envs,
                size=size,
                num_balls=args.num_balls,
                num_walls=args.num_walls,
                num_lava=args.num_lava,
                reward_system=reward_system,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                features_dim=args.features_dim,
                exit_reward=args.exit_reward if args.exit_reward is not None else (200.0 if reward_system == "path" else 50.0),
                ball_reward=args.ball_reward,
                key_reward=args.key_reward,
                exit_with_key_reward=args.exit_with_key_reward,
                lava_penalty=args.lava_penalty,
                step_penalty=args.step_penalty,
                path_progress_scale=args.path_progress_scale,
                obs_mode=args.obs_mode,
                local_size=args.local_size,
                seed=args.seed,
                output_dir=str(output_dir),
                eval_seeds_path=str(eval_seeds_path),
                results_csv=str(results_csv),
                tensorboard_log=tensorboard_log,
                norm_reward=not args.no_norm_reward,
                clip_reward=args.clip_reward,
                verbose=args.verbose,
                save_loss_plot=args.loss_plot,
            )


if __name__ == "__main__":
    main()
