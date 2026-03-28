"""Evaluate best-case hybrid composition by trying top-k DQN snapshots per sub-policy via GPI."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dqn.full_experiment import (
    GRID_SIZE,
    LOCAL_SIZE,
    OBS_MODE,
    evaluate_gpi_seed,
    evaluate_single_model,
    get_dqn_model,
    get_scratch_info,
    set_grid_params,
    set_obs_params,
)
from policy_reusability.data_generation.deeprl.train_dqn import (
    MinigridFeaturesExtractor,  # noqa: F401 – needed so DQN.load can unpickle the CNN
)

EXPERIMENTS = [
    {
        "setup": "path-gold",
        "type": "trivial",
        "subtasks": ["path", "gold"],
    },
    {
        "setup": "path-gold-hazard",
        "type": "trivial",
        "subtasks": ["path", "gold", "hazard"],
    },
    {
        "setup": "path-gold-hazard",
        "type": "double",
        "subtasks": ["path-gold", "hazard"],
    },
    {
        "setup": "path-gold-hazard-lever",
        "type": "trivial",
        "subtasks": ["path", "gold", "hazard", "lever"],
    },
    {
        "setup": "path-gold-hazard-lever",
        "type": "double",
        "subtasks": ["path-gold", "hazard-lever"],
    },
    {
        "setup": "path-gold-hazard-lever",
        "type": "triple",
        "subtasks": ["path-gold-hazard", "lever"],
    },
]


def select_top_k_snapshots(
    base_dir: str, reward_system: str, top_k: int = 3, prefilter: str = "v1"
) -> list[str]:
    """Return up to top_k model paths for best snapshots of a reward system.

    Prefilter v1: reward>0 AND reward_det>0, sort by [reward, reward_det] (asc).
    Prefilter v2: no positive filter, sort by [reward_det, success_rate] (asc).
    Best snapshots are taken from the end of the sorted list.
    """
    csv_path = os.path.join(base_dir, reward_system, "episode_rewards.csv")
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path)
    for col in ("reward", "reward_det", "timesteps", "success_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["reward", "reward_det", "timesteps"])

    if prefilter == "v1":
        candidates = df[(df["reward"] > 0) & (df["reward_det"] > 0)].copy()
        sort_cols = ["reward", "reward_det"]
    elif prefilter == "v2":
        candidates = df.copy()
        sort_cols = ["reward_det", "success_rate"]
    else:
        raise ValueError(f"Unknown prefilter: {prefilter}")

    if candidates.empty:
        return []

    candidates = candidates.sort_values(
        sort_cols, ascending=[True] * len(sort_cols)
    )
    model_paths = []
    for _, row in candidates.iloc[::-1].iterrows():
        timesteps = int(row["timesteps"])
        model_path = os.path.join(
            base_dir,
            reward_system,
            "episodes",
            f"episode_{timesteps:06d}",
            "model.zip",
        )
        if os.path.exists(model_path):
            model_paths.append(model_path)
            if len(model_paths) >= top_k:
                break

    return model_paths


def run_hybrid_direct(
    base_dir: str,
    eval_seeds_path: str,
    output_dir: str,
    top_k: int,
    device: str,
    prefilter: str,
):
    with open(eval_seeds_path, "r") as f:
        eval_seeds: list[int] = json.load(f)
    print(f"Loaded {len(eval_seeds)} eval seeds")

    rows: list[dict] = []

    for exp in EXPERIMENTS:
        setup = exp["setup"]
        exp_type = exp["type"]
        subtasks = exp["subtasks"]

        print(f"\n{'=' * 60}")
        print(f"Experiment: {setup} ({exp_type})")
        print(f"{'=' * 60}")

        # Scratch baseline
        scratch_model_path, _, _ = get_scratch_info(
            base_dir, setup, prefilter=prefilter
        )
        scratch_model = None
        if scratch_model_path and os.path.exists(scratch_model_path):
            scratch_model = get_dqn_model(scratch_model_path, device)
            print(f"Scratch model: {scratch_model_path}")
        else:
            print(f"No scratch model found for {setup}")

        # Load top-k snapshots per subtask
        per_task_models = []
        valid = True
        for task_name in subtasks:
            paths = select_top_k_snapshots(
                base_dir, task_name, top_k, prefilter
            )
            if not paths:
                print(f"  No snapshots for subtask '{task_name}', skipping experiment")
                valid = False
                break
            models = [get_dqn_model(p, device) for p in paths]
            per_task_models.append(models)
            print(f"  Subtask '{task_name}': {len(models)} snapshots")

        if not valid:
            # Still write rows with None for hybrid reward
            for seed in eval_seeds:
                scratch_reward = None
                if scratch_model is not None:
                    scratch_reward = evaluate_single_model(
                        scratch_model, setup, seed, device
                    )
                rows.append(
                    {
                        "setup": setup,
                        "type": exp_type,
                        "seed": seed,
                        "scratch_reward": scratch_reward,
                        "direct_hybrid_reward": None,
                    }
                )
            continue

        combos = list(itertools.product(*per_task_models))
        print(f"  {len(combos)} combos ({' x '.join(str(len(m)) for m in per_task_models)})")

        # Evaluate per seed
        print(f"Evaluating on {len(eval_seeds)} seeds...")
        for i, seed in enumerate(eval_seeds):
            scratch_reward = None
            if scratch_model is not None:
                scratch_reward = evaluate_single_model(
                    scratch_model, setup, seed, device
                )

            best_hybrid_reward = None
            for combo in combos:
                reward = evaluate_gpi_seed(list(combo), setup, seed, device)
                if best_hybrid_reward is None or reward > best_hybrid_reward:
                    best_hybrid_reward = reward

            rows.append(
                {
                    "setup": setup,
                    "type": exp_type,
                    "seed": seed,
                    "scratch_reward": scratch_reward,
                    "direct_hybrid_reward": best_hybrid_reward,
                }
            )

            if (i + 1) % 10 == 0:
                print(f"  evaluated {i + 1}/{len(eval_seeds)} seeds")

        # Print aggregate
        seed_rows = [
            r for r in rows if r["setup"] == setup and r["type"] == exp_type
        ]
        scratch_vals = [r["scratch_reward"] for r in seed_rows if r["scratch_reward"] is not None]
        hybrid_vals = [r["direct_hybrid_reward"] for r in seed_rows if r["direct_hybrid_reward"] is not None]
        print(f"  scratch mean: {sum(scratch_vals)/len(scratch_vals):.2f}" if scratch_vals else "  scratch mean: N/A")
        print(f"  hybrid  mean: {sum(hybrid_vals)/len(hybrid_vals):.2f}" if hybrid_vals else "  hybrid  mean: N/A")

    # Write CSV
    output_csv = os.path.join(output_dir, "hybrid_direct_dqn_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setup",
                "type",
                "seed",
                "scratch_reward",
                "direct_hybrid_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate best-case hybrid DQN composition via GPI (cross-product of top-k snapshots)."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="deeprl_runs_dqn_try6/8",
        help="Directory containing per-reward-system DQN policy folders.",
    )
    parser.add_argument(
        "--eval-seeds-path",
        type=str,
        default="deeprl_runs_dqn_try6/8/eval_env_seeds.json",
        help="JSON file with list of eval seed integers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hybrid_direct_dqn",
        help="Output directory for CSV results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k snapshots to keep per subtask.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--prefilter",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help=(
            "Snapshot prefilter: v1=positive reward filter, v2=no positive filter."
        ),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help="Grid size (must match training grid size).",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        choices=["partial", "local", "full"],
        default=OBS_MODE,
        help="Observation mode used for evaluation.",
    )
    parser.add_argument(
        "--local-size",
        type=int,
        default=LOCAL_SIZE,
        help="Patch side length for --obs-mode local (must match training).",
    )
    args = parser.parse_args()
    if args.grid_size != GRID_SIZE:
        set_grid_params(args.grid_size)
    set_obs_params(args.obs_mode, args.local_size)

    run_hybrid_direct(
        base_dir=args.base_dir,
        eval_seeds_path=args.eval_seeds_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        device=args.device,
        prefilter=args.prefilter,
    )


if __name__ == "__main__":
    main()
