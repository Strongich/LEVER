"""Evaluate best-case targeted composition by directly combining policy Q-tables."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tabular.full_experiment import (
    _infer_grid_size_from_states,
    combine_q_tables_list,
    greedy_eval,
    init_env_from_run,
    normalize_seed,
)


EXPERIMENTS = [
    {
        "setup": "path-gold",
        "type": "trivial",
        "combos": [
            ["path", "gold"],
        ],
    },
    {
        "setup": "path-gold-hazard",
        "type": "trivial",
        "combos": [
            ["path", "gold", "hazard"],
        ],
    },
    {
        "setup": "path-gold-hazard",
        "type": "double",
        "combos": [
            ["path-gold", "hazard"],
        ],
    },
    {
        "setup": "path-gold-hazard-lever",
        "type": "trivial",
        "combos": [
            ["path", "gold", "hazard", "lever"],
        ],
    },
    {
        "setup": "path-gold-hazard-lever",
        "type": "double",
        "combos": [
            ["path-gold", "hazard-lever"],
        ],
    },
    {
        "setup": "path-gold-hazard-lever",
        "type": "triple",
        "combos": [
            ["path-gold-hazard", "lever"],
        ],
    },
]


def _find_latest_run_dir(base_dir: str, spec: str) -> str | None:
    if not os.path.isdir(base_dir):
        return None
    candidates = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(f"{spec}_")
    ]
    if not candidates:
        return None
    return os.path.join(base_dir, sorted(candidates)[-1])


def _read_episode_rows(rewards_path: Path) -> list[tuple[float, int]]:
    if not rewards_path.exists():
        return []
    rows = []
    with rewards_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                episode_val = int(float(row.get("episode", "0")))
                reward_val = float(row.get("reward", "nan"))
            except ValueError:
                continue
            rows.append((reward_val, episode_val))
    return rows


def _select_first_max_episode(rewards_path: Path) -> int | None:
    """Return the first (earliest) episode that achieved the maximum reward."""
    rows = _read_episode_rows(rewards_path)
    if not rows:
        return None
    max_reward = max(r for r, _ in rows)
    top_eps = [ep for r, ep in rows if r == max_reward]
    if not top_eps:
        return None
    return min(top_eps)  # First episode to hit max reward


def _resolve_episode_dir(base_dir: Path, episode: int) -> Path | None:
    if episode is None:
        return None
    candidates = [
        base_dir / f"episode_{episode:06d}",
        base_dir / f"episode_{episode:05d}",
        base_dir / f"episode_{episode:04d}",
        base_dir / f"episode_{episode}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for entry in base_dir.glob("episode_*"):
        try:
            suffix = entry.name.split("_", 1)[1]
            if int(suffix) == episode:
                return entry
        except (IndexError, ValueError):
            continue
    return None


def load_best_q_table(run_dir: Path, policy: str, seed: str) -> np.ndarray | None:
    """Load Q-table from the first episode that achieved max reward."""
    rewards_path = run_dir / policy / f"seed_{seed}" / "episode_rewards.csv"
    episode_id = _select_first_max_episode(rewards_path)
    if episode_id is None:
        return None
    episodes_dir = run_dir / policy / f"seed_{seed}" / "episodes"
    episode_dir = _resolve_episode_dir(episodes_dir, episode_id)
    if episode_dir is None:
        return None
    q_table_path = episode_dir / "q_table.npy"
    if not q_table_path.exists():
        return None
    return np.load(q_table_path)


def load_scratch_q_table(run_dir: Path, setup: str, seed: str) -> np.ndarray | None:
    q_path = run_dir / setup / f"seed_{seed}" / "q_table_final.npy"
    if not q_path.exists():
        return None
    return np.load(q_path)


def collect_seeds(run_dir: Path) -> list[str]:
    seeds = []
    for seed_dir in sorted((run_dir / "path").glob("seed_*")):
        name = seed_dir.name.split("_")[-1]
        seeds.append(name)
    return seeds


def evaluate_combo(
    env,
    run_dir: Path,
    seed: str,
    combo: Iterable[str],
) -> float | None:
    """Evaluate a single combination using first-to-max Q-table per policy."""
    q_tables = []
    for policy in combo:
        q_table = load_best_q_table(run_dir, policy, seed)
        if q_table is None:
            return None
        q_tables.append(q_table)

    combined = combine_q_tables_list(q_tables)
    return greedy_eval(env, combined)


def run_spec(
    run_dir: Path,
    spec: str,
    seeds: list[str] | None,
    output_csv: Path,
    legacy_gold_exit_penalty: bool,
):
    grid_size = _infer_grid_size_from_states(str(run_dir)) or 16
    seed_list = seeds or collect_seeds(run_dir)

    rows = []
    for exp in EXPERIMENTS:
        setup = exp["setup"]
        exp_type = exp["type"]
        combos = exp["combos"]
        for seed in seed_list:
            seed_name = normalize_seed(seed)
            env = init_env_from_run(
                str(run_dir),
                setup,
                seed_name,
                grid_size,
                legacy_gold_exit_penalty=legacy_gold_exit_penalty,
            )

            scratch_reward = None
            scratch_q = load_scratch_q_table(run_dir, setup, seed_name)
            if scratch_q is not None:
                scratch_reward = greedy_eval(env, scratch_q)

            best_reward = None
            for combo in combos:
                reward = evaluate_combo(env, run_dir, seed_name, combo)
                if reward is None:
                    continue
                if best_reward is None or reward > best_reward:
                    best_reward = reward

            rows.append(
                {
                    "setup": setup,
                    "type": exp_type,
                    "spec": spec,
                    "seed": seed_name,
                    "scratch_reward": scratch_reward,
                    "direct_targeted_reward": best_reward,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setup",
                "type",
                "spec",
                "seed",
                "scratch_reward",
                "direct_targeted_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate best-case targeted composition by direct Q-table combination."
    )
    parser.add_argument(
        "--state-runs-dir",
        type=str,
        default="state_runs",
        help="Base directory containing X1/X5/X10 runs.",
    )
    parser.add_argument(
        "--specs",
        nargs="*",
        default=["X1", "X5", "X10"],
        help="Specs to evaluate (default: X1 X5 X10).",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Optional list of seeds to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="targeted_direct",
        help="Base output directory for CSV results.",
    )
    parser.add_argument(
        "--legacy-gold-exit-penalty",
        action="store_true",
        help="Use legacy gold exit penalty (-20 if not all gold collected).",
    )
    args = parser.parse_args()

    base_dir = Path(args.state_runs_dir)
    for spec in args.specs:
        run_dir = _find_latest_run_dir(str(base_dir), spec)
        if run_dir is None:
            print(f"Warning: no run dir found for {spec} in {base_dir}")
            continue
        output_csv = Path(args.output_dir) / spec / "targeted_direct_results.csv"
        run_spec(
            Path(run_dir),
            spec,
            args.seeds,
            output_csv,
            legacy_gold_exit_penalty=args.legacy_gold_exit_penalty,
        )


if __name__ == "__main__":
    main()
