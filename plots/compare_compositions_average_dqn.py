"""
Plot average reward/time for DQN policy composition (trivial/double/triple).

Reads results_dqn/full_experiment_dqn_{trivial,double,triple}.csv,
averages per-seed rewards across setups and modes, and produces a
side-by-side bar chart (reward | time) with one bar per approach.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import apply_paper_style, get_color

FONT_SCALE = 1.52
BAR_LABEL_SIZE = 14.0
apply_paper_style(font_scale=FONT_SCALE)

MODES = ["trivial", "double", "triple"]
APPROACHES = ["Training", "Targeted", "Exhaustive", "Hybrid"]


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _load_mode_rows(results_dir: str, mode: str, csv_prefix: str = "full_experiment_dqn"):
    csv_path = os.path.join(results_dir, f"{csv_prefix}_{mode}.csv")
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def load_means(results_dir: str, csv_prefix: str = "full_experiment_dqn"):
    """Load all mode CSVs, return (rewards, times) averaged across modes."""
    mode_vals = []
    for mode in MODES:
        rows = _load_mode_rows(results_dir, mode, csv_prefix=csv_prefix)
        if not rows:
            continue

        scratch_rewards = []
        targeted_rewards = []
        exhaustive_rewards = []
        hybrid_rewards = []
        scratch_times = []
        targeted_times = []
        exhaustive_times = []
        hybrid_times = []
        decomp_times = []

        for row in rows:
            try:
                scratch_rewards.append(float(row["scratch_reward"]))
                targeted_rewards.append(float(row["targeted_reward"]))
                exhaustive_rewards.append(float(row["exhaustive_reward"]))
                hybrid_rewards.append(float(row["hybrid_reward"]))
                scratch_times.append(float(row["scratch_time_s"]))
                targeted_times.append(float(row["targeted_time"]))
                exhaustive_times.append(float(row["exhaustive_time"]))
                hybrid_times.append(float(row["hybrid_time"]))
                decomp_times.append(float(row.get("decomp_time", 0.0) or 0.0))
            except (KeyError, ValueError):
                continue

        if not scratch_rewards:
            continue

        decomp_mean = _mean(decomp_times) if decomp_times else 0.0
        mode_vals.append(
            {
                "Training": _mean(scratch_rewards),
                "Targeted": _mean(targeted_rewards),
                "Exhaustive": _mean(exhaustive_rewards),
                "Hybrid": _mean(hybrid_rewards),
                "Training_time": _mean(scratch_times),
                "Targeted_time": _mean(targeted_times) + decomp_mean,
                "Exhaustive_time": _mean(exhaustive_times) + decomp_mean,
                "Hybrid_time": _mean(hybrid_times) + decomp_mean,
            }
        )

    if not mode_vals:
        return None

    means = defaultdict(list)
    for mode_data in mode_vals:
        for key, value in mode_data.items():
            if value == value:  # skip NaN
                means[key].append(value)

    rewards = {k: _mean(means[k]) for k in APPROACHES}
    times = {k: _mean(means[f"{k}_time"]) for k in APPROACHES}
    return rewards, times


def _load_direct_bounds(
    targeted_csv: str | None,
    exhaustive_csv: str | None,
    hybrid_csv: str | None,
) -> dict[str, float]:
    """Load direct eval CSVs, compute per-type mean then grand mean for each approach."""
    bounds: dict[str, float] = {}

    for label, csv_path, reward_col in [
        ("targeted", targeted_csv, "direct_targeted_reward"),
        ("exhaustive", exhaustive_csv, "direct_hybrid_reward"),
        ("hybrid", hybrid_csv, "direct_hybrid_reward"),
    ]:
        if csv_path is None or not os.path.exists(csv_path):
            continue
        per_type: dict[str, list[float]] = defaultdict(list)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    per_type[row["type"]].append(float(row[reward_col]))
                except (KeyError, ValueError):
                    continue
        if not per_type:
            continue
        type_means = [_mean(vals) for vals in per_type.values() if vals]
        if type_means:
            bounds[label] = _mean(type_means)

    return bounds


def plot_average(
    results_dir: str,
    output_path: str,
    targeted_direct_csv: str | None = None,
    exhaustive_direct_csv: str | None = None,
    hybrid_direct_csv: str | None = None,
    csv_prefix: str = "full_experiment_dqn",
):
    data = load_means(results_dir, csv_prefix=csv_prefix)
    if data is None:
        print(f"No data found in {results_dir}")
        return

    rewards, times = data
    direct_bounds = _load_direct_bounds(
        targeted_direct_csv, exhaustive_direct_csv, hybrid_direct_csv
    )

    x = np.arange(len(APPROACHES))
    width = 0.5
    colors = [get_color(i) for i in range(len(APPROACHES))]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)

    reward_vals = [rewards.get(k, np.nan) for k in APPROACHES]
    time_vals = [times.get(k, np.nan) for k in APPROACHES]

    bars_r = axes[0].bar(x, reward_vals, width, color=colors)
    bars_t = axes[1].bar(x, time_vals, width, color=colors)

    # Hatched upper-bound overlays on reward plot
    hatched_indices = {
        "targeted": 1,
        "exhaustive": 2,
        "hybrid": 3,
    }
    for bound_key, bar_idx in hatched_indices.items():
        if bound_key not in direct_bounds:
            continue
        bound_val = direct_bounds[bound_key]
        real_val = reward_vals[bar_idx]
        if np.isfinite(real_val) and bound_val <= real_val:
            continue
        axes[0].bar(
            x[bar_idx],
            bound_val,
            width,
            color="#cccccc",
            edgecolor="#888888",
            hatch="///",
            alpha=0.35,
            zorder=0,
        )
        # Skip hatched label if it would overlap the real bar label
        if np.isfinite(real_val) and abs(bound_val - real_val) < 0.5:
            continue
        axes[0].text(
            x[bar_idx],
            bound_val,
            f"{bound_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#888888",
        )

    for bar in bars_r:
        val = bar.get_height()
        if np.isfinite(val):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                val,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_SIZE,
                color="#555555",
            )
    for bar in bars_t:
        val = bar.get_height()
        if np.isfinite(val):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                val,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_SIZE,
                color="#555555",
            )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(APPROACHES)
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Composition - Reward")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(APPROACHES)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Average Composition - Time")
    axes[1].set_yscale("log")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot average DQN composition reward/time."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_dqn",
        help="Directory containing full_experiment_dqn_{trivial,double,triple}.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/average_dqn.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--targeted-direct-csv",
        type=str,
        default=None,
        help="Path to targeted_direct_dqn_results.csv for hatched upper-bound overlay",
    )
    parser.add_argument(
        "--exhaustive-direct-csv",
        type=str,
        default=None,
        help="Path to exhaustive upper-bound CSV (hybrid_direct format with k=5)",
    )
    parser.add_argument(
        "--hybrid-direct-csv",
        type=str,
        default=None,
        help="Path to hybrid_direct_dqn_results.csv for hatched upper-bound overlay",
    )
    parser.add_argument(
        "--csv-prefix",
        type=str,
        default="full_experiment_dqn",
        help="Filename prefix for result CSVs (e.g. full_experiment_dqn or full_experiment_ppo)",
    )
    args = parser.parse_args()

    plot_average(
        args.results_dir,
        args.output,
        targeted_direct_csv=args.targeted_direct_csv,
        exhaustive_direct_csv=args.exhaustive_direct_csv,
        hybrid_direct_csv=args.hybrid_direct_csv,
        csv_prefix=args.csv_prefix,
    )


if __name__ == "__main__":
    main()
