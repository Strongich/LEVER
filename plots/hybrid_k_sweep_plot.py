"""
Plot hybrid top-k sweep results (reward + time).

Reads the sweep CSV and overlays a scratch baseline reward from results CSVs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt

# Ensure local imports work when executed outside the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import apply_paper_style, get_color

FONT_SCALE = 1.52
apply_paper_style(font_scale=FONT_SCALE)


def load_sweep_csv(path: str):
    k_vals = []
    rewards = []
    times = []
    min_candidates = None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k_vals.append(int(row["k"]))
            rewards.append(float(row["avg_reward"]))
            times.append(float(row["avg_time_s"]))
            if min_candidates is None and "min_candidates" in row:
                try:
                    min_candidates = int(float(row["min_candidates"]))
                except ValueError:
                    min_candidates = None
    return k_vals, rewards, times, min_candidates


def load_scratch_baseline(results_dir: str) -> tuple[float | None, float | None]:
    scratch_rewards = []
    scratch_times = []
    pattern = os.path.join(results_dir, "*", "full_experiment_results_*.csv")
    for path in glob.glob(pattern):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    scratch_rewards.append(float(row["scratch_reward"]))
                    scratch_times.append(float(row["scratch_time_s"]))
                except (KeyError, ValueError):
                    continue
    if not scratch_rewards:
        return None, None
    avg_reward = sum(scratch_rewards) / len(scratch_rewards)
    avg_time = sum(scratch_times) / len(scratch_times) if scratch_times else None
    return avg_reward, avg_time


def _collect_decomp_override(results_dir: str) -> float | None:
    mode_vals = []
    pattern = os.path.join(results_dir, "*", "full_experiment_results_*.csv")
    for path in glob.glob(pattern):
        setup_vals = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                setup = row.get("setup")
                if not setup:
                    continue
                try:
                    decomp = float(row.get("decomp_time", 0.0) or 0.0)
                except ValueError:
                    continue
                setup_vals.setdefault(setup, []).append(decomp)
        per_setup_means = [
            sum(vals) / len(vals) for vals in setup_vals.values() if vals
        ]
        if per_setup_means:
            mode_vals.append(sum(per_setup_means) / len(per_setup_means))
    if not mode_vals:
        return None
    return sum(mode_vals) / len(mode_vals)


def load_targeted_exhaustive_baselines(
    results_dir: str, decomp_override: float | None
):
    targeted_rewards = []
    targeted_times = []
    exhaustive_rewards = []
    exhaustive_times = []
    pattern = os.path.join(results_dir, "*", "full_experiment_results_*.csv")
    for path in glob.glob(pattern):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if decomp_override is None:
                        decomp = float(row.get("decomp_time", 0.0) or 0.0)
                    else:
                        decomp = decomp_override
                    targeted_rewards.append(float(row["targeted_reward"]))
                    exhaustive_rewards.append(float(row["exhaustive_reward"]))
                    targeted_times.append(float(row["targeted_time"]) + decomp)
                    exhaustive_times.append(float(row["exhaustive_time"]) + decomp)
                except (KeyError, ValueError):
                    continue
    if not targeted_rewards:
        return None
    return {
        "targeted_reward": sum(targeted_rewards) / len(targeted_rewards),
        "targeted_time": sum(targeted_times) / len(targeted_times),
        "exhaustive_reward": sum(exhaustive_rewards) / len(exhaustive_rewards),
        "exhaustive_time": sum(exhaustive_times) / len(exhaustive_times),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot hybrid top-k sweep results."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="CSV produced by hybrid_k_sweep.py",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory to compute scratch baseline reward",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path (defaults to plots/<input_csv_basename>.png)",
    )
    args = parser.parse_args()

    k_vals, rewards, times, min_candidates = load_sweep_csv(args.input_csv)
    baseline_reward, baseline_time = load_scratch_baseline(args.results_dir)
    decomp_override = _collect_decomp_override(args.results_dir)
    te_baselines = load_targeted_exhaustive_baselines(
        args.results_dir, decomp_override
    )
    output_path = args.output
    if output_path is None:
        base = os.path.splitext(os.path.basename(args.input_csv))[0]
        output_path = os.path.join("plots", f"{base}.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    color_hybrid = get_color(0)
    color_base = get_color(1)
    color_targeted = "#C44E52"
    color_exhaustive = "#55A868"

    axes[0].plot(k_vals, rewards, marker="o", color=color_hybrid, label="Hybrid")
    if baseline_reward is not None:
        axes[0].axhline(
            baseline_reward,
            color=color_base,
            linestyle="--",
            label="Scratch",
        )
    if te_baselines:
        target_x = 1
        target_y = te_baselines["targeted_reward"]
        axes[0].plot(
            [target_x],
            [target_y],
            marker="^",
            markersize=8,
            linestyle="None",
            color=color_targeted,
            label="Targeted",
        )
        if min_candidates is not None:
            exhaust_x = min_candidates
            exhaust_y = te_baselines["exhaustive_reward"]
            axes[0].plot(
                [exhaust_x],
                [exhaust_y],
                marker="*",
                markersize=10,
                linestyle="None",
                color=color_exhaustive,
                label="Exhaustive",
            )
            line_x = [target_x] + k_vals + [exhaust_x]
            line_y = [target_y] + rewards + [exhaust_y]
            axes[0].plot(
                line_x, line_y, linestyle="--", color=color_hybrid, alpha=0.35, linewidth=1.0
            )
    axes[0].set_xlabel("Top-k")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Hybrid k Sweep - Reward")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    xticks = sorted(set(k_vals + [1] + ([min_candidates] if min_candidates else [])))
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels([str(k) for k in xticks])

    axes[1].plot(k_vals, times, marker="o", color=color_hybrid, label="Hybrid")
    if baseline_time is not None:
        axes[1].axhline(
            baseline_time,
            color=color_base,
            linestyle="--",
            label="Scratch",
        )
    if te_baselines:
        target_x = 1
        target_y = te_baselines["targeted_time"]
        axes[1].plot(
            [target_x],
            [target_y],
            marker="^",
            markersize=8,
            linestyle="None",
            color=color_targeted,
            label="Targeted",
        )
        if min_candidates is not None:
            exhaust_x = min_candidates
            exhaust_y = te_baselines["exhaustive_time"]
            axes[1].plot(
                [exhaust_x],
                [exhaust_y],
                marker="*",
                markersize=10,
                linestyle="None",
                color=color_exhaustive,
                label="Exhaustive",
            )
            line_x = [target_x] + k_vals + [exhaust_x]
            line_y = [target_y] + times + [exhaust_y]
            axes[1].plot(
                line_x, line_y, linestyle="--", color=color_hybrid, alpha=0.35, linewidth=1.0
            )
    axes[1].set_xlabel("Top-k")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Hybrid k Sweep - Time")
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([str(k) for k in xticks])
    axes[1].set_yscale("log")
    if baseline_time is not None:
        axes[1].legend(loc="best")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    main()
