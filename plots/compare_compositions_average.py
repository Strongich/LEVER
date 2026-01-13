"""
Plot average reward/time across X1/X5/X10 for trivial/double/triple modes.

Output is a single figure with two subplots (reward + time).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Ensure local imports work when executed outside the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import apply_paper_style, get_color

FONT_SCALE = 1.52
BAR_LABEL_SIZE = 8.0
apply_paper_style(font_scale=FONT_SCALE)

MODES = ["trivial", "double", "triple"]
APPROACHES = ["Training", "Targeted", "Exhaustive", "Hybrid"]


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _load_mode_rows(results_dir: str, spec: str, mode: str):
    csv_path = os.path.join(
        results_dir, spec, f"full_experiment_results_{mode}.csv"
    )
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _collect_decomp_overrides(results_dir: str, specs: list[str]):
    overrides = {}
    for mode in MODES:
        per_setup_means = defaultdict(list)
        for spec in specs:
            rows = _load_mode_rows(results_dir, spec, mode)
            if not rows:
                continue
            setup_vals = defaultdict(list)
            for row in rows:
                setup = row.get("setup")
                if not setup:
                    continue
                try:
                    decomp = float(row.get("decomp_time", 0.0) or 0.0)
                except ValueError:
                    continue
                setup_vals[setup].append(decomp)
            for setup, vals in setup_vals.items():
                if vals:
                    per_setup_means[setup].append(_mean(vals))
        setup_means = []
        for setup, vals in per_setup_means.items():
            if vals:
                setup_means.append(_mean(vals))
        if setup_means:
            overrides[mode] = _mean(setup_means)
    return overrides


def load_spec_means(
    results_dir: str,
    spec: str,
    decomp_overrides: dict[str, float] | None = None,
):
    mode_vals = []
    for mode in MODES:
        rows = _load_mode_rows(results_dir, spec, mode)
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

        if decomp_overrides and mode in decomp_overrides:
            decomp_mean = decomp_overrides[mode]
        else:
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


def plot_average(results_dir: str, output_path: str, specs: list[str]):
    decomp_overrides = _collect_decomp_overrides(results_dir, specs)
    spec_data = {}
    for spec in specs:
        data = load_spec_means(results_dir, spec, decomp_overrides=decomp_overrides)
        if data:
            spec_data[spec] = data

    if not spec_data:
        print(f"No data found in {results_dir}")
        return

    specs_used = [s for s in specs if s in spec_data]
    x = np.arange(len(APPROACHES))
    width = 0.8 / len(specs_used)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)

    for idx, spec in enumerate(specs_used):
        offset = (idx - (len(specs_used) - 1) / 2) * width
        rewards, times = spec_data[spec]
        reward_vals = [rewards.get(k, np.nan) for k in APPROACHES]
        time_vals = [times.get(k, np.nan) for k in APPROACHES]
        color = get_color(idx)
        label = spec

        bars_r = axes[0].bar(x + offset, reward_vals, width, label=label, color=color)
        bars_t = axes[1].bar(x + offset, time_vals, width, label=label, color=color)

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

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot averages across X1/X5/X10 for trivial/double/triple."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory containing X1/X5/X10 subfolders",
    )
    parser.add_argument(
        "--specs",
        nargs="*",
        default=["X1", "X5", "X10"],
        help="Spec labels to include (e.g., X1 X5 X10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output plot filename",
    )
    args = parser.parse_args()

    plot_average(args.results_dir, args.output, args.specs)


if __name__ == "__main__":
    main()
