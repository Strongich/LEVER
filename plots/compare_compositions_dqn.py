"""
Generate per-setup comparison plots (reward + time) for DQN composition approaches.

Reads results_dqn/full_experiment_dqn_{mode}.csv and produces one figure per setup
with reward (left) and time with log scale (right).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import apply_paper_style, get_color

FONT_SCALE = 1.52
BAR_LABEL_SIZE = 10.0
apply_paper_style(font_scale=FONT_SCALE)

MODE_SETUPS = {
    "trivial": ["path-gold", "path-gold-hazard", "path-gold-hazard-lever"],
    "double": ["path-gold-hazard", "path-gold-hazard-lever"],
    "triple": ["path-gold-hazard-lever"],
}

APPROACHES = ["Training", "Targeted", "Exhaustive", "Hybrid"]


def pretty_setup_name(setup: str) -> str:
    if setup.startswith("path-gold"):
        return setup.replace("path-gold", "gold-path", 1)
    return setup


def load_means(df: pd.DataFrame):
    rewards = {
        "Training": df["scratch_reward"].mean(),
        "Targeted": df["targeted_reward"].mean(),
        "Exhaustive": df["exhaustive_reward"].mean(),
        "Hybrid": df["hybrid_reward"].mean(),
    }
    decomp_mean = df["decomp_time"].mean() if "decomp_time" in df.columns else 0.0
    times = {
        "Training": df["scratch_time_s"].mean(),
        "Targeted": df["targeted_time"].mean() + decomp_mean,
        "Exhaustive": df["exhaustive_time"].mean() + decomp_mean,
        "Hybrid": df["hybrid_time"].mean() + decomp_mean,
    }
    return rewards, times


def load_direct_bounds(
    targeted_csv: str | None,
    exhaustive_csv: str | None,
    hybrid_csv: str | None,
) -> dict[tuple[str, str], dict[str, float]]:
    bounds: dict[tuple[str, str], dict[str, float]] = {}

    def _load_csv(path: str, reward_col: str, label: str):
        if not path or not os.path.exists(path):
            if path:
                print(f"Direct bounds: {path} not found, skipping {label}")
            return
        df = pd.read_csv(path)
        if reward_col not in df.columns:
            print(f"Direct bounds: {path} missing {reward_col}, skipping {label}")
            return
        grouped = df.groupby(["setup", "type"], dropna=False)[reward_col].mean()
        for (setup, mode), value in grouped.items():
            if value != value:
                continue
            bounds.setdefault((setup, mode), {})[label] = float(value)

    _load_csv(targeted_csv, "direct_targeted_reward", "targeted")
    _load_csv(exhaustive_csv, "direct_hybrid_reward", "exhaustive")
    _load_csv(hybrid_csv, "direct_hybrid_reward", "hybrid")

    return bounds


def plot_setup(
    setup: str,
    mode: str,
    df: pd.DataFrame,
    output_path: str,
    direct_bounds: dict[tuple[str, str], dict[str, float]] | None = None,
):
    subset = df[df["setup"] == setup]
    if subset.empty:
        print(f"Skipping {setup} ({mode}): no data")
        return

    rewards, times = load_means(subset)
    setup_label = pretty_setup_name(setup)

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
    if direct_bounds:
        bound_vals = direct_bounds.get((setup, mode), {})
        for bound_key, bar_idx in hatched_indices.items():
            if bound_key not in bound_vals:
                continue
            bound_val = bound_vals[bound_key]
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
    axes[0].set_title(f"{setup_label} ({mode}) - Reward")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(APPROACHES)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title(f"{setup_label} ({mode}) - Time")
    axes[1].set_yscale("log")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-setup comparison plots for DQN composition."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_dqn",
        help="Directory containing full_experiment_dqn_{mode}.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/comparisons_dqn",
        help="Base output directory for plots.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["trivial", "double", "triple"],
        help="Modes to plot.",
    )
    parser.add_argument(
        "--targeted-direct-csv",
        type=str,
        default="targeted_direct_dqn/targeted_direct_dqn_results.csv",
        help="Path to targeted_direct_dqn_results.csv for hatched upper-bound overlay",
    )
    parser.add_argument(
        "--exhaustive-direct-csv",
        type=str,
        default="exhaustive_direct_dqn/hybrid_direct_dqn_results.csv",
        help="Path to exhaustive upper-bound CSV (hybrid_direct format with k=5)",
    )
    parser.add_argument(
        "--hybrid-direct-csv",
        type=str,
        default="hybrid_direct_dqn/hybrid_direct_dqn_results.csv",
        help="Path to hybrid_direct_dqn_results.csv for hatched upper-bound overlay",
    )
    parser.add_argument(
        "--csv-prefix",
        type=str,
        default="full_experiment_dqn",
        help="Filename prefix for result CSVs (e.g. full_experiment_dqn or full_experiment_ppo)",
    )
    args = parser.parse_args()

    direct_bounds = load_direct_bounds(
        args.targeted_direct_csv,
        args.exhaustive_direct_csv,
        args.hybrid_direct_csv,
    )

    for mode in args.modes:
        csv_path = os.path.join(args.results_dir, f"{args.csv_prefix}_{mode}.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping mode '{mode}': {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        setups = MODE_SETUPS.get(mode, [])

        for setup in setups:
            out_path = os.path.join(args.output_dir, mode, f"{setup}.png")
            plot_setup(setup, mode, df, out_path, direct_bounds=direct_bounds)


if __name__ == "__main__":
    main()
