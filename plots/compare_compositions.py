"""
Generate a publication-quality bar chart comparing average rewards for
training, targeted, and exhaustive compositions.

Input:  full_experiment_results.csv (seed-level rewards)
Output: plots/comparison.png
Style:  paper-ready settings imported from plot_metrics.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Ensure we can import the shared paper-ready plotting settings
PAPER_SETTINGS_PATH = Path("/home/strongich/PythonProj/UNI/MASTERS")
if str(PAPER_SETTINGS_PATH) not in sys.path:
    sys.path.insert(0, str(PAPER_SETTINGS_PATH))

# Apply paper-ready style on import
from plot_metrics import apply_paper_style, get_color

apply_paper_style()


def load_means(csv_path: str):
    """Load experiment results and compute mean rewards and times."""
    df = pd.read_csv(csv_path)
    rewards = {
        "Training": df["scratch_reward"].mean(),
        "Targeted": df["targeted_reward"].mean(),
        "Exhaustive": df["exhaustive_reward"].mean(),
    }
    avg_decomp_time = df["decomp_time"].mean()
    times = {
        "Training": 13.54,  # provided constant
        "Targeted": df["targeted_time"].mean() + avg_decomp_time,  # Add decomp time
        "Exhaustive": df["exhaustive_time"].mean() + avg_decomp_time,  # Add decomp time
    }
    return rewards, times


def plot_comparison(rewards: dict, times: dict, output_path: str):
    labels = list(rewards.keys())
    values = [rewards[k] for k in labels]
    time_labels = [f"{times[k]:.2f} s" if pd.notna(times[k]) else "n/a" for k in labels]
    colors = [get_color(i) for i in range(len(labels))]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors)

    for bar, val, tlabel in zip(bars, values, time_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() / 2.0,
            tlabel,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Average Reward")
    ax.set_title("Training vs. Targeted vs. Exhaustive Composition Comparison")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    input_csv = "full_experiment_results_16_1.csv"
    output_path = "plots/comparison_1.png"
    rewards, times = load_means(input_csv)
    plot_comparison(rewards, times, output_path)
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    main()
