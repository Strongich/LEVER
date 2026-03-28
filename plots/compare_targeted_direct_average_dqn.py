"""
Plot average reward for direct targeted DQN composition (trivial/double/triple).

Reads targeted_direct_dqn/targeted_direct_dqn_results.csv and aggregates
scratch_reward vs direct_targeted_reward, grouped by type.
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

FONT_SCALE = 1.55
BAR_LABEL_SIZE = 10.5
apply_paper_style(font_scale=FONT_SCALE)

APPROACHES = ["Training", "Targeted"]
TYPES = ["trivial", "double", "triple"]


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def load_type_means(results_csv: str):
    """Load CSV, return per-type mean rewards."""
    if not os.path.exists(results_csv):
        return None

    per_type = defaultdict(lambda: {"scratch": [], "targeted": []})

    with open(results_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_type = row.get("type") or "unknown"
            try:
                per_type[row_type]["scratch"].append(float(row["scratch_reward"]))
                per_type[row_type]["targeted"].append(
                    float(row["direct_targeted_reward"])
                )
            except (KeyError, ValueError):
                continue

    if not per_type:
        return None

    type_data = {}
    for t in TYPES:
        if t not in per_type:
            continue
        values = per_type[t]
        if not values["scratch"] or not values["targeted"]:
            continue
        type_data[t] = {
            "Training": _mean(values["scratch"]),
            "Targeted": _mean(values["targeted"]),
        }

    return type_data if type_data else None


def plot_average(results_csv: str, output_path: str):
    type_data = load_type_means(results_csv)
    if type_data is None:
        print(f"No data found in {results_csv}")
        return

    types_used = [t for t in TYPES if t in type_data]
    x = np.arange(len(APPROACHES))
    width = 0.65 / len(types_used)

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 3.6), constrained_layout=True)

    for idx, t in enumerate(types_used):
        offset = (idx - (len(types_used) - 1) / 2) * width
        rewards = type_data[t]
        reward_vals = [rewards.get(k, np.nan) for k in APPROACHES]
        color = get_color(idx)

        bars = ax.bar(x + offset, reward_vals, width, label=t, color=color)

        for bar in bars:
            val = bar.get_height()
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_SIZE,
                    color="#555555",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(APPROACHES)
    ax.set_ylabel("Average Reward")
    ax.set_title("Direct Targeted (DQN) - Reward")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.72),
        ncol=1,
        frameon=False,
        fontsize=14,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average direct targeted DQN results (trivial/double/triple)."
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="targeted_direct_dqn/targeted_direct_dqn_results.csv",
        help="Path to targeted_direct_dqn_results.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="targeted_direct_dqn/average_targeted_direct_dqn.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

    plot_average(args.results_csv, args.output)


if __name__ == "__main__":
    main()
