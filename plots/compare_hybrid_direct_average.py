"""
Plot average reward for direct hybrid composition across X1/X5/X10.

Reads hybrid_direct/<spec>/hybrid_direct_results.csv and aggregates
scratch_reward vs direct_hybrid_reward. Mirrors compare_targeted_direct_average.py.
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

FONT_SCALE = 1.55
BAR_LABEL_SIZE = 10.5
apply_paper_style(font_scale=FONT_SCALE)

APPROACHES = ["Training", "Hybrid"]


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def load_spec_means(results_dir: str, spec: str):
    csv_path = os.path.join(results_dir, spec, "hybrid_direct_results.csv")
    if not os.path.exists(csv_path):
        return None

    per_type = defaultdict(lambda: {"scratch": [], "hybrid": []})

    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_type = row.get("type") or "unknown"
            try:
                per_type[row_type]["scratch"].append(float(row["scratch_reward"]))
                per_type[row_type]["hybrid"].append(
                    float(row["direct_hybrid_reward"])
                )
            except (KeyError, ValueError):
                continue

    if not per_type:
        return None

    type_means = []
    for values in per_type.values():
        if not values["scratch"] or not values["hybrid"]:
            continue
        type_means.append(
            {
                "Training": _mean(values["scratch"]),
                "Hybrid": _mean(values["hybrid"]),
            }
        )

    if not type_means:
        return None

    rewards = {
        "Training": _mean([m["Training"] for m in type_means]),
        "Hybrid": _mean([m["Hybrid"] for m in type_means]),
    }
    return rewards


def plot_average(results_dir: str, output_path: str, specs: list[str]):
    spec_data = {}
    for spec in specs:
        data = load_spec_means(results_dir, spec)
        if data:
            spec_data[spec] = data

    if not spec_data:
        print(f"No data found in {results_dir}")
        return

    specs_used = [s for s in specs if s in spec_data]
    x = np.arange(len(APPROACHES))
    width = 0.65 / len(specs_used)

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 3.6), constrained_layout=True)

    for idx, spec in enumerate(specs_used):
        offset = (idx - (len(specs_used) - 1) / 2) * width
        rewards = spec_data[spec]
        reward_vals = [rewards.get(k, np.nan) for k in APPROACHES]
        color = get_color(idx)
        label = spec

        bars = ax.bar(x + offset, reward_vals, width, label=label, color=color)

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
    ax.set_title("Direct Hybrid - Reward")

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
        description="Plot average direct hybrid results across X1/X5/X10."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="hybrid_direct",
        help="Directory containing hybrid_direct/<spec>/hybrid_direct_results.csv",
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
        default="hybrid_direct/average_hybrid_direct.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

    plot_average(args.results_dir, args.output, args.specs)


if __name__ == "__main__":
    main()
