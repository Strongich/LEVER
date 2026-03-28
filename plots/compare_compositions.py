"""
Generate comparison plots (reward + time) for composition approaches.

Outputs one figure per setup with reward/time stacked horizontally and bars grouped
by approach with X1/X5/X10 columns.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure local imports work when executed outside the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import apply_paper_style, get_color

FONT_SCALE = 1.52
BAR_LABEL_SIZE = 8.0
apply_paper_style(font_scale=FONT_SCALE)

DEFAULT_SPECS = ["X1", "X5", "X10"]
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


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _decomp_mean_for_setup(
    df_all: pd.DataFrame, setup: str, specs: list[str] | None
) -> float | None:
    if "decomp_time" not in df_all.columns:
        return None
    if "spec" not in df_all.columns or not specs:
        vals = df_all[df_all["setup"] == setup]["decomp_time"].dropna().tolist()
        return _mean([float(v) for v in vals])
    per_spec = []
    for spec in specs:
        subset = df_all[(df_all["setup"] == setup) & (df_all["spec"] == spec)]
        vals = subset["decomp_time"].dropna().tolist()
        mean_val = _mean([float(v) for v in vals])
        if mean_val is not None:
            per_spec.append(mean_val)
    return _mean(per_spec)


def load_means(df: pd.DataFrame, decomp_override: float | None = None):
    """Load experiment results and compute mean rewards and times."""
    rewards = {
        "Training": df["scratch_reward"].mean(),
        "Targeted": df["targeted_reward"].mean(),
        "Exhaustive": df["exhaustive_reward"].mean(),
        "Hybrid": df["hybrid_reward"].mean(),
    }
    avg_decomp_time = (
        decomp_override if decomp_override is not None else df["decomp_time"].mean()
    )
    times = {
        "Training": df["scratch_time_s"].mean(),
        "Targeted": df["targeted_time"].mean() + avg_decomp_time,
        "Exhaustive": df["exhaustive_time"].mean() + avg_decomp_time,
        "Hybrid": df["hybrid_time"].mean() + avg_decomp_time,
    }
    return rewards, times


def load_direct_bounds(path: str | None) -> dict[str, dict[str, float]]:
    """Load direct bounds CSV (spec,targeted,exhaustive,hybrid), keyed by spec."""
    if not path or not os.path.exists(path):
        return {}
    bounds: dict[str, dict[str, float]] = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        spec = row.get("spec")
        if not spec:
            continue
        spec_bounds = {}
        for key in ("targeted", "exhaustive", "hybrid"):
            try:
                val = float(row.get(key, "nan"))
                if val == val:  # skip NaN
                    spec_bounds[key] = val
            except (ValueError, TypeError):
                continue
        if spec_bounds:
            bounds[spec] = spec_bounds
    return bounds


def plot_reward_time(
    setup: str,
    mode: str,
    spec_data: dict,
    output_path: str,
    direct_bounds: dict[str, dict[str, float]] | None = None,
):
    specs = [s for s in DEFAULT_SPECS if s in spec_data]
    if not specs:
        print(f"Skipping {setup} ({mode}): no spec data available")
        return

    rewards_by_spec = {spec: spec_data[spec][0] for spec in specs}
    times_by_spec = {spec: spec_data[spec][1] for spec in specs}

    x = np.arange(len(APPROACHES))
    width = 0.8 / len(specs)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    setup_label = pretty_setup_name(setup)
    for idx, spec in enumerate(specs):
        offset = (idx - (len(specs) - 1) / 2) * width
        reward_vals = [rewards_by_spec[spec].get(k, np.nan) for k in APPROACHES]
        time_vals = [times_by_spec[spec].get(k, np.nan) for k in APPROACHES]
        color = get_color(idx)

        bars_r = axes[0].bar(x + offset, reward_vals, width, label=spec, color=color)
        bars_t = axes[1].bar(x + offset, time_vals, width, label=spec, color=color)

        for bar in bars_r:
            val = bar.get_height()
            if pd.notna(val):
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
            if pd.notna(val):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_SIZE,
                    color="#555555",
                )

    # Hatched upper-bound overlays on reward plot.
    hatched_indices = {
        "targeted": 1,
        "exhaustive": 2,
        "hybrid": 3,
    }
    if direct_bounds:
        for idx, spec in enumerate(specs):
            bound_vals = direct_bounds.get(spec, {})
            if not bound_vals:
                continue
            offset = (idx - (len(specs) - 1) / 2) * width
            reward_vals = [rewards_by_spec[spec].get(k, np.nan) for k in APPROACHES]
            for bound_key, bar_idx in hatched_indices.items():
                if bound_key not in bound_vals:
                    continue
                bound_val = bound_vals[bound_key]
                if bound_val != bound_val:  # NaN
                    continue
                real_val = reward_vals[bar_idx]
                if np.isfinite(real_val) and bound_val <= real_val:
                    continue
                axes[0].bar(
                    x[bar_idx] + offset,
                    bound_val,
                    width,
                    color="#cccccc",
                    edgecolor="#888888",
                    hatch="///",
                    alpha=0.35,
                    zorder=0,
                    label="_nolegend_",
                )
                if np.isfinite(real_val) and abs(bound_val - real_val) < 0.5:
                    continue
                axes[0].text(
                    x[bar_idx] + offset,
                    bound_val,
                    f"{bound_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#888888",
                )

    for ax, ylabel, title in zip(
        axes,
        ["Average Reward", "Time (s)"],
        [
            f"{setup_label} ({mode}) - Reward",
            f"{setup_label} ({mode}) - Time",
        ],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(APPROACHES)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylabel == "Time (s)":
            ax.set_yscale("log")

    handles, labels = axes[1].get_legend_handles_labels()
    # Position legend inside the axes near the upper-left (above Targeted),
    # without overlapping the title.
    axes[1].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        bbox_transform=axes[1].transAxes,
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for composition experiments."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="CSV with setup/spec columns from tabular/full_experiment.py",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory containing X1/X5/X10 subfolders",
    )
    parser.add_argument(
        "--mode",
        choices=["trivial", "double", "triple"],
        default="trivial",
        help="Which experiment mode to plot",
    )
    parser.add_argument(
        "--setups",
        nargs="*",
        default=None,
        help="Optional setup list override (defaults to mode presets)",
    )
    parser.add_argument(
        "--specs",
        nargs="*",
        default=DEFAULT_SPECS,
        help="Spec labels to plot (e.g., X1 X5 X10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for plots (defaults to plots/comparisons/<mode>)",
    )
    parser.add_argument(
        "--direct-bounds-csv",
        type=str,
        default=None,
        help="CSV with spec,targeted,exhaustive,hybrid columns for hatched upper-bound overlay "
        "(defaults to {results-dir}/direct_bounds.csv if it exists)",
    )
    args = parser.parse_args()
    setups = args.setups or MODE_SETUPS.get(args.mode, [])

    bounds_csv = args.direct_bounds_csv
    if bounds_csv is None:
        auto_path = os.path.join(args.results_dir, "direct_bounds.csv")
        if os.path.exists(auto_path):
            bounds_csv = auto_path
    direct_bounds = load_direct_bounds(bounds_csv)

    def build_spec_data(df_all: pd.DataFrame, setup: str):
        decomp_mean = _decomp_mean_for_setup(df_all, setup, args.specs)
        spec_data = {}
        for spec in args.specs:
            if "spec" in df_all.columns:
                subset = df_all[(df_all["setup"] == setup) & (df_all["spec"] == spec)]
            else:
                subset = df_all[df_all["setup"] == setup]
            if subset.empty:
                continue
            spec_data[spec] = load_means(subset, decomp_override=decomp_mean)
        return spec_data

    if args.input_csv:
        df_all = pd.read_csv(args.input_csv)
        for setup in setups:
            spec_data = build_spec_data(df_all, setup)
            out_dir = args.output_dir or os.path.join("plots", "comparisons", args.mode)
            out_path = os.path.join(out_dir, f"{setup}.png")
            plot_reward_time(setup, args.mode, spec_data, out_path, direct_bounds=direct_bounds)
        return

    results = {}
    for spec in args.specs:
        csv_path = os.path.join(
            args.results_dir, spec, f"full_experiment_results_{args.mode}.csv"
        )
        if not os.path.exists(csv_path):
            print(f"Skipping missing CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        results[spec] = df

    for setup in setups:
        if results:
            df_all = pd.concat(results.values(), ignore_index=True)
        else:
            df_all = pd.DataFrame()
        decomp_mean = _decomp_mean_for_setup(df_all, setup, args.specs)
        spec_data = {}
        for spec, df in results.items():
            subset = df[df["setup"] == setup]
            if subset.empty:
                continue
            spec_data[spec] = load_means(subset, decomp_override=decomp_mean)
        out_dir = args.output_dir or os.path.join("plots", "comparisons", args.mode)
        out_path = os.path.join(out_dir, f"{setup}.png")
        plot_reward_time(setup, args.mode, spec_data, out_path, direct_bounds=direct_bounds)


if __name__ == "__main__":
    main()
