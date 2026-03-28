"""
Plot average reward/time across X1/X5/X10 for trivial/double/triple modes.

Optionally overlays upper-bound direct-composition rewards (targeted/hybrid/exhaustive)
computed from state_runs and stores them in a direct_bounds CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure local imports work when executed outside the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import apply_paper_style, get_color
from tabular.full_experiment import init_env_from_run

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


def _compute_direct_bounds(
    state_runs_dir: str,
    specs: list[str],
    legacy_gold_exit_penalty: bool,
):
    from tabular.full_experiment import _infer_grid_size_from_states, normalize_seed
    from tabular import targeted_direct_eval as tde
    from tabular import hybrid_direct_eval as hde

    bounds = {}
    for spec in specs:
        run_dir = tde._find_latest_run_dir(state_runs_dir, spec)
        if run_dir is None:
            continue
        run_path = Path(run_dir)
        grid_size = _infer_grid_size_from_states(str(run_path)) or 16
        seeds = tde.collect_seeds(run_path)

        per_method = {
            "targeted": defaultdict(list),
            "hybrid": defaultdict(list),
            "exhaustive": defaultdict(list),
        }

        targeted_map = {(e["setup"], e["type"]): e for e in tde.EXPERIMENTS}
        hybrid_map = {(e["setup"], e["type"]): e for e in hde.EXPERIMENTS}

        for key, texp in targeted_map.items():
            setup, exp_type = key
            combos = texp["combos"]
            hexp = hybrid_map.get(key)
            subtasks = hexp["subtasks"] if hexp else None
            for seed in seeds:
                seed_name = normalize_seed(seed)
                env = init_env_from_run(
                    str(run_path),
                    setup,
                    seed_name,
                    grid_size,
                    legacy_gold_exit_penalty=legacy_gold_exit_penalty,
                )

                targeted_reward = None
                for combo in combos:
                    reward = tde.evaluate_combo(env, run_path, seed_name, combo)
                    if reward is None:
                        continue
                    if targeted_reward is None or reward > targeted_reward:
                        targeted_reward = reward
                if targeted_reward is not None:
                    per_method["targeted"][exp_type].append(targeted_reward)

                if subtasks:
                    hybrid_reward = hde.evaluate_hybrid(
                        env, run_path, seed_name, subtasks, top_k=3
                    )
                    if hybrid_reward is not None:
                        per_method["hybrid"][exp_type].append(hybrid_reward)

                    exhaustive_reward = hde.evaluate_hybrid(
                        env, run_path, seed_name, subtasks, top_k=5
                    )
                    if exhaustive_reward is not None:
                        per_method["exhaustive"][exp_type].append(exhaustive_reward)

        spec_bounds = {}
        for method, by_type in per_method.items():
            type_means = []
            for vals in by_type.values():
                if vals:
                    type_means.append(_mean(vals))
            if type_means:
                spec_bounds[method] = _mean(type_means)
        if spec_bounds:
            bounds[spec] = spec_bounds
    return bounds


def _read_direct_bounds(path: str):
    if not path or not os.path.exists(path):
        return {}
    bounds = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            spec = row.get("spec")
            if not spec:
                continue
            spec_bounds = {}
            for key in ("targeted", "exhaustive", "hybrid"):
                try:
                    spec_bounds[key] = float(row.get(key, "nan"))
                except ValueError:
                    continue
            if spec_bounds:
                bounds[spec] = spec_bounds
    return bounds


def _write_direct_bounds(path: str, bounds: dict[str, dict[str, float]]):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["spec", "targeted", "exhaustive", "hybrid"]
        )
        writer.writeheader()
        for spec, spec_bounds in bounds.items():
            writer.writerow(
                {
                    "spec": spec,
                    "targeted": spec_bounds.get("targeted"),
                    "exhaustive": spec_bounds.get("exhaustive"),
                    "hybrid": spec_bounds.get("hybrid"),
                }
            )


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
    parser_state_runs = plot_average.state_runs_dir
    direct_bounds_path = plot_average.direct_bounds_csv
    legacy_gold_exit_penalty = plot_average.legacy_gold_exit_penalty
    overwrite_bounds = plot_average.overwrite_bounds

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

    # Overlay direct bounds as translucent bars (reward plot only).
    bounds = {}
    if not overwrite_bounds:
        bounds = _read_direct_bounds(direct_bounds_path)
    if (not bounds) and parser_state_runs:
        bounds = _compute_direct_bounds(
            parser_state_runs,
            specs_used,
            legacy_gold_exit_penalty=legacy_gold_exit_penalty,
        )
        _write_direct_bounds(direct_bounds_path, bounds)

    if bounds:
        hatched_indices = {
            "targeted": 1,
            "exhaustive": 2,
            "hybrid": 3,
        }
        for idx, spec in enumerate(specs_used):
            if spec not in bounds:
                continue
            offset = (idx - (len(specs_used) - 1) / 2) * width
            spec_bounds = bounds[spec]
            rewards, _ = spec_data[spec]
            for bound_key, bar_idx in hatched_indices.items():
                if bound_key not in spec_bounds:
                    continue
                bound_val = spec_bounds[bound_key]
                if bound_val != bound_val:  # skip NaN
                    continue
                real_val = rewards.get(APPROACHES[bar_idx], np.nan)
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
    axes[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.88, 1.02))

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
    parser.add_argument(
        "--state-runs-dir",
        type=str,
        default=None,
        help="Optional state_runs directory used to compute direct bounds.",
    )
    parser.add_argument(
        "--direct-bounds-csv",
        type=str,
        default=None,
        help="Optional CSV path to store/read direct bounds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute direct bounds even if the CSV already exists.",
    )
    parser.add_argument(
        "--legacy-gold-exit-penalty",
        action="store_true",
        help="Use legacy gold exit penalty when computing bounds.",
    )
    args = parser.parse_args()

    plot_average.state_runs_dir = args.state_runs_dir
    bounds_csv = args.direct_bounds_csv
    if bounds_csv is None:
        auto_path = os.path.join(args.results_dir, "direct_bounds.csv")
        if os.path.exists(auto_path):
            bounds_csv = auto_path
    plot_average.direct_bounds_csv = bounds_csv
    plot_average.legacy_gold_exit_penalty = args.legacy_gold_exit_penalty
    plot_average.overwrite_bounds = args.overwrite

    plot_average(args.results_dir, args.output, args.specs)


if __name__ == "__main__":
    main()
