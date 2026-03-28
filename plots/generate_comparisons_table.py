"""
Generate the LaTeX comparisons table from results CSVs.

Example:
  python plots/generate_comparisons_table.py \
    --results-8 results_8 \
    --results-16 results_16 \
    --output tables/comparisons.tex
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SPECS_DEFAULT = ["X1", "X5", "X10"]
MODES = ["trivial", "double", "triple"]

MODE_SETUPS = {
    "trivial": [
        ("path-gold", "P+G"),
        ("path-gold-hazard", "P+G+H"),
        ("path-gold-hazard-lever", "P+G+H+L"),
    ],
    "double": [
        ("path-gold-hazard", "PG+H"),
        ("path-gold-hazard-lever", "PG+HL"),
    ],
    "triple": [
        ("path-gold-hazard-lever", "PGH+L"),
    ],
}

METHODS = ["TFS", "TC", "EC", "HC"]
REWARD_COL = {
    "TFS": "scratch_reward",
    "TC": "targeted_reward",
    "EC": "exhaustive_reward",
    "HC": "hybrid_reward",
}
TIME_COL = {
    "TFS": "scratch_time_s",
    "TC": "targeted_time",
    "EC": "exhaustive_time",
    "HC": "hybrid_time",
}


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _std(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_float_or_zero(val) -> float:
    if val is None or val == "":
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _load_rows(results_dir: Path, spec: str, mode: str) -> list[dict]:
    path = results_dir / spec / f"full_experiment_results_{mode}.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _compute_decomp_override(rows_by_spec: dict[str, list[dict]], setup: str) -> float:
    per_spec_means = []
    for rows in rows_by_spec.values():
        vals = [
            _safe_float_or_zero(r.get("decomp_time"))
            for r in rows
            if r.get("setup") == setup
        ]
        if vals:
            per_spec_means.append(_mean(vals))
    override = _mean([v for v in per_spec_means if v is not None])
    return override if override is not None else 0.0


def _compute_setup_means(
    rows_by_spec: dict[str, list[dict]], setup: str
) -> dict[str, dict[str, float | None]]:
    decomp_override = _compute_decomp_override(rows_by_spec, setup)
    per_spec_rewards: dict[str, list[float]] = {m: [] for m in METHODS}
    per_spec_times: dict[str, list[float]] = {m: [] for m in METHODS}
    pooled_rewards: dict[str, list[float]] = {m: [] for m in METHODS}

    for rows in rows_by_spec.values():
        setup_rows = [r for r in rows if r.get("setup") == setup]
        if not setup_rows:
            continue

        for method in METHODS:
            reward_vals = [_safe_float(r.get(REWARD_COL[method])) for r in setup_rows]
            time_vals = [_safe_float(r.get(TIME_COL[method])) for r in setup_rows]
            reward_mean = _mean(reward_vals)
            time_mean = _mean(time_vals)
            if reward_mean is not None:
                per_spec_rewards[method].append(reward_mean)
            if time_mean is not None:
                per_spec_times[method].append(time_mean)
            pooled_rewards[method].extend([v for v in reward_vals if v is not None])

    means = {}
    for method in METHODS:
        reward = _mean(per_spec_rewards[method])
        reward_std = _std(pooled_rewards[method])
        time = _mean(per_spec_times[method])
        if method != "TFS" and time is not None:
            time += decomp_override
        means[method] = {"reward": reward, "reward_std": reward_std, "time": time}
    return means


def _format_value(val: float | None, bold: bool = False) -> str:
    if val is None or not math.isfinite(val):
        return "--"
    text = f"{val:.2f}"
    return f"\\textbf{{{text}}}" if bold else text


def _format_percent(val: float | None, bold: bool = False) -> str:
    if val is None or not math.isfinite(val):
        return "--"
    text = f"{val:.2f}\\%"
    return f"\\textbf{{{text}}}" if bold else text


def _format_reward(mean: float | None, std: float | None, bold: bool = False) -> str:
    if mean is None or not math.isfinite(mean):
        return "--"
    if std is None or not math.isfinite(std):
        text = f"{mean:.2f}"
    else:
        text = f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{text}}}" if bold else text


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _build_section(results_dir: Path, specs: list[str], title: str) -> list[str]:
    lines = []
    lines.append(f"  \\textbf{{{title}}}\\\\[0pt]")
    lines.append("  \\begin{tabular}{llrrrr}")
    lines.append("    \\toprule")
    lines.append("    Setup & Method & Reward & Time & $\\Delta$R & Speedup \\\\")
    lines.append("    \\midrule")

    for mode in MODES:
        lines.append(
            f"    \\multicolumn{{6}}{{l}}{{\\textit{{{mode.capitalize()}}}}}\\\\"
        )
        rows_by_spec = {spec: _load_rows(results_dir, spec, mode) for spec in specs}

        for setup, setup_label in MODE_SETUPS[mode]:
            means = _compute_setup_means(rows_by_spec, setup)
            tfs_reward = means["TFS"]["reward"]
            tfs_time = means["TFS"]["time"]

            reward_candidates = {
                m: means[m]["reward"]
                for m in ("TC", "EC", "HC")
                if means[m]["reward"] is not None
            }
            speedup_candidates = {}
            for m in ("TC", "EC", "HC"):
                time_val = means[m]["time"]
                if tfs_time and time_val is not None:
                    speedup_candidates[m] = (tfs_time - time_val) / tfs_time * 100.0

            best_reward = max(reward_candidates.values()) if reward_candidates else None
            best_speedup = (
                max(speedup_candidates.values()) if speedup_candidates else None
            )

            for idx, method in enumerate(METHODS):
                reward = means[method]["reward"]
                time_val = means[method]["time"]

                if method == "TFS":
                    delta = None
                    speedup = None
                else:
                    delta = (
                        (reward - tfs_reward) / tfs_reward * 100.0
                        if tfs_reward and reward is not None
                        else None
                    )
                speedup = speedup_candidates.get(method)

                bold_reward = (
                    method in ("TC", "EC", "HC")
                    and best_reward is not None
                    and reward is not None
                    and math.isfinite(best_reward)
                    and abs(reward - best_reward) < 1e-9
                )
                bold_speedup = (
                    method in ("TC", "EC", "HC")
                    and best_speedup is not None
                    and speedup is not None
                    and math.isfinite(best_speedup)
                    and abs(speedup - best_speedup) < 1e-9
                )

                setup_cell = setup_label if idx == 0 else ""
                method_label = method
                lines.append(
                    f"    {setup_cell} & {method_label}  & "
                    f"{_format_reward(reward, means[method]['reward_std'], bold=bold_reward)} & "
                    f"{_format_value(time_val)} & "
                    f"{_format_percent(delta)} & "
                    f"{_format_percent(speedup, bold=bold_speedup)} \\\\"
                )

            if setup != MODE_SETUPS[mode][-1][0]:
                lines.append("    \\cmidrule{1-6}")

        if mode != MODES[-1]:
            lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    return lines


def _render_table(
    results_left: Path,
    results_right: Path,
    specs: list[str],
    title_left: str,
    title_right: str,
    note: str | None,
) -> str:
    lines = []
    lines.append("\\begin{table*}[h]")
    lines.append("  \\centering")
    lines.append("  \\scriptsize")
    lines.append("  \\setlength{\\tabcolsep}{2pt}")
    lines.append("  \\renewcommand{\\arraystretch}{0.9}")
    lines.append(
        "  \\caption{Averages across X1/X5/X10. Bold indicates the best value among TC/"
        " EC/HC for Reward and Speedup.}"
    )
    lines.append("  \\label{tab:comparisons-side-by-side}")
    lines.append("  \\begin{minipage}{0.49\\textwidth}")
    lines.append("    \\centering")
    lines.extend(_build_section(results_left, specs, title_left))
    lines.append("  \\end{minipage}")
    lines.append("  \\hfill")
    lines.append("  \\begin{minipage}{0.49\\textwidth}")
    lines.append("    \\centering")
    lines.extend(_build_section(results_right, specs, title_right))
    lines.append("  \\end{minipage}")
    lines.append("\\end{table*}")
    if note:
        escaped = _latex_escape(note)
        lines.append("")
        lines.append(f"\\par\\noindent\\footnotesize\\textit{{{escaped}}}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX comparison table from results CSVs."
    )
    parser.add_argument(
        "--results-8",
        type=Path,
        default=Path("results_8"),
        help="Results directory for the 8x8 table (contains X1/X5/X10).",
    )
    parser.add_argument(
        "--results-16",
        type=Path,
        default=Path("results_16"),
        help="Results directory for the 16x16 table (contains X1/X5/X10).",
    )
    parser.add_argument(
        "--specs",
        nargs="*",
        default=SPECS_DEFAULT,
        help="Spec labels to include (e.g., X1 X5 X10).",
    )
    parser.add_argument(
        "--title-left",
        type=str,
        default="8$\\times$8",
        help="Title text for the left table.",
    )
    parser.add_argument(
        "--title-right",
        type=str,
        default="16$\\times$16",
        help="Title text for the right table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the LaTeX table.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note to append after the table (set to empty to omit).",
    )
    args = parser.parse_args()

    table = _render_table(
        args.results_8,
        args.results_16,
        args.specs,
        args.title_left,
        args.title_right,
        args.note.strip() if args.note else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"Wrote table to {args.output}")


if __name__ == "__main__":
    main()
