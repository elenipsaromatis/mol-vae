"""LD50 noise-robustness comparison: noise-free vs. 10% vs. 20% relative noise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_DIR_PATTERN = re.compile(r"seed_(\d+)")
METHOD_DIRS = ("ucb", "pareto", "ei", "random")
METHOD_DIR_TO_LABEL = {"ucb": "UCB", "pareto": "Pareto", "ei": "EI", "random": "Random"}
METHOD_ORDER = ["UCB", "Pareto", "EI", "Random"]

NOISE_LEVEL_COLORS = {
    "0%": "tab:gray",
    "10%": "tab:blue",
    "20%": "tab:red",
}


def find_trace_files(results_dir: Path) -> list[tuple[Path, int, str]]:
    records: list[tuple[Path, int, str]] = []
    for method_dir in METHOD_DIRS:
        for path in sorted(results_dir.glob(f"seed_*/{method_dir}/bo_trace.csv")):
            seed_match = SEED_DIR_PATTERN.fullmatch(path.parent.parent.name)
            if seed_match is None:
                continue
            records.append((path, int(seed_match.group(1)), method_dir))
    return records


def load_trace(path: Path, seed: int, method_dir: str, noise_level: str) -> pd.DataFrame:
    trace = pd.read_csv(path)

    if "best_ld50_true_after" not in trace.columns:
        trace["best_ld50_true_after"] = trace["best_ld50_after"]
    if "initial_best_ld50_true" not in trace.columns:
        trace["initial_best_ld50_true"] = trace["initial_best_ld50"]

    trace = trace.copy()
    trace["seed"] = seed
    trace["method"] = METHOD_DIR_TO_LABEL[method_dir]
    trace["noise_level"] = noise_level
    trace["evaluation"] = trace["iteration"].astype(int) + 1
    return trace


def load_all_traces(results_dir: Path, noise_level: str) -> pd.DataFrame:
    records = find_trace_files(results_dir)
    if not records:
        raise FileNotFoundError(
            f"No bo_trace.csv files found under {results_dir}/seed_*/"
            f"{{{','.join(METHOD_DIRS)}}}/"
        )
    frames = [
        load_trace(path, seed, method_dir, noise_level)
        for path, seed, method_dir in records
    ]
    return pd.concat(frames, ignore_index=True)


def build_true_convergence_series(combined: pd.DataFrame) -> pd.DataFrame:
    """Long-form best-observed-*true*-LD50 series per (noise_level, method,
    seed), including evaluation 0 for the shared initial sample."""
    rows = []
    for (noise_level, method, seed), group in combined.groupby(
        ["noise_level", "method", "seed"]
    ):
        group = group.sort_values("iteration")
        initial_best_true = float(group["initial_best_ld50_true"].iloc[0])

        rows.append(
            {
                "noise_level": noise_level,
                "method": method,
                "seed": seed,
                "evaluation": 0,
                "best_ld50_true": initial_best_true,
            }
        )
        for _, row in group.iterrows():
            rows.append(
                {
                    "noise_level": noise_level,
                    "method": method,
                    "seed": seed,
                    "evaluation": int(row["evaluation"]),
                    "best_ld50_true": float(row["best_ld50_true_after"]),
                }
            )

    return pd.DataFrame(rows)


def summarize_across_seeds(series: pd.DataFrame) -> pd.DataFrame:
    summary = (
        series.groupby(["noise_level", "method", "evaluation"])["best_ld50_true"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def ordered_noise_levels(present: pd.Series) -> list[str]:
    order = ["0%", "10%", "20%"]
    present_set = set(present.unique())
    ordered = [level for level in order if level in present_set]
    ordered += sorted(present_set - set(ordered))
    return ordered


def plot_noise_robustness(
    summary: pd.DataFrame, method: str, figures_dir: Path
) -> Path:
    method_summary = summary[summary["method"] == method]

    fig, ax = plt.subplots(figsize=(8, 5))

    for noise_level in ordered_noise_levels(method_summary["noise_level"]):
        level_summary = method_summary[
            method_summary["noise_level"] == noise_level
        ].sort_values("evaluation")

        color = NOISE_LEVEL_COLORS.get(noise_level)
        ax.plot(
            level_summary["evaluation"],
            level_summary["mean"],
            linewidth=2.0,
            label=f"{noise_level} noise",
            color=color,
        )
        ax.fill_between(
            level_summary["evaluation"],
            level_summary["mean"] - level_summary["std"],
            level_summary["mean"] + level_summary["std"],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Best observed true LD50")
    ax.set_title(f"LD50 Noise Robustness ({method})")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path = figures_dir / f"noise_robustness_{method.lower()}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BO convergence on true LD50 across noise-free, 10%, "
            "and 20% relative-noise runs."
        )
    )
    parser.add_argument("--baseline-dir", default="bo/results/baseline_1000")
    parser.add_argument("--noise10-dir", default="bo/results/noise_10pct")
    parser.add_argument("--noise20-dir", default="bo/results/noise_20pct")
    parser.add_argument(
        "--figures-dir", default="analysis/figures/noise_robustness"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["UCB", "Pareto", "EI"],
        help="Method labels to produce a robustness figure for.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    inputs = {
        "0%": Path(args.baseline_dir),
        "10%": Path(args.noise10_dir),
        "20%": Path(args.noise20_dir),
    }

    frames = []
    for noise_level, results_dir in inputs.items():
        try:
            frames.append(load_all_traces(results_dir, noise_level))
        except FileNotFoundError as error:
            print(f"Error: {error}", file=sys.stderr)
            raise SystemExit(1) from error

    combined = pd.concat(frames, ignore_index=True)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    series = build_true_convergence_series(combined)
    summary = summarize_across_seeds(series)

    methods_found = sorted(set(summary["method"].unique()) & set(args.strategies))
    missing = set(args.strategies) - set(summary["method"].unique())
    if missing:
        print(
            f"Warning: no traces found for strategies {sorted(missing)}; "
            "skipping their figures.",
            file=sys.stderr,
        )

    figure_paths = [
        plot_noise_robustness(summary, method, figures_dir)
        for method in [m for m in METHOD_ORDER if m in methods_found]
    ]

    summary_path = figures_dir / "noise_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Loaded {len(combined)} trace rows across noise levels {list(inputs)}")
    print("\nFigures:")
    for path in figure_paths:
        print(f"  - {path}")
    print(f"\nSummary table: {summary_path}")


if __name__ == "__main__":
    main()
