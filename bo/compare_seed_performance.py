"""Compare Bayesian optimisation performance across multiple reproducible
seeds, produced by `bo/run_bo.py`'s multi-seed sweep.

Reads bo/results/seed_{seed}/{strategy}/bo_trace.csv for every requested
(seed, strategy) pair, validates completeness, aggregates the running-best
selected LD50 across seeds per strategy/iteration, and produces a paired
UCB vs. Pareto final-performance comparison -- paired because both
strategies were run from the exact same Latin-Hypercube initial sample for
a given seed.

Does not read, modify, or overwrite anything under an individual
seed_{seed}/{strategy}/ folder; every output goes to --output-dir.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "bo" / "results"
DEFAULT_OUTPUT_DIR = ROOT / "bo" / "results" / "seed_comparison"

SEED_DIR_PATTERN = re.compile(r"seed_(\d+)")

STRATEGY_COLORS = {"ucb": "tab:blue", "pareto": "tab:orange"}
STRATEGY_LABELS = {"ucb": "UCB", "pareto": "Pareto Front"}


def discover_seeds(results_dir: Path) -> list[int]:
    """Find every seed_{N} directory directly under results_dir."""
    seeds = []
    for entry in sorted(results_dir.glob("seed_*")):
        if not entry.is_dir():
            continue
        match = SEED_DIR_PATTERN.fullmatch(entry.name)
        if match:
            seeds.append(int(match.group(1)))
    return sorted(seeds)


def trace_path_for(results_dir: Path, seed: int, strategy: str) -> Path:
    return results_dir / f"seed_{seed}" / strategy / "bo_trace.csv"


def load_one_trace(
    path: Path,
    seed: int,
    strategy: str,
    iteration_col: str,
    metric_col: str,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load and validate one bo_trace.csv.

    Returns (frame, None) on success or (None, problem_description) if the
    file is missing, malformed, or has non-contiguous iterations. Never
    raises -- the caller decides whether a problem is fatal (--strict) or a
    skip-with-warning.
    """
    if not path.exists():
        return None, f"trace not found at {path}"

    frame = pd.read_csv(path)

    required = {iteration_col, metric_col}
    missing = required.difference(frame.columns)
    if missing:
        return None, (
            f"missing required column(s) {sorted(missing)} in {path}"
        )

    frame = frame.sort_values(iteration_col).reset_index(drop=True)

    iterations = frame[iteration_col].to_numpy()
    expected_iterations = np.arange(len(frame))
    if not np.array_equal(iterations, expected_iterations):
        return None, (
            f"'{iteration_col}' values are not a contiguous 0..N-1 range "
            f"in {path} (got {iterations.tolist()})"
        )

    frame = frame.copy()
    # Path-derived, not trusting whatever the CSV's own seed/selection
    # columns say -- the folder layout is the source of truth here.
    frame["seed"] = seed
    frame["strategy"] = strategy
    frame[f"running_best_{metric_col}"] = frame[metric_col].cummax()

    return frame, None


def load_all_traces(
    results_dir: Path,
    seeds: list[int],
    strategies: list[str],
    iteration_col: str,
    metric_col: str,
    strict: bool,
) -> pd.DataFrame:
    """Load every requested (seed, strategy) trace, drop incomplete ones
    (or raise if --strict), and return one combined long-format frame."""
    loaded: dict[tuple[int, str], pd.DataFrame] = {}

    for seed in seeds:
        for strategy in strategies:
            path = trace_path_for(results_dir, seed, strategy)
            frame, problem = load_one_trace(
                path, seed, strategy, iteration_col, metric_col
            )
            if problem is not None:
                message = f"seed {seed} [{strategy}]: {problem}"
                if strict:
                    raise ValueError(f"Rejected under --strict: {message}")
                print(f"Warning: skipping {message}", file=sys.stderr)
                continue
            loaded[(seed, strategy)] = frame

    if not loaded:
        raise ValueError("No usable traces were loaded; nothing to compare.")

    lengths = [len(frame) for frame in loaded.values()]
    expected_length = int(pd.Series(lengths).mode().iloc[0])

    complete: dict[tuple[int, str], pd.DataFrame] = {}
    for (seed, strategy), frame in loaded.items():
        if len(frame) != expected_length:
            message = (
                f"seed {seed} [{strategy}]: {len(frame)} iterations, "
                f"expected {expected_length} (the most common length "
                "across all loaded traces)"
            )
            if strict:
                raise ValueError(
                    f"Incomplete run rejected under --strict: {message}"
                )
            print(f"Warning: incomplete run skipped: {message}", file=sys.stderr)
            continue
        complete[(seed, strategy)] = frame

    if not complete:
        raise ValueError("No complete traces remain after validation.")

    requested = len(seeds) * len(strategies)
    print(
        f"Loaded {len(complete)} complete (seed, strategy) trace(s) out of "
        f"{requested} requested."
    )

    return pd.concat(complete.values(), ignore_index=True)


def compute_iteration_summary(
    combined: pd.DataFrame,
    iteration_col: str,
    running_best_col: str,
) -> pd.DataFrame:
    """Per strategy/iteration: n_seeds, mean, std, median, min, max, sem,
    and a 95% CI on the mean using the Student t distribution."""
    records = []

    for (strategy, iteration), group in combined.groupby(
        ["strategy", iteration_col]
    ):
        values = group[running_best_col].to_numpy(dtype=float)
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0

        if n > 1:
            t_critical = float(stats.t.ppf(0.975, df=n - 1))
            ci_half_width = t_critical * sem
        else:
            ci_half_width = 0.0

        records.append(
            {
                "strategy": strategy,
                iteration_col: iteration,
                "n_seeds": n,
                "mean": mean,
                "std": std,
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "sem": sem,
                "ci95_lower": mean - ci_half_width,
                "ci95_upper": mean + ci_half_width,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(["strategy", iteration_col])
        .reset_index(drop=True)
    )


def compute_final_performance(
    combined: pd.DataFrame,
    iteration_col: str,
    running_best_col: str,
) -> pd.DataFrame:
    """One row per (seed, strategy): the running-best metric value at that
    run's final iteration -- the best molecule value that seed/strategy
    found overall."""
    records = []

    for (seed, strategy), group in combined.groupby(["seed", "strategy"]):
        final_row = group.loc[group[iteration_col].idxmax()]
        records.append(
            {
                "seed": seed,
                "strategy": strategy,
                "final_iteration": int(final_row[iteration_col]),
                "final_value": float(final_row[running_best_col]),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(["strategy", "seed"])
        .reset_index(drop=True)
    )


def compute_paired_comparison(
    final_df: pd.DataFrame,
) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Pair UCB vs. Pareto final values by seed (valid because both share
    the same initial sample for a given seed). Returns (None, None) if
    either strategy is entirely absent from final_df."""
    pivot = final_df.pivot(index="seed", columns="strategy", values="final_value")

    if "ucb" not in pivot.columns or "pareto" not in pivot.columns:
        return None, None

    paired = pivot.dropna(subset=["ucb", "pareto"]).copy()
    if paired.empty:
        return None, None

    paired["difference_ucb_minus_pareto"] = paired["ucb"] - paired["pareto"]
    paired = paired.reset_index().rename(
        columns={"ucb": "ucb_final_value", "pareto": "pareto_final_value"}
    )

    stats_summary = None
    if len(paired) >= 2:
        diffs = paired["difference_ucb_minus_pareto"].to_numpy(dtype=float)
        t_stat, p_value = stats.ttest_rel(
            paired["ucb_final_value"], paired["pareto_final_value"]
        )
        stats_summary = {
            "n_pairs": int(len(paired)),
            "mean_difference": float(np.mean(diffs)),
            "median_difference": float(np.median(diffs)),
            "paired_ttest_statistic": float(t_stat),
            "paired_ttest_pvalue": float(p_value),
        }

    return paired, stats_summary


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Matches the save-figure convention used by analysis/analyze_bo.py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def plot_convergence_mean_ci(
    summary: pd.DataFrame,
    iteration_col: str,
    metric_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    n_seeds_by_label = {}

    for strategy, group in summary.groupby("strategy"):
        group = group.sort_values(iteration_col)
        color = STRATEGY_COLORS.get(strategy)
        label = STRATEGY_LABELS.get(strategy, strategy)
        n_seeds_by_label[label] = int(group["n_seeds"].iloc[0])

        ax.plot(
            group[iteration_col],
            group["mean"],
            linewidth=2.2,
            color=color,
            label=label,
        )
        ax.fill_between(
            group[iteration_col],
            group["ci95_lower"],
            group["ci95_upper"],
            color=color,
            alpha=0.2,
        )

    caption = ", ".join(
        f"{label}: n={n} seeds" for label, n in n_seeds_by_label.items()
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best observed LD50")
    ax.set_title(f"BO Convergence Across Seeds (mean ± 95% CI)\n{caption}")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    save_figure(fig, output_path)


def plot_convergence_individual_seeds(
    combined: pd.DataFrame,
    summary: pd.DataFrame,
    iteration_col: str,
    running_best_col: str,
    metric_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, group in combined.groupby("strategy"):
        color = STRATEGY_COLORS.get(strategy)
        label = STRATEGY_LABELS.get(strategy, strategy)

        for _seed, seed_group in group.groupby("seed"):
            seed_group = seed_group.sort_values(iteration_col)
            ax.plot(
                seed_group[iteration_col],
                seed_group[running_best_col],
                color=color,
                alpha=0.25,
                linewidth=0.8,
            )

        mean_group = summary.loc[summary["strategy"] == strategy].sort_values(
            iteration_col
        )
        ax.plot(
            mean_group[iteration_col],
            mean_group["mean"],
            color=color,
            linewidth=2.5,
            label=f"{label} (mean)",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best observed LD50")
    ax.set_title("BO Convergence: Individual Seeds vs. Mean")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    save_figure(fig, output_path)


def plot_final_performance(
    final_df: pd.DataFrame,
    metric_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    strategies_present = [
        strategy
        for strategy in ("ucb", "pareto")
        if strategy in final_df["strategy"].unique()
    ]
    positions = np.arange(1, len(strategies_present) + 1)

    box_data = [
        final_df.loc[final_df["strategy"] == strategy, "final_value"].to_numpy()
        for strategy in strategies_present
    ]

    boxplot = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
    )

    for patch, strategy in zip(boxplot["boxes"], strategies_present):
        color = STRATEGY_COLORS.get(strategy, "tab:grey")
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for position, strategy in zip(positions, strategies_present):
        values = final_df.loc[
            final_df["strategy"] == strategy, "final_value"
        ].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            color=STRATEGY_COLORS.get(strategy, "black"),
            edgecolors="black",
            linewidths=0.4,
            s=40,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [STRATEGY_LABELS.get(strategy, strategy) for strategy in strategies_present]
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Final best observed LD50")
    ax.set_title(
        f"Final Performance by Seed (n={final_df['seed'].nunique()} seeds)"
    )
    ax.grid(True, alpha=0.3, axis="y")

    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Bayesian optimisation performance across multiple "
            "reproducible seeds (paired UCB vs. Pareto)."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Specific seeds to compare, e.g. '--seeds 11 22 33'. Default: "
            "auto-discover every seed_* folder under --results-dir."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["ucb", "pareto"],
        default=["ucb", "pareto"],
        help="Which strategies to include. Default: both.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Root of per-seed BO results. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write comparison outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Reject (raise) on any missing or incomplete trace instead of "
            "skipping it with a warning."
        ),
    )
    parser.add_argument(
        "--iteration-column",
        default="iteration",
        help=(
            "Iteration column name in bo_trace.csv. Override only if "
            "automatic detection is unreliable. Default: 'iteration'."
        ),
    )
    parser.add_argument(
        "--metric-column",
        default="selected_ld50_raw",
        help=(
            "Performance metric column in bo_trace.csv; its cumulative "
            "max is used as the convergence/performance curve. Override "
            "only if automatic detection is unreliable. Default: "
            "'selected_ld50_raw'."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not results_dir.exists():
        print(f"Error: --results-dir does not exist: {results_dir}", file=sys.stderr)
        raise SystemExit(1)

    seeds = args.seeds if args.seeds else discover_seeds(results_dir)
    if not seeds:
        print(f"Error: no seed_* folders found under {results_dir}", file=sys.stderr)
        raise SystemExit(1)

    strategies = args.strategies

    print(f"Results directory: {results_dir}")
    print(f"Seeds: {seeds}")
    print(f"Strategies: {strategies}")
    print(f"Output directory: {output_dir}")
    print(f"Iteration column: {args.iteration_column}")
    print(f"Metric column: {args.metric_column}")

    try:
        combined = load_all_traces(
            results_dir=results_dir,
            seeds=seeds,
            strategies=strategies,
            iteration_col=args.iteration_column,
            metric_col=args.metric_column,
            strict=args.strict,
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    output_dir.mkdir(parents=True, exist_ok=True)

    running_best_col = f"running_best_{args.metric_column}"
    metric_label = f"Running best {args.metric_column}"

    combined_path = output_dir / "combined_traces.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved table: {combined_path}")

    summary = compute_iteration_summary(
        combined, args.iteration_column, running_best_col
    )
    summary_path = output_dir / "summary_by_iteration.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved table: {summary_path}")

    final_df = compute_final_performance(
        combined, args.iteration_column, running_best_col
    )
    final_path = output_dir / "final_performance_by_seed.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Saved table: {final_path}")

    paired_df, paired_stats = compute_paired_comparison(final_df)
    if paired_df is not None:
        paired_path = output_dir / "paired_final_comparison.csv"
        paired_df.to_csv(paired_path, index=False)
        print(f"Saved table: {paired_path}")

        if paired_stats is not None:
            print(
                "Paired UCB vs. Pareto final-performance difference "
                f"(n={paired_stats['n_pairs']} seeds): "
                f"mean={paired_stats['mean_difference']:.4f}, "
                f"median={paired_stats['median_difference']:.4f}, "
                f"paired t-test p={paired_stats['paired_ttest_pvalue']:.4f}. "
                "This is exploratory with a small number of seeds -- "
                "treat the p-value as descriptive, not confirmatory."
            )
    else:
        print(
            "Skipped paired comparison: both 'ucb' and 'pareto' results "
            "are required (only "
            f"{sorted(final_df['strategy'].unique())} were loaded)."
        )

    plot_convergence_mean_ci(
        summary,
        args.iteration_column,
        metric_label,
        output_dir / "convergence_mean_ci.png",
    )
    plot_convergence_individual_seeds(
        combined,
        summary,
        args.iteration_column,
        running_best_col,
        metric_label,
        output_dir / "convergence_individual_seeds.png",
    )
    plot_final_performance(
        final_df,
        metric_label,
        output_dir / "final_performance_comparison.png",
    )

    print("\nMulti-seed comparison complete.")


if __name__ == "__main__":
    main()
