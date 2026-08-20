"""Multi-seed comparison of latent-space UCB BO, Pareto BO, Pareto-Expert BO,
EI BO, and random search.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_DIR_TO_LABEL = {
    "ucb": "UCB",
    "pareto": "Pareto",
    "pareto_expert": "Pareto-Expert",
    "ei": "EI",
    "random": "Random",
}
METHOD_ORDER = ["UCB", "Pareto", "Pareto-Expert", "EI", "Random"]

TOPN_RECOVERY_VALUES = [10, 20, 30, 50]

REQUIRED_COLUMNS = [
    "seed",
    "iteration",
    "selected_recipe_id",
    "selected_smiles",
    "selected_ld50_raw",
    "selected_true_ld50_rank",
    "selected_is_true_top_10",
    "initial_best_ld50",
    "best_ld50_before",
    "best_ld50_after",
    "improved_best",
]

SELECTED_MOLECULES_COLUMNS = [
    "seed",
    "method",
    "iteration",
    "selected_recipe_id",
    "selected_smiles",
    "selected_ld50_raw",
    "selected_true_ld50_rank",
    "selected_is_true_top_10",
    "initial_best_ld50",
    "best_ld50_before",
    "best_ld50_after",
    "improved_best",
]

SEED_DIR_PATTERN = re.compile(r"seed_(\d+)")

TIE_ATOL = 1e-9

ALL_METHOD_DIRS = ("ucb", "pareto", "pareto_expert", "ei", "random")


def find_trace_files(
    results_dir: Path, method_dirs: tuple[str, ...] = ALL_METHOD_DIRS
) -> list[tuple[Path, int, str]]:
    """Find every `seed_*/<method_dir>/bo_trace.csv` under `results_dir`,
    for each `method_dir` in `method_dirs` (default: all known strategies)."""
    records: list[tuple[Path, int, str]] = []

    for method_dir in method_dirs:
        for path in sorted(results_dir.glob(f"seed_*/{method_dir}/bo_trace.csv")):
            seed_match = SEED_DIR_PATTERN.fullmatch(path.parent.parent.name)
            if seed_match is None:
                continue
            records.append((path, int(seed_match.group(1)), method_dir))

    return records


def load_trace(path: Path, seed: int, method_dir: str) -> pd.DataFrame:
    trace = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in trace.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(missing)}"
        )

    trace = trace.copy()
    trace["seed"] = seed
    trace["method"] = METHOD_DIR_TO_LABEL[method_dir]
    trace["evaluation"] = trace["iteration"].astype(int) + 1

    return trace


def load_all_traces(
    results_dir: Path, exclude_methods: tuple[str, ...] = ()
) -> pd.DataFrame:
    method_dirs = tuple(m for m in ALL_METHOD_DIRS if m not in exclude_methods)
    records = find_trace_files(results_dir, method_dirs=method_dirs)

    if not records:
        raise FileNotFoundError(
            "No bo_trace.csv files found under "
            f"{results_dir}/seed_*/{{{','.join(method_dirs)}}}/"
        )

    frames = [load_trace(path, seed, method_dir) for path, seed, method_dir in records]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["seed", "method", "iteration"]).reset_index(
        drop=True
    )

    return combined



def build_convergence_series(combined: pd.DataFrame) -> pd.DataFrame:
    """Long-form best-observed-LD50 series per (method, seed), including
    evaluation 0 for the shared initial sample."""
    rows = []

    for (method, seed), group in combined.groupby(["method", "seed"]):
        group = group.sort_values("iteration")
        initial_best = float(group["initial_best_ld50"].iloc[0])

        rows.append(
            {"method": method, "seed": seed, "evaluation": 0, "best_ld50": initial_best}
        )
        for _, row in group.iterrows():
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "evaluation": int(row["evaluation"]),
                    "best_ld50": float(row["best_ld50_after"]),
                }
            )

    return pd.DataFrame(rows)


def build_topn_series(combined: pd.DataFrame, n: int) -> pd.DataFrame:
    """Long-form cumulative unique-true-top-n-found series per (method, seed),
    including evaluation 0 (nothing found yet before any selection).

    "True top n" is derived from `selected_true_ld50_rank` directly (already
    computed by bo/run_bo.py for every selection), so this works for any n
    without needing a dedicated `selected_is_true_top_n` column per n.
    """
    rows = []

    for (method, seed), group in combined.groupby(["method", "seed"]):
        group = group.sort_values("iteration")

        rows.append(
            {"method": method, "seed": seed, "evaluation": 0, "cumulative_found": 0}
        )

        seen: set = set()
        count = 0
        for _, row in group.iterrows():
            rank = row["selected_true_ld50_rank"]
            is_top_n = pd.notna(rank) and rank <= n

            if is_top_n and pd.notna(row["selected_recipe_id"]):
                molecule_id = row["selected_recipe_id"]
                if molecule_id not in seen:
                    seen.add(molecule_id)
                    count += 1

            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "evaluation": int(row["evaluation"]),
                    "cumulative_found": count,
                }
            )

    return pd.DataFrame(rows)


def summarize_across_seeds(
    series: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    """Mean/std of `value_col` across seeds, grouped by method and evaluation."""
    summary = (
        series.groupby(["method", "evaluation"])[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def build_first_found_series(
    combined: pd.DataFrame,
    target: float,
    atol: float = 1e-6,
    objective: str = "maximize",
) -> pd.DataFrame:
    """Per (method, seed): the first evaluation at which the running-best
    LD50 reaches `target` (within `atol`), or np.inf if never reached.

    "Reaches" means >= target for maximize (values climb toward the
    target) and <= target for minimize (values fall toward it).
    """
    series = build_convergence_series(combined)
    rows = []

    for (method, seed), group in series.groupby(["method", "seed"]):
        group = group.sort_values("evaluation")
        if objective == "minimize":
            found = group.loc[group["best_ld50"] <= target + atol, "evaluation"]
        else:
            found = group.loc[group["best_ld50"] >= target - atol, "evaluation"]
        first_evaluation = float(found.iloc[0]) if not found.empty else np.inf
        rows.append(
            {"method": method, "seed": seed, "first_evaluation": first_evaluation}
        )

    return pd.DataFrame(rows)


def ordered_methods(present: pd.Series) -> list[str]:
    present_set = set(present.unique())
    ordered = [m for m in METHOD_ORDER if m in present_set]
    ordered += sorted(present_set - set(ordered))
    return ordered



def plot_mean_convergence(combined: pd.DataFrame, figures_dir: Path) -> Path:
    series = build_convergence_series(combined)
    summary = summarize_across_seeds(series, "best_ld50")

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ordered_methods(summary["method"]):
        method_summary = summary[summary["method"] == method].sort_values(
            "evaluation"
        )
        ax.plot(
            method_summary["evaluation"],
            method_summary["mean"],
            linewidth=2.0,
            label=method,
        )
        ax.fill_between(
            method_summary["evaluation"],
            method_summary["mean"] - method_summary["std"],
            method_summary["mean"] + method_summary["std"],
            alpha=0.2,
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Best observed pLD50")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path = figures_dir / "mean_best_ld50_convergence.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_fraction_found_top(
    combined: pd.DataFrame,
    figures_dir: Path,
    atol: float = 1e-6,
    objective: str = "maximize",
) -> Path:
    """Cumulative fraction of seeds (per method) that have discovered the
    globally best-observed LD50 molecule by each iteration.

    Explains step-jumps in the mean/median convergence plots: a jump there
    happens exactly when one seed's curve here crosses from "not found" to
    "found". "Best" is the max of best_ld50_after for maximize, the min
    for minimize.
    """
    if objective == "minimize":
        target = float(combined["best_ld50_after"].min())
    else:
        target = float(combined["best_ld50_after"].max())
    first_found = build_first_found_series(combined, target, atol=atol, objective=objective)
    max_evaluation = int(combined["evaluation"].max())

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ordered_methods(first_found["method"]):
        method_rows = first_found[first_found["method"] == method]
        total = len(method_rows)
        finite_evals = np.sort(
            method_rows.loc[
                np.isfinite(method_rows["first_evaluation"]), "first_evaluation"
            ].to_numpy()
        )

        x = np.concatenate([[0], finite_evals, [max_evaluation]])
        cumulative_counts = np.concatenate(
            [[0], np.arange(1, len(finite_evals) + 1), [len(finite_evals)]]
        )
        y = cumulative_counts / total

        ax.step(
            x,
            y,
            where="post",
            linewidth=2.0,
            label=f"{method} ({len(finite_evals)}/{total} seeds)",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fraction of seeds that found it")
    ax.set_ylim(-0.05, 1.05)
    # Target value shown as a small in-axes annotation rather than a
    # title, since the caption already states what this plot is; the
    # exact target pLD50 is the one piece of per-dataset information not
    # otherwise captured by the caption.
    ax.text(
        0.02,
        0.98,
        f"Target pLD50 = {target:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.7"),
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path = figures_dir / "fraction_seeds_found_global_best.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_final_best_by_method(combined: pd.DataFrame, figures_dir: Path) -> Path:
    finals = (
        combined.sort_values("iteration")
        .groupby(["method", "seed"])["best_ld50_after"]
        .last()
        .reset_index()
    )

    methods = ordered_methods(finals["method"])
    data = [
        finals.loc[finals["method"] == method, "best_ld50_after"].to_numpy()
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(1, len(methods) + 1)

    ax.boxplot(data, positions=positions, tick_labels=methods, showfliers=False)

    rng = np.random.default_rng(0)
    for position, values in zip(positions, data):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            alpha=0.7,
            zorder=3,
        )

    ax.set_xlabel("Selection method")
    ax.set_ylabel("Final best observed pLD50")
    ax.grid(True, alpha=0.25, axis="y")

    output_path = figures_dir / "final_best_ld50_by_method.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_topn_recovery(combined: pd.DataFrame, figures_dir: Path, n: int) -> Path:
    series = build_topn_series(combined, n)
    summary = summarize_across_seeds(series, "cumulative_found")

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ordered_methods(summary["method"]):
        method_summary = summary[summary["method"] == method].sort_values(
            "evaluation"
        )
        ax.plot(
            method_summary["evaluation"],
            method_summary["mean"],
            linewidth=2.0,
            label=method,
        )
        ax.fill_between(
            method_summary["evaluation"],
            method_summary["mean"] - method_summary["std"],
            method_summary["mean"] + method_summary["std"],
            alpha=0.2,
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(f"Cumulative true top-{n} molecules found")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path = figures_dir / f"top{n}_recovery_comparison.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_selected_ld50_distribution(combined: pd.DataFrame, figures_dir: Path) -> Path:
    methods = ordered_methods(combined["method"])
    data = [
        combined.loc[combined["method"] == method, "selected_ld50_raw"].to_numpy()
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=methods, showfliers=True)

    ax.set_xlabel("Selection method")
    ax.set_ylabel("Selected molecule pLD50")
    ax.grid(True, alpha=0.25, axis="y")

    output_path = figures_dir / "selected_ld50_distribution.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_mean_selected_by_iteration(combined: pd.DataFrame, figures_dir: Path) -> Path:
    summary = (
        combined.groupby(["method", "evaluation"])["selected_ld50_raw"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ordered_methods(summary["method"]):
        method_summary = summary[summary["method"] == method].sort_values(
            "evaluation"
        )
        ax.plot(
            method_summary["evaluation"],
            method_summary["mean"],
            linewidth=2.0,
            label=method,
        )
        ax.fill_between(
            method_summary["evaluation"],
            method_summary["mean"] - method_summary["std"],
            method_summary["mean"] + method_summary["std"],
            alpha=0.2,
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean selected pLD50")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path = figures_dir / "mean_selected_ld50_by_iteration.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path



def build_per_seed_summary(
    combined: pd.DataFrame, objective: str = "maximize"
) -> pd.DataFrame:
    records = []

    for (seed, method), group in combined.groupby(["seed", "method"]):
        group = group.sort_values("iteration")

        initial_best_ld50 = float(group["initial_best_ld50"].iloc[0])
        final_best_ld50 = float(group["best_ld50_after"].iloc[-1])

        best_selected_ld50 = (
            float(group["selected_ld50_raw"].min())
            if objective == "minimize"
            else float(group["selected_ld50_raw"].max())
        )

        record = {
            "seed": seed,
            "method": method,
            "n_iterations": int(len(group)),
            "initial_best_ld50": initial_best_ld50,
            "final_best_ld50": final_best_ld50,
            "improvement_over_initial": final_best_ld50 - initial_best_ld50,
            "best_selected_ld50": best_selected_ld50,
            "best_true_rank_selected": float(
                group["selected_true_ld50_rank"].min()
            ),
            "mean_selected_ld50": float(group["selected_ld50_raw"].mean()),
            "median_selected_ld50": float(group["selected_ld50_raw"].median()),
        }

        for n in TOPN_RECOVERY_VALUES:
            topn_rows = group.loc[group["selected_true_ld50_rank"] <= n]
            record[f"top{n}_molecules_found"] = int(
                topn_rows["selected_recipe_id"].dropna().nunique()
            )
            record[f"first_top{n}_iteration"] = (
                float(topn_rows["evaluation"].iloc[0])
                if not topn_rows.empty
                else np.nan
            )

        record["number_of_best_improvements"] = int(
            group["improved_best"].fillna(False).sum()
        )

        records.append(record)

    per_seed = pd.DataFrame(records).sort_values(["method", "seed"]).reset_index(
        drop=True
    )
    return per_seed


def build_method_summary(
    per_seed: pd.DataFrame, objective: str = "maximize"
) -> pd.DataFrame:
    random_finals = (
        per_seed.loc[per_seed["method"] == "Random"]
        .set_index("seed")["final_best_ld50"]
    )

    records = []

    for method in ordered_methods(per_seed["method"]):
        group = per_seed[per_seed["method"] == method]

        record = {
            "method": method,
            "n_seeds": int(len(group)),
            "mean_final_best_ld50": float(group["final_best_ld50"].mean()),
            "std_final_best_ld50": float(group["final_best_ld50"].std()),
            "median_final_best_ld50": float(group["final_best_ld50"].median()),
            "mean_improvement_over_initial": float(
                group["improvement_over_initial"].mean()
            ),
            "std_improvement_over_initial": float(
                group["improvement_over_initial"].std()
            ),
            "mean_best_selected_ld50": float(group["best_selected_ld50"].mean()),
            "mean_selected_ld50": float(group["mean_selected_ld50"].mean()),
            "std_selected_ld50": float(group["mean_selected_ld50"].std()),
        }

        for n in TOPN_RECOVERY_VALUES:
            record[f"mean_top{n}_molecules_found"] = float(
                group[f"top{n}_molecules_found"].mean()
            )
            record[f"std_top{n}_molecules_found"] = float(
                group[f"top{n}_molecules_found"].std()
            )
            record[f"mean_first_top{n}_iteration"] = float(
                group[f"first_top{n}_iteration"].mean(skipna=True)
            )

        record["mean_number_of_best_improvements"] = float(
            group["number_of_best_improvements"].mean()
        )

        if method == "Random":
            record.update(
                {
                    "wins_vs_random": np.nan,
                    "ties_vs_random": np.nan,
                    "losses_vs_random": np.nan,
                    "success_rate_beating_random": np.nan,
                    "mean_final_difference_vs_random": np.nan,
                }
            )
        else:
            matched = (
                group.set_index("seed")["final_best_ld50"]
                .to_frame("method_final")
                .join(random_finals.rename("random_final"), how="inner")
            )

            if matched.empty:
                record.update(
                    {
                        "wins_vs_random": np.nan,
                        "ties_vs_random": np.nan,
                        "losses_vs_random": np.nan,
                        "success_rate_beating_random": np.nan,
                        "mean_final_difference_vs_random": np.nan,
                    }
                )
            else:
                diff = matched["method_final"] - matched["random_final"]
                tie_mask = np.isclose(diff, 0.0, atol=TIE_ATOL, rtol=0.0)
                # For "maximize" a higher final value than Random is a win;
                # for "minimize" a lower final value than Random is a win.
                beats_random = (diff < 0) if objective == "minimize" else (diff > 0)
                win_mask = beats_random & ~tie_mask
                loss_mask = ~beats_random & ~tie_mask
                n_matched = len(matched)

                record.update(
                    {
                        "wins_vs_random": int(win_mask.sum()),
                        "ties_vs_random": int(tie_mask.sum()),
                        "losses_vs_random": int(loss_mask.sum()),
                        "success_rate_beating_random": float(
                            win_mask.sum() / n_matched
                        ),
                        "mean_final_difference_vs_random": float(diff.mean()),
                    }
                )

        records.append(record)

    method_summary = pd.DataFrame(records)
    return method_summary


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table writer (no external deps)."""

    def format_cell(value) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows]) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare latent-space UCB BO, Pareto BO, Pareto-Expert BO, EI "
            "BO, and random search across all seeds in a bo/run_bo.py "
            "results directory."
        )
    )

    parser.add_argument(
        "--results-dir",
        default="bo/results/baseline_full",
        help=(
            "Directory containing "
            "seed_*/{ucb,pareto,pareto_expert,ei,random}/bo_trace.csv"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        default="analysis/figures/bo_baseline_comparison",
        help="Directory to write comparison figures into.",
    )
    parser.add_argument(
        "--tables-dir",
        default="analysis/tables/bo_baseline_comparison",
        help="Directory to write comparison tables into.",
    )
    parser.add_argument(
        "--exclude-methods",
        choices=ALL_METHOD_DIRS,
        nargs="+",
        default=[],
        help=(
            "Selection strategy dirs to leave out of this comparison run "
            "even if present under --results-dir, e.g. "
            "'--exclude-methods pareto_expert' to compare only the "
            "original ucb/pareto/ei/random quartet. Does not delete or "
            "modify anything under --results-dir -- write to a different "
            "--figures-dir/--tables-dir than a prior run over the same "
            "results-dir if you want to keep both comparisons around."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=["maximize", "minimize"],
        default="maximize",
        help=(
            "Must match the --objective the results-dir was run with "
            "(bo/run_bo.py's --objective). Controls which direction counts "
            "as 'best' when finding the global-best molecule and each "
            "seed's best selected LD50. Default 'maximize' preserves "
            "behaviour for pre-existing (maximize-objective) results dirs."
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)

    try:
        combined = load_all_traces(
            results_dir, exclude_methods=tuple(args.exclude_methods)
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    seeds_found = sorted(combined["seed"].unique().tolist())
    methods_found = ordered_methods(combined["method"])

    selected_molecules = combined[SELECTED_MOLECULES_COLUMNS].sort_values(
        ["seed", "method", "iteration"]
    )
    selected_molecules_path = tables_dir / "selected_molecules.csv"
    selected_molecules.to_csv(selected_molecules_path, index=False)

    figure_paths = [
        plot_mean_convergence(combined, figures_dir),
        plot_fraction_found_top(combined, figures_dir, objective=args.objective),
        plot_final_best_by_method(combined, figures_dir),
        *[
            plot_topn_recovery(combined, figures_dir, n)
            for n in TOPN_RECOVERY_VALUES
        ],
        plot_selected_ld50_distribution(combined, figures_dir),
        plot_mean_selected_by_iteration(combined, figures_dir),
    ]

    per_seed_summary = build_per_seed_summary(combined, objective=args.objective)
    per_seed_path = tables_dir / "per_seed_summary.csv"
    per_seed_summary.to_csv(per_seed_path, index=False)

    method_summary = build_method_summary(per_seed_summary, objective=args.objective)
    method_summary_path = tables_dir / "method_summary.csv"
    method_summary.to_csv(method_summary_path, index=False)

    method_summary_md_path = tables_dir / "method_summary.md"
    method_summary_md_path.write_text(dataframe_to_markdown(method_summary))

    table_paths = [
        per_seed_path,
        method_summary_path,
        method_summary_md_path,
        selected_molecules_path,
    ]

    print(f"Loaded {len(combined)} trace rows from {results_dir}")
    print(f"Seeds found: {seeds_found}")
    print(f"Methods found: {methods_found}")

    print("\nFigures:")
    for path in figure_paths:
        print(f"  - {path}")

    print("\nTables:")
    for path in table_paths:
        print(f"  - {path}")

    print("\nMethod summary:")
    print(method_summary.to_string(index=False))


if __name__ == "__main__":
    main()
