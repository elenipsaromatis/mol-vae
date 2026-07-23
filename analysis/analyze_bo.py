from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES_DIR = ROOT / "analysis" / "figures"
DEFAULT_TABLES_DIR = ROOT / "analysis" / "tables"

SEED_DIR_PATTERN = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)

REQUIRED_COLUMNS = {
    "iteration",
    "selected_ld50_raw",
}

OPTIONAL_COLUMNS = {
    "selected_true_ld50_rank",
    "selected_is_true_top_10",
}


def detect_seed_and_strategy(
    path: Path,
) -> tuple[Optional[int], Optional[str]]:
    seed: Optional[int] = None
    strategy: Optional[str] = None

    for part in path.parts:
        if seed is None:
            match = SEED_DIR_PATTERN.fullmatch(part)
            if match:
                seed = int(match.group(1))
        if strategy is None and part.lower() in ("ucb", "pareto"):
            strategy = part.lower()

    return seed, strategy


def require_existing_path(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def resolve_traces_and_output_dirs(
    args: argparse.Namespace,
) -> tuple[list[str], Path, Path, Optional[int]]:
    modes_used = sum(
        [
            bool(args.trace),
            bool(args.ucb_trace or args.pareto_trace),
            bool(args.traces),
        ]
    )
    if modes_used == 0:
        raise ValueError(
            "No trace input provided. Use --trace, --ucb-trace together "
            "with --pareto-trace, or pass one or more trace paths "
            "positionally."
        )
    if modes_used > 1:
        raise ValueError(
            "Pass exactly one of: --trace, --ucb-trace/--pareto-trace, or "
            "positional trace paths -- not a combination."
        )

    if args.trace:
        trace_path = require_existing_path(args.trace, "--trace")
        seed, strategy = detect_seed_and_strategy(trace_path)
        output_dir = trace_path.parent / "analysis"

        print(f"Selected trace: {trace_path}")
        print(f"Detected seed: {seed if seed is not None else 'unknown'}")
        print(
            "Detected strategy: "
            f"{strategy if strategy is not None else 'unknown'}"
        )
        print(f"Output directory: {output_dir}")

        figures_dir = args.figures_dir or output_dir
        tables_dir = args.tables_dir or output_dir
        return [str(trace_path)], figures_dir, tables_dir, seed

    if args.ucb_trace or args.pareto_trace:
        if not (args.ucb_trace and args.pareto_trace):
            raise ValueError(
                "--ucb-trace and --pareto-trace must both be provided "
                "together for a paired comparison."
            )

        ucb_path = require_existing_path(args.ucb_trace, "--ucb-trace")
        pareto_path = require_existing_path(
            args.pareto_trace, "--pareto-trace"
        )

        ucb_seed, ucb_strategy = detect_seed_and_strategy(ucb_path)
        pareto_seed, pareto_strategy = detect_seed_and_strategy(pareto_path)

        if ucb_seed is not None and pareto_seed is not None:
            if ucb_seed != pareto_seed:
                raise ValueError(
                    f"--ucb-trace (seed {ucb_seed}) and --pareto-trace "
                    f"(seed {pareto_seed}) do not belong to the same seed."
                )
            seed = ucb_seed
        else:
            print(
                "Warning: could not detect a seed number from one or both "
                "trace paths; skipping the same-seed validation.",
                file=sys.stderr,
            )
            seed = None

        output_dir = ucb_path.parent.parent / "comparison"

        print(f"Selected UCB trace: {ucb_path}")
        print(f"Selected Pareto trace: {pareto_path}")
        print(f"Detected seed: {seed if seed is not None else 'unknown'}")
        print(
            "Detected strategies: "
            f"{ucb_strategy or 'unknown'}, {pareto_strategy or 'unknown'}"
        )
        print(f"Output directory: {output_dir}")

        figures_dir = args.figures_dir or output_dir
        tables_dir = args.tables_dir or output_dir
        return [str(ucb_path), str(pareto_path)], figures_dir, tables_dir, seed

    resolved_paths = [
        require_existing_path(raw, "trace path") for raw in args.traces
    ]
    figures_dir = args.figures_dir or DEFAULT_FIGURES_DIR
    tables_dir = args.tables_dir or DEFAULT_TABLES_DIR

    detected_seeds = {
        detected
        for detected, _strategy in (
            detect_seed_and_strategy(path) for path in resolved_paths
        )
        if detected is not None
    }
    seed = detected_seeds.pop() if len(detected_seeds) == 1 else None

    resolved = [str(path) for path in resolved_paths]
    print(f"Selected traces: {resolved}")
    print(f"Output directories: figures={figures_dir}, tables={tables_dir}")

    return resolved, figures_dir, tables_dir, seed


def normalise_label(path: Path) -> str:
    """Create a readable plot/table label from a trace filename.

    Falls back to the parent directory name when the filename itself is
    the generic "bo_trace.csv" (the per-seed/-strategy BO output layout
    names every trace file identically), since the stem alone would give
    every trace the same meaningless "bo" label.
    """
    label = path.stem
    label = re.sub(r"_trace$", "", label, flags=re.IGNORECASE)
    label = label.replace("_", " ").strip()

    if label.lower() == "bo":
        label = path.parent.name.replace("_", " ").strip()

    return label or path.stem


def make_unique_label(label: str, existing: Iterable[str]) -> str:
    """Avoid collisions when multiple files produce the same label."""
    if label not in existing:
        return label

    index = 2
    while f"{label} ({index})" in existing:
        index += 1
    return f"{label} ({index})"


def coerce_boolean(series: pd.Series) -> pd.Series:
    """Convert common boolean encodings to True/False."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)

    true_values = {"true", "1", "yes", "y", "t"}
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(true_values)
    )


def first_existing_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    """Return the first candidate column present in a DataFrame."""
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def load_traces(paths: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """Load and validate BO trace files."""
    traces: Dict[str, pd.DataFrame] = {}

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Trace file does not exist: {path}")

        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, received: {path}")

        trace = pd.read_csv(path)

        missing = REQUIRED_COLUMNS.difference(trace.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )

        trace = trace.copy()
        trace["iteration"] = pd.to_numeric(
            trace["iteration"], errors="coerce"
        )
        trace["selected_ld50_raw"] = pd.to_numeric(
            trace["selected_ld50_raw"], errors="coerce"
        )

        if "selected_true_ld50_rank" in trace.columns:
            trace["selected_true_ld50_rank"] = pd.to_numeric(
                trace["selected_true_ld50_rank"], errors="coerce"
            )

        if "selected_is_true_top_10" in trace.columns:
            trace["selected_is_true_top_10"] = coerce_boolean(
                trace["selected_is_true_top_10"]
            )

        trace = (
            trace.dropna(subset=["iteration", "selected_ld50_raw"])
            .sort_values("iteration")
            .reset_index(drop=True)
        )

        if trace.empty:
            raise ValueError(f"No valid rows remain after cleaning: {path}")

        label = make_unique_label(normalise_label(path), traces.keys())
        traces[label] = trace

        missing_optional = OPTIONAL_COLUMNS.difference(trace.columns)
        if missing_optional:
            print(
                f"Warning: {label} is missing optional column(s): "
                f"{', '.join(sorted(missing_optional))}. "
                "Some analyses will be skipped.",
                file=sys.stderr,
            )

    return traces


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure using consistent settings"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def compute_dataset_max_ld50() -> Optional[float]:
    """Compute the maximum raw LD50 value across the full BO candidate pool,
    straight from the dataset. No VAE checkpoint is needed for this --
    LD50 raw values don't depend on the model, only on the labels.

    Returns None (with a printed warning) if the dataset can't be loaded,
    so callers can still produce every other plot/table without it.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from data import build_dataloaders_multitask

        (
            train_loader,
            valid_loader,
            test_loader,
            _vocab_size,
            _char2idx,
            _idx2char,
            _train_labels,
            _sol_mean,
            _sol_std,
            ld50_mean,
            ld50_std,
        ) = build_dataloaders_multitask()
    except Exception as error:
        print(
            f"Warning: could not compute the dataset maximum LD50 ({error}); "
            "the convergence plot will omit that reference line.",
            file=sys.stderr,
        )
        return None

    ld50_mean = float(ld50_mean)
    ld50_std = float(ld50_std)

    max_raw = -np.inf
    for loader in (train_loader, valid_loader, test_loader):
        for _batch, labels in loader:
            ld50_raw = labels[:, 1].numpy() * ld50_std + ld50_mean
            max_raw = max(max_raw, float(ld50_raw.max()))

    return max_raw


def convergence_display_label(label: str) -> str:
    """Rename method tokens for the convergence plot legend."""
    label = re.sub(r"\bucb\b", "UCB", label, flags=re.IGNORECASE)
    label = re.sub(r"\bpareto\b", "Pareto Front", label, flags=re.IGNORECASE)
    return label


def plot_convergence(
    traces: Dict[str, pd.DataFrame],
    dataset_max_ld50: Optional[float],
    figures_dir: Path,
    seed: Optional[int] = None,
) -> None:
    """Plot cumulative best observed LD50 for every run."""

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, trace in traces.items():
        legend_label = convergence_display_label(label)
        trace = trace.sort_values("iteration").copy()

        observed_ld50 = pd.to_numeric(
            trace["selected_ld50_raw"],
            errors="coerce",
        )

        best_so_far = observed_ld50.cummax()

        ax.step(
            trace["iteration"],
            best_so_far,
            where="post",
            linewidth=2.2,
            label=legend_label,
        )

        best_discovered_ld50 = best_so_far.max()

        first_best_mask = np.isclose(
            best_so_far.to_numpy(dtype=float),
            best_discovered_ld50,
            equal_nan=False,
        )

        first_best_position = np.flatnonzero(first_best_mask)[0]
        first_best_iteration = trace["iteration"].iloc[first_best_position]

        # Highlight the first discovery of the best molecule for this run
        ax.scatter(
            first_best_iteration,
            best_discovered_ld50,
            s=75,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )

    if dataset_max_ld50 is not None:
        ax.axhline(
            y=dataset_max_ld50,
            linestyle="--",
            linewidth=1.8,
            color="black",
            label=f"Dataset maximum LD50 = {dataset_max_ld50:.3f}",
            zorder=1,
        )

    best_observed_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.7,
        label="Best Observed Value",
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(best_observed_handle)
    labels.append(best_observed_handle.get_label())

    ax.set_xlabel("Iterations")
    ax.set_ylabel("LD50 Values")
    ax.set_title(
        "Bayesian Optimization Convergence Plot"
        + (f" (seed {seed})" if seed is not None else "")
    )
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels)

    save_figure(
        fig,
        figures_dir / "bo_convergence_comparison.png",
    )

def add_top10_recovery(
    trace: pd.DataFrame,
) -> pd.Series:
    """
    Return cumulative top-10 recovery.

    When a molecule identifier is available, repeated selections of the same
    molecule count once. Otherwise, this falls back to the cumulative number
    of top-10 selections.
    """
    top10 = trace["selected_is_true_top_10"].fillna(False)

    id_column = first_existing_column(
        trace,
        (
            "selected_recipe_id",
            "RecipeID",
            "recipe_id",
            "selected_smiles",
            "SMILES",
            "smiles",
        ),
    )

    if id_column is None:
        return top10.astype(int).cumsum()

    seen = set()
    counts = []
    total = 0

    for is_top10, molecule_id in zip(top10, trace[id_column]):
        if bool(is_top10) and pd.notna(molecule_id):
            key = str(molecule_id)
            if key not in seen:
                seen.add(key)
                total += 1
        counts.append(total)

    return pd.Series(counts, index=trace.index, dtype=int)


def plot_top10_recovery(
    traces: Dict[str, pd.DataFrame],
    figures_dir: Path,
    seed: Optional[int] = None,
) -> None:
    """Compare cumulative recovery of true top-10 molecules."""
    usable = {
        label: trace
        for label, trace in traces.items()
        if "selected_is_true_top_10" in trace.columns
    }

    if not usable:
        print(
            "Skipped top-10 recovery plot: no trace contains "
            "'selected_is_true_top_10'."
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, trace in usable.items():
        recovery = add_top10_recovery(trace)
        ax.step(
            trace["iteration"],
            recovery,
            where="post",
            linewidth=2.2,
            label=label,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative true top-10 molecules found")
    ax.set_title(
        "True top-10 recovery"
        + (f" (seed {seed})" if seed is not None else "")
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        figures_dir / "top10_recovery_comparison.png",
    )


def plot_selected_true_rank(
    traces: Dict[str, pd.DataFrame],
    figures_dir: Path,
    seed: Optional[int] = None,
) -> None:
    """Plot the true LD50 rank of the molecule selected each iteration."""
    usable = {
        label: trace
        for label, trace in traces.items()
        if "selected_true_ld50_rank" in trace.columns
    }

    if not usable:
        print(
            "Skipped selected-rank plot: no trace contains "
            "'selected_true_ld50_rank'."
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, trace in usable.items():
        ax.scatter(
            trace["iteration"],
            trace["selected_true_ld50_rank"],
            alpha=0.7,
            label=label,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("True LD50 rank of selected molecule")
    ax.set_title(
        "Quality of selected molecules"
        + (f" (seed {seed})" if seed is not None else "")
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        figures_dir / "selected_true_rank_comparison.png",
    )


def create_selection_summary(
    traces: Dict[str, pd.DataFrame],
    tables_dir: Path,
) -> pd.DataFrame:
    """Create a per-run summary of selection quality."""
    records = []

    for label, trace in traces.items():
        running_best = trace["selected_ld50_raw"].cummax()
        improvement_mask = running_best.gt(
            running_best.shift(1, fill_value=-np.inf)
        )

        record = {
            "method": label,
            "iterations": int(len(trace)),
            "first_selected_ld50": float(
                trace["selected_ld50_raw"].iloc[0]
            ),
            "final_best_ld50": float(running_best.iloc[-1]),
            "improvement_from_first_selection": float(
                running_best.iloc[-1]
                - trace["selected_ld50_raw"].iloc[0]
            ),
            "mean_best_ld50_over_iterations": float(
                running_best.mean()
            ),
            "number_of_improvements": int(improvement_mask.sum()),
        }

        if "selected_true_ld50_rank" in trace.columns:
            ranks = trace["selected_true_ld50_rank"].dropna()
            record.update(
                {
                    "best_true_rank_selected": (
                        float(ranks.min()) if not ranks.empty else np.nan
                    ),
                    "mean_true_rank_selected": (
                        float(ranks.mean()) if not ranks.empty else np.nan
                    ),
                    "median_true_rank_selected": (
                        float(ranks.median()) if not ranks.empty else np.nan
                    ),
                }
            )
        else:
            record.update(
                {
                    "best_true_rank_selected": np.nan,
                    "mean_true_rank_selected": np.nan,
                    "median_true_rank_selected": np.nan,
                }
            )

        if "selected_is_true_top_10" in trace.columns:
            top10 = trace["selected_is_true_top_10"].fillna(False)
            top10_rows = trace.loc[top10]

            first_top10_iteration = (
                float(top10_rows["iteration"].iloc[0])
                if not top10_rows.empty
                else np.nan
            )

            id_column = first_existing_column(
                trace,
                (
                    "selected_recipe_id",
                    "RecipeID",
                    "recipe_id",
                    "selected_smiles",
                    "SMILES",
                    "smiles",
                ),
            )

            unique_top10_found = np.nan
            if id_column is not None:
                unique_top10_found = int(
                    trace.loc[top10, id_column].dropna().astype(str).nunique()
                )

            record.update(
                {
                    "top10_molecules_selected": int(top10.sum()),
                    "top10_selection_percentage": float(
                        100.0 * top10.mean()
                    ),
                    "first_top10_iteration": first_top10_iteration,
                    "unique_top10_molecules_found": unique_top10_found,
                }
            )
        else:
            record.update(
                {
                    "top10_molecules_selected": np.nan,
                    "top10_selection_percentage": np.nan,
                    "first_top10_iteration": np.nan,
                    "unique_top10_molecules_found": np.nan,
                }
            )

        records.append(record)

    summary = pd.DataFrame(records)
    output_path = tables_dir / "selection_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved table: {output_path}")
    return summary


def create_best_molecules_table(
    traces: Dict[str, pd.DataFrame],
    tables_dir: Path,
) -> pd.DataFrame:
    """Record every iteration that establishes a new best observed LD50."""
    records = []

    id_column_candidates = (
        "selected_recipe_id",
        "RecipeID",
        "recipe_id",
    )
    smiles_column_candidates = (
        "selected_smiles",
        "SMILES",
        "smiles",
    )

    for label, trace in traces.items():
        running_best = trace["selected_ld50_raw"].cummax()
        previous_best = running_best.shift(1, fill_value=-np.inf)
        improvement_rows = trace.loc[running_best.gt(previous_best)].copy()

        id_column = first_existing_column(trace, id_column_candidates)
        smiles_column = first_existing_column(
            trace, smiles_column_candidates
        )

        for _, row in improvement_rows.iterrows():
            record = {
                "method": label,
                "iteration": int(row["iteration"]),
                "selected_ld50_raw": float(row["selected_ld50_raw"]),
            }

            if id_column is not None:
                record["recipe_id"] = row[id_column]

            if smiles_column is not None:
                record["smiles"] = row[smiles_column]

            if "selected_true_ld50_rank" in trace.columns:
                record["true_ld50_rank"] = row[
                    "selected_true_ld50_rank"
                ]

            if "selected_is_true_top_10" in trace.columns:
                record["is_true_top_10"] = bool(
                    row["selected_is_true_top_10"]
                )

            records.append(record)

    best_molecules = pd.DataFrame(records)
    output_path = tables_dir / "best_molecules_discovered.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_molecules.to_csv(output_path, index=False)
    print(f"Saved table: {output_path}")
    return best_molecules



def create_method_overlap_tables(
    traces: Dict[str, pd.DataFrame],
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare selected molecules across optimisation methods.

    Produces:
      - method_overlap_summary.csv
      - shared_and_unique_molecules.csv
    """
    molecule_sets: Dict[str, set[str]] = {}
    top10_sets: Dict[str, set[str]] = {}
    molecule_metadata: Dict[str, dict] = {}

    for label, trace in traces.items():
        id_column = first_existing_column(
            trace,
            (
                "selected_recipe_id",
                "RecipeID",
                "recipe_id",
                "selected_smiles",
                "SMILES",
                "smiles",
            ),
        )

        if id_column is None:
            print(
                f"Skipped overlap analysis for {label}: "
                "no molecule identifier column was found."
            )
            continue

        ids = trace[id_column].dropna().astype(str)
        molecule_sets[label] = set(ids)

        if "selected_is_true_top_10" in trace.columns:
            top10_mask = trace["selected_is_true_top_10"].fillna(False)
            top10_sets[label] = set(
                trace.loc[top10_mask, id_column].dropna().astype(str)
            )
        else:
            top10_sets[label] = set()

        for _, row in trace.iterrows():
            if pd.isna(row[id_column]):
                continue

            molecule_id = str(row[id_column])
            metadata = molecule_metadata.setdefault(
                molecule_id,
                {
                    "molecule_id": molecule_id,
                    "smiles": np.nan,
                    "selected_ld50_raw": np.nan,
                    "true_ld50_rank": np.nan,
                },
            )

            smiles_column = first_existing_column(
                trace,
                ("selected_smiles", "SMILES", "smiles"),
            )
            if smiles_column is not None and pd.notna(row[smiles_column]):
                metadata["smiles"] = row[smiles_column]

            if "selected_ld50_raw" in trace.columns:
                value = pd.to_numeric(
                    pd.Series([row["selected_ld50_raw"]]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(value):
                    metadata["selected_ld50_raw"] = float(value)

            if "selected_true_ld50_rank" in trace.columns:
                rank = pd.to_numeric(
                    pd.Series([row["selected_true_ld50_rank"]]),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(rank):
                    metadata["true_ld50_rank"] = float(rank)

    if len(molecule_sets) < 2:
        print(
            "Skipped method-overlap tables: at least two traces with "
            "molecule identifiers are required."
        )
        return pd.DataFrame(), pd.DataFrame()

    labels = list(molecule_sets)
    summary_records = []

    for i, left_label in enumerate(labels):
        for right_label in labels[i + 1:]:
            left = molecule_sets[left_label]
            right = molecule_sets[right_label]
            shared = left & right
            union = left | right

            left_top10 = top10_sets.get(left_label, set())
            right_top10 = top10_sets.get(right_label, set())
            shared_top10 = left_top10 & right_top10

            summary_records.append(
                {
                    "method_a": left_label,
                    "method_b": right_label,
                    "unique_molecules_method_a": len(left),
                    "unique_molecules_method_b": len(right),
                    "shared_molecules": len(shared),
                    "molecules_only_method_a": len(left - right),
                    "molecules_only_method_b": len(right - left),
                    "jaccard_similarity": (
                        len(shared) / len(union) if union else np.nan
                    ),
                    "top10_molecules_method_a": len(left_top10),
                    "top10_molecules_method_b": len(right_top10),
                    "shared_top10_molecules": len(shared_top10),
                    "top10_only_method_a": len(left_top10 - right_top10),
                    "top10_only_method_b": len(right_top10 - left_top10),
                }
            )

    overlap_summary = pd.DataFrame(summary_records)
    summary_path = tables_dir / "method_overlap_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    overlap_summary.to_csv(summary_path, index=False)
    print(f"Saved table: {summary_path}")

    molecule_records = []
    all_molecules = set().union(*molecule_sets.values())

    for molecule_id in sorted(all_molecules):
        selected_by = [
            label for label, values in molecule_sets.items()
            if molecule_id in values
        ]
        top10_for = [
            label for label, values in top10_sets.items()
            if molecule_id in values
        ]

        metadata = molecule_metadata.get(
            molecule_id,
            {
                "molecule_id": molecule_id,
                "smiles": np.nan,
                "selected_ld50_raw": np.nan,
                "true_ld50_rank": np.nan,
            },
        ).copy()

        metadata.update(
            {
                "selected_by_methods": "; ".join(selected_by),
                "number_of_methods": len(selected_by),
                "top10_for_methods": "; ".join(top10_for),
                "is_shared": len(selected_by) > 1,
                "is_top10_for_any_method": len(top10_for) > 0,
            }
        )
        molecule_records.append(metadata)

    molecule_table = pd.DataFrame(molecule_records)
    molecule_table = molecule_table.sort_values(
        by=["is_top10_for_any_method", "true_ld50_rank", "molecule_id"],
        ascending=[False, True, True],
        na_position="last",
    )

    molecule_path = tables_dir / "shared_and_unique_molecules.csv"
    molecule_table.to_csv(molecule_path, index=False)
    print(f"Saved table: {molecule_path}")

    return overlap_summary, molecule_table

def create_final_summary(
    selection_summary: pd.DataFrame,
    tables_dir: Path,
) -> pd.DataFrame:
    """Create a compact table suitable for the thesis Results chapter."""
    preferred_columns = [
        "method",
        "iterations",
        "final_best_ld50",
        "improvement_from_first_selection",
        "mean_best_ld50_over_iterations",
        "number_of_improvements",
        "best_true_rank_selected",
        "mean_true_rank_selected",
        "top10_molecules_selected",
        "first_top10_iteration",
        "unique_top10_molecules_found",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in selection_summary.columns
    ]

    final_summary = selection_summary[available_columns].copy()
    output_path = tables_dir / "final_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_summary.to_csv(output_path, index=False)
    print(f"Saved table: {output_path}")
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Bayesian optimisation figures and summary tables "
            "from one or more bo_trace CSV files."
        )
    )

    parser.add_argument(
        "traces",
        nargs="*",
        default=[],
        help=(
            "Legacy: one or more paths to BO trace CSV files, analysed/"
            "compared generically. Prefer --trace or --ucb-trace/"
            "--pareto-trace for seed-aware output folders. Mutually "
            "exclusive with those options."
        ),
    )

    parser.add_argument(
        "--trace",
        default=None,
        help=(
            "Analyse a single BO trace CSV, e.g. "
            "bo/results/seed_11/ucb/bo_trace.csv. Output defaults to "
            "'<trace_dir>/analysis/'."
        ),
    )

    parser.add_argument(
        "--ucb-trace",
        default=None,
        help=(
            "UCB bo_trace.csv for a paired UCB-vs-Pareto comparison. Must "
            "be passed together with --pareto-trace."
        ),
    )

    parser.add_argument(
        "--pareto-trace",
        default=None,
        help=(
            "Pareto bo_trace.csv for a paired UCB-vs-Pareto comparison. "
            "Must be passed together with --ucb-trace."
        ),
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help=(
            "Directory for PNG figures. Default depends on the input "
            "mode: '<trace_dir>/analysis/' for --trace, "
            "'<seed_dir>/comparison/' for --ucb-trace/--pareto-trace, or "
            "analysis/figures for the legacy positional traces."
        ),
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help=(
            "Directory for CSV tables. Same default rules as "
            "--figures-dir; analysis/tables for the legacy positional "
            "traces."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        trace_paths, figures_dir, tables_dir, seed = resolve_traces_and_output_dirs(
            args
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    figures_dir = Path(figures_dir).expanduser().resolve()
    tables_dir = Path(tables_dir).expanduser().resolve()

    try:
        traces = load_traces(trace_paths)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Loaded {len(traces)} trace file(s):")
    for label, trace in traces.items():
        print(f"  - {label}: {len(trace)} iterations")

        molecule_id_column = first_existing_column(
            trace,
            (
                "selected_recipe_id",
                "RecipeID",
                "recipe_id",
                "selected_smiles",
                "SMILES",
                "smiles",
            ),
        )
        if molecule_id_column is None:
            print(
                "    Note: no molecule identifier is present, so "
                "'unique_top10_molecules_found' will be left blank."
            )

    dataset_max_ld50 = compute_dataset_max_ld50()
    plot_convergence(traces, dataset_max_ld50, figures_dir, seed=seed)
    plot_top10_recovery(traces, figures_dir, seed=seed)
    plot_selected_true_rank(traces, figures_dir, seed=seed)

    selection_summary = create_selection_summary(traces, tables_dir)
    create_best_molecules_table(traces, tables_dir)
    create_method_overlap_tables(traces, tables_dir)
    create_final_summary(selection_summary, tables_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()