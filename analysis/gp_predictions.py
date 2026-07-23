from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_DIR = ROOT / "mlruns"
DEFAULT_FIGURES_DIR = ROOT / "analysis" / "figures" / "final_gp"
DEFAULT_TABLES_DIR = ROOT / "analysis" / "tables" / "final_gp"

REQUIRED_COLUMNS = {
    "LD50_raw",
    "LD50_standardised",
    "mu",
    "sigma",
    "observed",
    "candidate",
    "is_selected",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Per-Iteration GP Predictions From MLflow."
    )
    parser.add_argument(
        "--ucb-run-id",
        default="401b7fc59ad349d09a9865323764c496",
        help="MLflow Run ID For The UCB Experiment.",
    )
    parser.add_argument(
        "--pareto-run-id",
        default="ea67a924d17846889cd82a2bd1332487",
        help="MLflow Run ID For The Pareto Experiment.",
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=DEFAULT_TRACKING_DIR,
        help="Path To The Local MLflow Tracking Store.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory For Output Figures.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Directory For Output Tables.",
    )
    return parser.parse_args()


def list_prediction_artifacts(
    client: MlflowClient,
    run_id: str,
    path: str = "",
) -> list[str]:
    matches: list[str] = []

    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            matches.extend(
                list_prediction_artifacts(client, run_id, artifact.path)
            )
        elif artifact.path.endswith("all_candidate_predictions.csv"):
            matches.append(artifact.path)

    return matches


def extract_iteration(path: str, frame: pd.DataFrame) -> int:
    patterns = [
        r"(?:iteration|iter)[_-]?(\d+)",
        r"(?:^|/)(\d+)(?:/|$)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, path, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1])

    return int(frame["observed"].astype(bool).sum())


def infer_target_scaling(frame: pd.DataFrame) -> tuple[float, float]:
    """
    Infer raw LD50 = intercept + scale * standardized LD50.

    This avoids relying on checkpoint metadata and allows GP means and
    uncertainties to be reported in the original LD50 units.
    """
    valid = frame[["LD50_standardised", "LD50_raw"]].dropna()

    if len(valid) < 2:
        raise ValueError("Not Enough Values To Infer LD50 Scaling.")

    scale, intercept = np.polyfit(
        valid["LD50_standardised"].to_numpy(dtype=float),
        valid["LD50_raw"].to_numpy(dtype=float),
        deg=1,
    )

    if not np.isfinite(scale) or abs(scale) < 1e-12:
        raise ValueError("Invalid LD50 Standardisation Scale.")

    return float(intercept), float(scale)


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)

    raise ValueError(f"Unknown Correlation Method: {method}")


def calculate_metrics(
    frame: pd.DataFrame,
    method: str,
    artifact_path: str,
    provisional_iteration: int,
) -> dict:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{artifact_path} Is Missing Columns: {sorted(missing)}"
        )

    evaluation_frame = frame.copy()

    if evaluation_frame.empty:
        raise ValueError(f"No Rows Found In {artifact_path}")

    intercept, scale = infer_target_scaling(frame)

    evaluation_frame["predicted_ld50_raw"] = (
        intercept + scale * evaluation_frame["mu"].astype(float)
    )
    evaluation_frame["sigma_raw"] = (
        abs(scale) * evaluation_frame["sigma"].astype(float)
    )

    true_values = evaluation_frame["LD50_raw"].to_numpy(dtype=float)
    predictions = evaluation_frame["predicted_ld50_raw"].to_numpy(dtype=float)

    selected = frame.loc[frame["is_selected"].astype(bool)]
    selected_sigma_raw = float("nan")
    selected_error_raw = float("nan")

    if len(selected) == 1:
        selected_mu = float(selected.iloc[0]["mu"])
        selected_true = float(selected.iloc[0]["LD50_raw"])
        selected_pred = intercept + scale * selected_mu
        selected_sigma_raw = abs(scale) * float(selected.iloc[0]["sigma"])
        selected_error_raw = abs(selected_true - selected_pred)

    return {
        "method": method,
        "artifact_path": artifact_path,
        "provisional_iteration": provisional_iteration,
        "n_observed": int(frame["observed"].astype(bool).sum()),
        "n_full_dataset": int(len(evaluation_frame)),
        "full_dataset_rmse": float(
            np.sqrt(mean_squared_error(true_values, predictions))
        ),
        "full_dataset_mae": float(mean_absolute_error(true_values, predictions)),
        "full_dataset_pearson": safe_correlation(
            true_values, predictions, "pearson"
        ),
        "full_dataset_spearman": safe_correlation(
            true_values, predictions, "spearman"
        ),
        "full_dataset_mean_sigma": float(evaluation_frame["sigma_raw"].mean()),
        "full_dataset_median_sigma": float(evaluation_frame["sigma_raw"].median()),
        "full_dataset_maximum_sigma": float(evaluation_frame["sigma_raw"].max()),
        "selected_sigma": selected_sigma_raw,
        "selected_absolute_error": selected_error_raw,
        "ld50_intercept": intercept,
        "ld50_scale": scale,
    }


def load_run_predictions(
    client: MlflowClient,
    run_id: str,
    method: str,
    download_root: Path,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    artifact_paths = list_prediction_artifacts(client, run_id)

    if not artifact_paths:
        raise FileNotFoundError(
            f"No all_candidate_predictions.csv Artifacts Found For {method} "
            f"Run {run_id}."
        )

    records: list[dict] = []
    frames_by_provisional_iteration: dict[int, pd.DataFrame] = {}

    for artifact_path in artifact_paths:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=str(download_root / method),
        )
        frame = pd.read_csv(local_path)
        provisional_iteration = extract_iteration(artifact_path, frame)

        records.append(
            calculate_metrics(
                frame=frame,
                method=method,
                artifact_path=artifact_path,
                provisional_iteration=provisional_iteration,
            )
        )
        frames_by_provisional_iteration[provisional_iteration] = frame

    metrics = pd.DataFrame(records)

    provisional_values = sorted(metrics["provisional_iteration"].unique())
    path_has_zero_based_iteration = (
        min(provisional_values) == 0
        and max(provisional_values) <= len(provisional_values)
    )

    if path_has_zero_based_iteration:
        metrics["iteration"] = metrics["provisional_iteration"].astype(int)
        frames_by_iteration = frames_by_provisional_iteration
    else:
        minimum_observed = int(metrics["n_observed"].min())
        metrics["iteration"] = (
            metrics["n_observed"].astype(int) - minimum_observed
        )
        frames_by_iteration = {}
        for provisional, frame in frames_by_provisional_iteration.items():
            iteration = int(
                frame["observed"].astype(bool).sum() - minimum_observed
            )
            frames_by_iteration[iteration] = frame

    metrics.sort_values("iteration", inplace=True)
    metrics.reset_index(drop=True, inplace=True)

    return metrics, frames_by_iteration


def add_raw_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    intercept, scale = infer_target_scaling(output)
    output["predicted_ld50_raw"] = (
        intercept + scale * output["mu"].astype(float)
    )
    output["sigma_raw"] = abs(scale) * output["sigma"].astype(float)
    return output


def save_final_prediction_plot(
    frames_by_method: dict[str, dict[int, pd.DataFrame]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, method in zip(axes, ["UCB", "Pareto"]):
        iterations = frames_by_method[method]
        final_iteration = max(iterations)
        frame = add_raw_prediction_columns(iterations[final_iteration])
        evaluation_frame = frame.copy()

        ax.scatter(
            evaluation_frame["LD50_raw"],
            evaluation_frame["predicted_ld50_raw"],
            s=18,
            alpha=0.55,
        )

        lower = float(
            min(
                evaluation_frame["LD50_raw"].min(),
                evaluation_frame["predicted_ld50_raw"].min(),
            )
        )
        upper = float(
            max(
                evaluation_frame["LD50_raw"].max(),
                evaluation_frame["predicted_ld50_raw"].max(),
            )
        )

        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("True LD50")
        ax.set_ylabel("Predicted LD50")
        ax.set_title(f"{method} — Iteration {final_iteration} (Full Dataset)")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Predicted vs. True LD50 (Full Dataset)", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {output_path}")


def save_error_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method, group in metrics.groupby("method"):
        ax.plot(
            group["iteration"],
            group["full_dataset_rmse"],
            marker="o",
            markersize=3,
            label=f"{method} Full-Dataset RMSE",
        )
        ax.plot(
            group["iteration"],
            group["full_dataset_mae"],
            marker="o",
            markersize=3,
            linestyle="--",
            label=f"{method} Full-Dataset MAE",
        )

    ax.set_xlabel("Bayesian Optimisation Iteration")
    ax.set_ylabel("Full-Dataset Prediction Error In Raw LD50 Units")
    ax.set_title("GP Prediction Error Over Iterations (Full Dataset)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {output_path}")


def save_correlation_plot(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method, group in metrics.groupby("method"):
        ax.plot(
            group["iteration"],
            group["full_dataset_pearson"],
            marker="o",
            markersize=3,
            label=f"{method} Full-Dataset Pearson",
        )
        ax.plot(
            group["iteration"],
            group["full_dataset_spearman"],
            marker="o",
            markersize=3,
            linestyle="--",
            label=f"{method} Full-Dataset Spearman",
        )

    ax.set_xlabel("Bayesian Optimisation Iteration")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("GP Prediction Correlation Over Iterations (Full Dataset)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {output_path}")


def save_uncertainty_plot(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method, group in metrics.groupby("method"):
        ax.plot(
            group["iteration"],
            group["full_dataset_mean_sigma"],
            marker="o",
            markersize=3,
            label=f"{method} Mean Full-Dataset Sigma",
        )
        ax.plot(
            group["iteration"],
            group["selected_sigma"],
            marker="o",
            markersize=3,
            linestyle="--",
            label=f"{method} Selected Sigma",
        )

    ax.set_xlabel("Bayesian Optimisation Iteration")
    ax.set_ylabel("Predictive Standard Deviation In Raw LD50 Units")
    ax.set_title("GP Uncertainty Over Iterations (Full Dataset & Selected)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {output_path}")


def build_final_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method, group in metrics.groupby("method"):
        group = group.sort_values("iteration")
        first = group.iloc[0]
        final = group.iloc[-1]

        rows.append(
            {
                "method": method,
                "n_iterations_analyzed": int(len(group)),
                "first_iteration": int(first["iteration"]),
                "final_iteration": int(final["iteration"]),
                "initial_full_dataset_rmse": float(first["full_dataset_rmse"]),
                "final_full_dataset_rmse": float(final["full_dataset_rmse"]),
                "full_dataset_rmse_change": float(
                    final["full_dataset_rmse"] - first["full_dataset_rmse"]
                ),
                "initial_full_dataset_mae": float(first["full_dataset_mae"]),
                "final_full_dataset_mae": float(final["full_dataset_mae"]),
                "full_dataset_mae_change": float(
                    final["full_dataset_mae"] - first["full_dataset_mae"]
                ),
                "initial_full_dataset_pearson": float(
                    first["full_dataset_pearson"]
                ),
                "final_full_dataset_pearson": float(
                    final["full_dataset_pearson"]
                ),
                "initial_full_dataset_spearman": float(
                    first["full_dataset_spearman"]
                ),
                "final_full_dataset_spearman": float(
                    final["full_dataset_spearman"]
                ),
                "initial_full_dataset_mean_sigma": float(
                    first["full_dataset_mean_sigma"]
                ),
                "final_full_dataset_mean_sigma": float(
                    final["full_dataset_mean_sigma"]
                ),
                "full_dataset_mean_sigma_change": float(
                    final["full_dataset_mean_sigma"]
                    - first["full_dataset_mean_sigma"]
                ),
                "mean_selected_sigma": float(group["selected_sigma"].mean()),
                "mean_selected_absolute_error": float(
                    group["selected_absolute_error"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    tracking_dir = args.tracking_dir.expanduser().resolve()
    figures_dir = args.figures_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()

    if not tracking_dir.exists():
        raise FileNotFoundError(
            f"MLflow Tracking Directory Does Not Exist: {tracking_dir}"
        )

    mlflow.set_tracking_uri(tracking_dir.as_uri())
    client = MlflowClient()

    run_ids = {
        "UCB": args.ucb_run_id,
        "Pareto": args.pareto_run_id,
    }

    all_metrics = []
    frames_by_method: dict[str, dict[int, pd.DataFrame]] = {}

    with tempfile.TemporaryDirectory(prefix="gp_predictions_") as temp_dir:
        download_root = Path(temp_dir)

        for method, run_id in run_ids.items():
            print(f"Reading {method} Run: {run_id}")
            metrics, frames = load_run_predictions(
                client=client,
                run_id=run_id,
                method=method,
                download_root=download_root,
            )
            all_metrics.append(metrics)
            frames_by_method[method] = frames
            print(
                f"Found {len(metrics)} Prediction Snapshots For {method}."
            )

        metrics = pd.concat(all_metrics, ignore_index=True)
        metrics.sort_values(["method", "iteration"], inplace=True)

        save_final_prediction_plot(
            frames_by_method,
            figures_dir / "predicted_vs_true_ld50.png",
        )

    save_error_plot(
        metrics,
        figures_dir / "gp_prediction_error_over_iterations.png",
    )
    save_correlation_plot(
        metrics,
        figures_dir / "gp_prediction_correlation_over_iterations.png",
    )
    save_uncertainty_plot(
        metrics,
        figures_dir / "gp_uncertainty_over_iterations.png",
    )

    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_output = metrics.drop(
        columns=["provisional_iteration"],
        errors="ignore",
    )
    metrics_path = tables_dir / "gp_prediction_metrics.csv"
    metrics_output.to_csv(metrics_path, index=False)
    print(f"Saved Table: {metrics_path}")

    summary = build_final_summary(metrics)
    summary_path = tables_dir / "gp_final_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved Table: {summary_path}")

    print("\nGP Prediction Analysis Complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error