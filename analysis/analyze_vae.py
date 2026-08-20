"""Generate VAE training figures and summary tables from a local MLflow run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Charter", "Georgia", "DejaVu Serif"],
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_DIR = ROOT / "notebooks" / "mlruns"
DEFAULT_FIGURES_DIR = ROOT / "analysis" / "figures" / "final_vae"
DEFAULT_TABLES_DIR = ROOT / "analysis" / "tables" / "final_vae"

EPOCH_METRICS = [
    "train_loss", "train_recon", "train_kl", "train_reg_loss",
    "train_recon_acc", "train_token_acc", "train_validity",
    "valid_loss", "valid_recon", "valid_kl", "valid_reg_loss",
    "valid_rmse", "valid_mae", "grad_norm", "beta", "learning_rate",
    "valid_recon_acc", "valid_token_acc", "valid_validity",
]

SUMMARY_METRICS = [
    "best_valid_loss", "valid_loss", "valid_recon", "valid_kl",
    "valid_reg_loss", "valid_rmse", "valid_mae",
    "valid_recon_acc", "valid_token_acc", "valid_validity",
    "test_rmse", "test_mae",
    "test_recon_acc", "test_validity", "test_canonical_acc",
    "test_ld50_rmse", "test_ld50_mae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VAE figures and tables from a local MLflow run."
    )
    parser.add_argument("--run-id", required=True, help="Full MLflow run ID.")
    parser.add_argument(
        "--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR,
        help="Local MLflow file-store directory.",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=DEFAULT_TABLES_DIR,
        help="Output directory for CSV tables.",
    )
    return parser.parse_args()


def configure_mlflow(tracking_dir: Path) -> MlflowClient:
    tracking_dir = tracking_dir.expanduser().resolve()
    if not tracking_dir.exists():
        raise FileNotFoundError(
            f"MLflow tracking directory does not exist: {tracking_dir}"
        )
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    return MlflowClient()


def metric_history_to_frame(
    client: MlflowClient,
    run_id: str,
    metric_names: Iterable[str],
) -> pd.DataFrame:
    records = []
    for metric_name in metric_names:
        for point in client.get_metric_history(run_id, metric_name):
            records.append({
                "metric": metric_name,
                "step": int(point.step),
                "value": float(point.value),
                "timestamp": int(point.timestamp),
            })

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    return (
        frame.sort_values(["metric", "step", "timestamp"])
        .drop_duplicates(subset=["metric", "step"], keep="last")
        .reset_index(drop=True)
    )


def history_series(history: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    return (
        history.loc[history["metric"].eq(metric_name), ["step", "value"]]
        .sort_values("step")
        .copy()
    )


def plot_metric_pair(
    ax: plt.Axes,
    history: pd.DataFrame,
    train_metric: str,
    valid_metric: str,
    title: str,
    ylabel: str,
    show_legend: bool = True,
) -> None:
    plotted = False
    for metric_name, label in ((train_metric, "Train"), (valid_metric, "Validation")):
        series = history_series(history, metric_name)
        if not series.empty:
            ax.plot(series["step"], series["value"], linewidth=2, label=label)
            plotted = True

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        if show_legend:
            ax.legend()
    else:
        ax.text(0.5, 0.5, "Metric history unavailable", ha="center", va="center", transform=ax.transAxes)


def save_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # Train/Validation legend only on the Total loss panel (top-left) -- it
    # applies identically to all four panels, so showing it once avoids
    # repeating the same legend four times.
    plot_metric_pair(axes[0, 0], history, "train_loss", "valid_loss", "Total Loss", "Loss", show_legend=True)
    plot_metric_pair(axes[0, 1], history, "train_recon", "valid_recon", "Reconstruction Loss", "Loss", show_legend=False)
    plot_metric_pair(axes[1, 0], history, "train_kl", "valid_kl", "KL divergence", "KL Loss", show_legend=False)
    plot_metric_pair(axes[1, 1], history, "train_reg_loss", "valid_reg_loss", "Solubility Regression Loss", "Loss", show_legend=False)
    fig.suptitle("VAE Training History", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def save_training_diagnostics(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    diagnostics = [
        ("beta", "KL weight", "Beta"),
        ("learning_rate", "Learning rate", "Learning rate"),
        ("grad_norm", "Gradient norm", "Gradient norm"),
    ]
    for ax, (metric_name, title, ylabel) in zip(axes, diagnostics):
        series = history_series(history, metric_name)
        if series.empty:
            ax.text(0.5, 0.5, "Metric history unavailable", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(series["step"], series["value"], linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.suptitle("VAE Optimisation Diagnostics", y=1.03, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def latest_metric_values(history: pd.DataFrame) -> Dict[str, float]:
    values: Dict[str, float] = {}
    if history.empty:
        return values
    for metric_name, group in history.groupby("metric"):
        latest = group.sort_values(["step", "timestamp"]).iloc[-1]
        values[metric_name] = float(latest["value"])
    return values


def best_validation_epoch(history: pd.DataFrame) -> float:
    valid_loss = history_series(history, "valid_loss")
    if valid_loss.empty:
        return np.nan
    row = valid_loss.loc[valid_loss["value"].idxmin()]
    return float(row["step"])


def create_summary_table(run, history: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    latest = latest_metric_values(history)
    record: Dict[str, object] = {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "best_validation_epoch": best_validation_epoch(history),
    }

    for metric_name in SUMMARY_METRICS:
        if metric_name in run.data.metrics:
            record[metric_name] = float(run.data.metrics[metric_name])
        elif metric_name in latest:
            record[metric_name] = latest[metric_name]
        else:
            record[metric_name] = np.nan

    useful_params = [
        "dataset", "split", "trained_head", "epochs", "batch_size",
        "latent_dim", "hidden_dim", "n_layers", "prop_hidden_size",
        "dropout", "beta_max", "kl_free_bits", "gamma", "learning_rate",
        "anneal_epochs", "reg_mean", "reg_std",
    ]
    for param_name in useful_params:
        record[param_name] = run.data.params.get(param_name, np.nan)

    record["best_checkpoint"] = run.data.tags.get("best_checkpoint", "")
    record["eval_checkpoint"] = run.data.tags.get("eval_checkpoint", "")

    summary = pd.DataFrame([record])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved table: {output_path}")
    return summary


def main() -> None:
    args = parse_args()
    tracking_dir = args.tracking_dir.expanduser().resolve()
    figures_dir = args.figures_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()

    try:
        client = configure_mlflow(tracking_dir)
        run = client.get_run(args.run_id)
    except Exception as error:
        print(f"Error loading MLflow run: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Loaded MLflow run: {run.info.run_id}")
    print("Run name:", run.data.tags.get("mlflow.runName", "(unnamed)"))

    history = metric_history_to_frame(client, args.run_id, EPOCH_METRICS)
    if history.empty:
        print("Error: no epoch-level metric history was found for this run.", file=sys.stderr)
        raise SystemExit(1)

    save_training_curves(history, figures_dir / "vae_training_curves.png")
    save_training_diagnostics(history, figures_dir / "vae_training_diagnostics.png")

    tables_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(tables_dir / "vae_metric_history.csv", index=False)
    print(f"Saved table: {tables_dir / 'vae_metric_history.csv'}")
    create_summary_table(run, history, tables_dir / "vae_summary.csv")

    print("\nVAE analysis complete")


if __name__ == "__main__":
    main()