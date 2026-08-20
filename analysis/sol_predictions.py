from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Charter", "Georgia", "DejaVu Serif"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_dataloaders_multitask  # noqa: E402
from model import VAE  


DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints" / "vae_solubility_112564ed_best.pth"
)
DEFAULT_FIGURES_DIR = ROOT / "analysis" / "figures" / "final_vae"
DEFAULT_TABLES_DIR = ROOT / "analysis" / "tables" / "final_vae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate The Solubility Prediction Head On The Scaffold-Split "
            "Test Set."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path To The Trained VAE Checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch Size Used To Rebuild The Data Loaders.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device Used For Evaluation.",
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


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Was Requested But Is Not Available.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS Was Requested But Is Not Available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint Does Not Exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = {"state_dict", "build", "standardise"}
    missing = required_keys.difference(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint Is Missing Required Key(s): "
            + ", ".join(sorted(missing))
        )

    return checkpoint


def build_model(
    checkpoint: dict,
    device: torch.device,
) -> VAE:
    build = checkpoint["build"]

    required_build_keys = {
        "vocab_size",
        "seq_len",
        "hidden_dim",
        "latent_dim",
        "n_layers",
        "dropout",
        "prop_hidden_size",
    }
    missing = required_build_keys.difference(build)
    if missing:
        raise ValueError(
            "Checkpoint Build Configuration Is Missing Key(s): "
            + ", ".join(sorted(missing))
        )

    model = VAE(
        vocab_size=int(build["vocab_size"]),
        seq_len=int(build["seq_len"]),
        hidden_dim=int(build["hidden_dim"]),
        latent_dim=int(build["latent_dim"]),
        n_layers=int(build["n_layers"]),
        dropout=float(build["dropout"]),
        prop_hidden_size=int(build["prop_hidden_size"]),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def collect_test_predictions(
    model: VAE,
    test_loader,
    device: torch.device,
    reg_mean: float,
    reg_std: float,
) -> pd.DataFrame:
    predicted_values = []
    true_values = []

    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)

            mu, _ = model.encoder(batch)
            standardized_predictions = (
                model.reg_predictor(mu).reshape(-1).cpu()
            )
            standardized_labels = labels[:, 0].reshape(-1).cpu()

            predicted = standardized_predictions * reg_std + reg_mean
            true = standardized_labels * reg_std + reg_mean

            predicted_values.extend(predicted.numpy().tolist())
            true_values.extend(true.numpy().tolist())

    frame = pd.DataFrame(
        {
            "test_sample_index": np.arange(len(true_values), dtype=int),
            "true_solubility": true_values,
            "predicted_solubility": predicted_values,
        }
    )

    frame["residual"] = (
        frame["predicted_solubility"] - frame["true_solubility"]
    )
    frame["absolute_error"] = frame["residual"].abs()

    return frame


def calculate_metrics(predictions: pd.DataFrame) -> dict:
    residuals = predictions["residual"].to_numpy(dtype=float)
    true_values = predictions["true_solubility"].to_numpy(dtype=float)
    predicted_values = predictions[
        "predicted_solubility"
    ].to_numpy(dtype=float)

    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))

    if len(predictions) > 1:
        pearson = float(np.corrcoef(true_values, predicted_values)[0, 1])
        spearman = float(
            pd.Series(true_values).corr(
                pd.Series(predicted_values),
                method="spearman",
            )
        )
    else:
        pearson = np.nan
        spearman = np.nan

    return {
        "n_test_samples": int(len(predictions)),
        "test_rmse": rmse,
        "test_mae": mae,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "mean_residual": float(np.mean(residuals)),
        "residual_standard_deviation": float(
            np.std(residuals, ddof=1)
        )
        if len(residuals) > 1
        else np.nan,
    }


def save_prediction_plot(
    predictions: pd.DataFrame,
    metrics: dict,
    output_path: Path,
) -> None:
    true_values = predictions["true_solubility"]
    predicted_values = predictions["predicted_solubility"]

    lower = float(min(true_values.min(), predicted_values.min()))
    upper = float(max(true_values.max(), predicted_values.max()))
    padding = 0.05 * (upper - lower if upper > lower else 1.0)
    lower -= padding
    upper += padding

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        true_values,
        predicted_values,
        alpha=0.7,
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("True Solubility")
    ax.set_ylabel("Predicted Solubility")
    ax.set_title("Predicted vs. True Solubility")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13)

    metrics_text = (
        f"RMSE = {metrics['test_rmse']:.3f}\n"
        f"MAE = {metrics['test_mae']:.3f}\n"
        f"Pearson r = {metrics['pearson_correlation']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {output_path}")


def save_residual_plot(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        predictions["predicted_solubility"],
        predictions["residual"],
        alpha=0.7,
    )
    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=2,
        label="Zero Residual",
    )

    ax.set_xlabel("Predicted Solubility")
    ax.set_ylabel("Residual")
    ax.set_title("Solubility Prediction Residuals")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {output_path}")


def save_tables(
    predictions: pd.DataFrame,
    metrics: dict,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = tables_dir / "solubility_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    print(f"Saved Table: {predictions_path}")

    summary_path = tables_dir / "solubility_prediction_summary.csv"
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    print(f"Saved Table: {summary_path}")


def main() -> None:
    args = parse_args()

    try:
        device = get_device(args.device)
        checkpoint = load_checkpoint(args.checkpoint, device)

        (
            _train_loader,
            _valid_loader,
            test_loader,
            _vocab_size,
            _char2idx,
            _idx2char,
            _train_labels,
            loader_reg_mean,
            loader_reg_std,
            _ld50_mean,
            _ld50_std,
        ) = build_dataloaders_multitask(
            batch_size=args.batch_size
        )

        standardise = checkpoint["standardise"]
        reg_mean = float(
            standardise.get(
                "reg_mean",
                float(loader_reg_mean.item()),
            )
        )
        reg_std = float(
            standardise.get(
                "reg_std",
                float(loader_reg_std.item()),
            )
        )

        model = build_model(checkpoint, device)
        predictions = collect_test_predictions(
            model=model,
            test_loader=test_loader,
            device=device,
            reg_mean=reg_mean,
            reg_std=reg_std,
        )
        metrics = calculate_metrics(predictions)

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    figures_dir = args.figures_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()

    print(f"Using Device: {device}")
    print(f"Test Samples: {len(predictions)}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(
        "Pearson Correlation: "
        f"{metrics['pearson_correlation']:.4f}"
    )
    print(
        "Spearman Correlation: "
        f"{metrics['spearman_correlation']:.4f}"
    )

    save_prediction_plot(
        predictions,
        metrics,
        figures_dir / "predicted_vs_true_solubility.png",
    )
    save_residual_plot(
        predictions,
        figures_dir / "solubility_residuals.png",
    )
    save_tables(
        predictions,
        metrics,
        tables_dir,
    )

    print("\nSolubility Prediction Analysis Complete")


if __name__ == "__main__":
    main()