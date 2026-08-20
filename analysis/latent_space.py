from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import plotly.express as px

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Charter", "Georgia", "DejaVu Serif"],
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_dataloaders_multitask 
from model import VAE 


DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints" / "vae_solubility_112564ed_best.pth"
)
DEFAULT_FIGURES_DIR = ROOT / "analysis" / "figures" / "final_vae"
DEFAULT_TABLES_DIR = ROOT / "analysis" / "tables" / "final_vae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCA Visualizations Of The VAE Latent Space."
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
        help="Batch Size Used For Latent Encoding.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device Used For Encoding.",
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

    required = {"state_dict", "build", "standardise"}
    missing = required.difference(checkpoint)
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


def make_ordered_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def encode_split(
    model: VAE,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    sol_mean: float,
    sol_std: float,
    ld50_mean: float,
    ld50_std: float,
) -> pd.DataFrame:
    latent_batches = []
    label_batches = []

    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            mu, _ = model.encoder(batch)

            latent_batches.append(mu.cpu())
            label_batches.append(labels.cpu())

    latent = torch.cat(latent_batches, dim=0).numpy()
    labels = torch.cat(label_batches, dim=0).numpy()

    solubility = labels[:, 0] * sol_std + sol_mean
    ld50 = labels[:, 1] * ld50_std + ld50_mean

    smiles = loader.dataset.smiles
    if smiles is None:
        smiles = [""] * len(latent)

    frame = pd.DataFrame(
        {
            "split": split_name,
            "split_index": np.arange(len(latent), dtype=int),
            "smiles": list(smiles),
            "solubility": solubility,
            "ld50": ld50,
        }
    )

    for index in range(latent.shape[1]):
        frame[f"latent_{index + 1}"] = latent[:, index]

    return frame


def add_pca_coordinates(frame: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    latent_columns = [
        column for column in frame.columns if column.startswith("latent_")
    ]

    pca = PCA(n_components=3)
    coordinates = pca.fit_transform(frame[latent_columns].to_numpy())
    print("PCA coordinates shape:", coordinates.shape)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())
    output = frame.copy()
    output["pc1"] = coordinates[:, 0]
    output["pc2"] = coordinates[:, 1]
    output["pc3"] = coordinates[:, 2]
    output.insert(
        0,
        "recipe_id",
        np.arange(len(output), dtype=int),
    )

    return output, pca


def fit_ld50_only_pca(frame: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    """Independent 2-component PCA fit used only for the standalone LD50 plot.

    Fit separately from add_pca_coordinates() so the LD50-only view isn't
    tied to the shared solubility/combined/comparison-table axes.
    """
    latent_columns = [
        column for column in frame.columns if column.startswith("latent_")
    ]

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(frame[latent_columns].to_numpy())

    output = frame.copy()
    output["pc1"] = coordinates[:, 0]
    output["pc2"] = coordinates[:, 1]

    return output, pca


def save_single_property_plot(
    frame: pd.DataFrame,
    property_column: str,
    property_label: str,
    title: str,
    output_path: Path,
    explained_variance: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        frame["pc1"],
        frame["pc2"],
        c=frame[property_column],
        alpha=0.75,
    )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(property_label)

    ax.set_xlabel(
    f"Principal Component 1 "
    f"({explained_variance[0] * 100:.1f}%)"
    )
    ax.set_ylabel(
    f"Principal Component 2 "
    f"({explained_variance[1] * 100:.1f}%)"
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {output_path}")


def save_combined_plot(
    frame: pd.DataFrame,
    output_path: Path,
    explained_variance: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Both colorbars use identical fraction/pad/aspect so they render at the
    # exact same size regardless of the two properties' different value
    # ranges/tick-label widths (the thesis PDF's version let the second
    # colorbar auto-scale differently -- e.g. gaining "extend" arrow caps --
    # which made the two panels visually mismatched).
    colorbar_kwargs = dict(fraction=0.046, pad=0.04, aspect=20, extend="neither")

    sol_scatter = axes[0].scatter(
        frame["pc1"],
        frame["pc2"],
        c=frame["solubility"],
        alpha=0.75,
    )
    sol_colorbar = fig.colorbar(sol_scatter, ax=axes[0], **colorbar_kwargs)
    sol_colorbar.ax.tick_params(labelsize=12)
    axes[0].set_xlabel(
        f"Principal Component 1 "
        f"({explained_variance[0] * 100:.1f}%)"
    )
    axes[0].set_ylabel(
        f"Principal Component 2 "
        f"({explained_variance[1] * 100:.1f}%)"
    )
    axes[0].set_title("Latent Space Coloured By Solubility")
    axes[0].grid(True, alpha=0.25)

    ld50_scatter = axes[1].scatter(
        frame["pc1"],
        frame["pc2"],
        c=frame["ld50"],
        alpha=0.75,
    )
    ld50_colorbar = fig.colorbar(ld50_scatter, ax=axes[1], **colorbar_kwargs)
    ld50_colorbar.ax.tick_params(labelsize=12)
    axes[1].set_xlabel(
        f"Principal Component 1 "
        f"({explained_variance[0] * 100:.1f}%)"
    )
    axes[1].set_ylabel(
        f"Principal Component 2 "
        f"({explained_variance[1] * 100:.1f}%)"
    )
    axes[1].set_title("Latent Space Coloured By pLD50")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Latent Space Property Comparison")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure: {output_path}")


def save_tables(
    frame: pd.DataFrame,
    pca: PCA,
    ld50_pca: PCA,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    pca_path = tables_dir / "latent_space_pca.csv"
    frame.to_csv(pca_path, index=False)
    print(f"Saved Table: {pca_path}")

    summary = pd.DataFrame(
        [
            {
                "n_molecules": int(len(frame)),
                "latent_dimension": int(
                    sum(
                        column.startswith("latent_")
                        for column in frame.columns
                    )
                ),
                "pc1_explained_variance_ratio": float(
                    pca.explained_variance_ratio_[0]
                ),
                "pc2_explained_variance_ratio": float(
                    pca.explained_variance_ratio_[1]
                ),
                "pc3_explained_variance_ratio": float(
                    pca.explained_variance_ratio_[2]
                ),
                "total_explained_variance_ratio": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "solubility_pc1_correlation": float(
                    frame["solubility"].corr(frame["pc1"])
                ),
                "solubility_pc2_correlation": float(
                    frame["solubility"].corr(frame["pc2"])
                ),
                "solubility_pc3_correlation": float(
                    frame["solubility"].corr(frame["pc3"])
                ),
                "ld50_pc1_correlation": float(
                    frame["ld50"].corr(frame["pc1"])
                ),
                "ld50_pc2_correlation": float(
                    frame["ld50"].corr(frame["pc2"])
                ),
                "ld50_pc3_correlation": float(
                    frame["ld50"].corr(frame["pc3"])
                ),
                "solubility_ld50_correlation": float(
                    frame["solubility"].corr(frame["ld50"])
                ),
                "ld50_only_pc1_explained_variance_ratio": float(
                    ld50_pca.explained_variance_ratio_[0]
                ),
                "ld50_only_pc2_explained_variance_ratio": float(
                    ld50_pca.explained_variance_ratio_[1]
                ),
                "ld50_only_total_explained_variance_ratio": float(
                    ld50_pca.explained_variance_ratio_.sum()
                ),
            }
        ]
    )

    summary_path = tables_dir / "latent_space_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved Table: {summary_path}")

def save_interactive_3d_plot(
    frame: pd.DataFrame,
    property_column: str,
    property_label: str,
    title: str,
    output_path: Path,
    explained_variance: np.ndarray,
) -> None:
    plot_frame = frame.copy()

    figure = px.scatter_3d(
        plot_frame,
        x="pc1",
        y="pc2",
        z="pc3",
        color=property_column,
        hover_data={
            "recipe_id": True,
            "smiles": True,
            "split": True,
            "solubility": ":.3f",
            "ld50": ":.3f",
            "pc1": ":.3f",
            "pc2": ":.3f",
            "pc3": ":.3f",
        },
        labels={
            "pc1": (
                f"Principal Component 1 "
                f"({explained_variance[0] * 100:.1f}%)"
            ),
            "pc2": (
                f"Principal Component 2 "
                f"({explained_variance[1] * 100:.1f}%)"
            ),
            "pc3": (
                f"Principal Component 3 "
                f"({explained_variance[2] * 100:.1f}%)"
            ),
            property_column: property_label,
        },
        title=title,
        opacity=0.65,
    )

    figure.update_traces(
        marker={
            "size": 3,
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.write_html(output_path)

    print(f"Saved Interactive Figure: {output_path}")

def save_3d_property_plot(
    frame: pd.DataFrame,
    property_column: str,
    property_label: str,
    title: str,
    output_path: Path,
    explained_variance: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        frame["pc1"],
        frame["pc2"],
        frame["pc3"],
        c=frame[property_column],
        s=18,
        alpha=0.60,
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.75,
        pad=0.10,
    )
    colorbar.set_label(property_label)

    ax.set_xlabel(
        f"Principal Component 1 "
        f"({explained_variance[0] * 100:.1f}%)"
    )
    ax.set_ylabel(
        f"Principal Component 2 "
        f"({explained_variance[1] * 100:.1f}%)"
    )
    ax.set_zlabel(
        f"Principal Component 3 "
        f"({explained_variance[2] * 100:.1f}%)"
    )

    ax.set_title(title)
    ax.view_init(elev=25, azim=45)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved Figure: {output_path}")

def main() -> None:
    args = parse_args()

    try:
        device = get_device(args.device)
        checkpoint = load_checkpoint(args.checkpoint, device)
        model = build_model(checkpoint, device)

        (
            train_loader,
            valid_loader,
            test_loader,
            _vocab_size,
            _char2idx,
            _idx2char,
            _train_labels,
            loader_sol_mean,
            loader_sol_std,
            loader_ld50_mean,
            loader_ld50_std,
        ) = build_dataloaders_multitask(
            batch_size=args.batch_size
        )

        standardise = checkpoint["standardise"]
        sol_mean = float(
            standardise.get("reg_mean", loader_sol_mean.item())
        )
        sol_std = float(
            standardise.get("reg_std", loader_sol_std.item())
        )
        ld50_mean = float(
            standardise.get("ld50_mean", loader_ld50_mean.item())
        )
        ld50_std = float(
            standardise.get("ld50_std", loader_ld50_std.item())
        )

        split_frames = []
        for split_name, source_loader in (
            ("Train", train_loader),
            ("Validation", valid_loader),
            ("Test", test_loader),
        ):
            ordered_loader = make_ordered_loader(
                source_loader.dataset,
                args.batch_size,
            )
            split_frames.append(
                encode_split(
                    model=model,
                    loader=ordered_loader,
                    split_name=split_name,
                    device=device,
                    sol_mean=sol_mean,
                    sol_std=sol_std,
                    ld50_mean=ld50_mean,
                    ld50_std=ld50_std,
                )
            )

        latent_frame = pd.concat(
            split_frames,
            ignore_index=True,
        )
        latent_frame, pca = add_pca_coordinates(latent_frame)
        ld50_only_frame, ld50_pca = fit_ld50_only_pca(latent_frame)

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    figures_dir = args.figures_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()

    print(f"Using Device: {device}")
    print(f"Encoded Molecules: {len(latent_frame)}")
    print(
        "Total Explained Variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )
    print(
        "LD50-Only Explained Variance: "
        f"{ld50_pca.explained_variance_ratio_.sum():.4f}"
    )

    save_single_property_plot(
        frame=latent_frame,
        property_column="solubility",
        property_label="Solubility",
        title="Latent Space Colored By Solubility",
        output_path=figures_dir / "latent_space_solubility.png",
        explained_variance=pca.explained_variance_ratio_,
    )
    save_single_property_plot(
        frame=ld50_only_frame,
        property_column="ld50",
        property_label="LD50",
        title="Latent Space Colored By LD50",
        output_path=figures_dir / "latent_space_ld50.png",
        explained_variance=ld50_pca.explained_variance_ratio_,
    )
    save_combined_plot(
        frame=latent_frame,
        output_path=figures_dir / "latent_space_properties.png",
        explained_variance=pca.explained_variance_ratio_,
    )
    save_interactive_3d_plot(
        frame=latent_frame,
        property_column="solubility",
        property_label="Solubility",
        title="Three-Dimensional Latent Space Colored By Solubility",
        output_path=(
            figures_dir / "latent_space_3d_solubility.html"
        ),
        explained_variance=pca.explained_variance_ratio_,
    )
    save_interactive_3d_plot(
        frame=latent_frame,
        property_column="ld50",
        property_label="LD50",
        title="Three-Dimensional Latent Space Colored By LD50",
        output_path=(
            figures_dir / "latent_space_3d_ld50.html"
        ),
        explained_variance=pca.explained_variance_ratio_,
    )
    save_3d_property_plot(
        frame=latent_frame,
        property_column="solubility",
        property_label="Solubility",
        title="Three-Dimensional Latent Space Colored By Solubility",
        output_path=(
            figures_dir / "latent_space_3d_solubility.png"
        ),
        explained_variance=pca.explained_variance_ratio_,
    )
    save_3d_property_plot(
        frame=latent_frame,
        property_column="ld50",
        property_label="LD50",
        title="Three-Dimensional Latent Space Colored By LD50",
        output_path=(
            figures_dir / "latent_space_3d_ld50.png"
        ),
        explained_variance=pca.explained_variance_ratio_,
    )
    save_tables(
        frame=latent_frame,
        pca=pca,
        ld50_pca=ld50_pca,
        tables_dir=tables_dir,
    )

    print("\nLatent Space Analysis Complete")


if __name__ == "__main__":
    main()