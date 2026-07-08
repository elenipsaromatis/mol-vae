"""Problem definition for BO of LD50 toxicity.

Loads a trained VAE checkpoint, encodes the overlap molecules into latent space,
and returns LD50 labels for candidate selection BO. 
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import build_dataloaders_multitask
from model import VAE


@dataclass
class BOProblem:
    X: np.ndarray              # VAE latent embeddings, shape: (n_molecules, latent_dim)
    y: np.ndarray              # standardised LD50 labels, shape: (n_molecules,)
    y_raw: np.ndarray          # raw LD50 labels, shape: (n_molecules,)
    ld50_mean: float
    ld50_std: float


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_bo_problem(
    checkpoint_path: str | Path,
    batch_size: int = 64,
    device: str | None = None,
    ld50_col: int = 1,
) -> BOProblem:
    """Create the fixed candidate pool for BO.

    Assumes build_dataloaders_multitask returns labels where:
    labels[:, 0] = solubility
    labels[:, 1] = LD50
    """
    device = get_device(device)
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    build = checkpoint["build"]

    model = VAE(
        vocab_size=build["vocab_size"],
        seq_len=build["seq_len"],
        hidden_dim=build["hidden_dim"],
        latent_dim=build["latent_dim"],
        n_layers=build["n_layers"],
        dropout=build["dropout"],
        prop_hidden_size=build["prop_hidden_size"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    train_loader, valid_loader, test_loader, *_rest = build_dataloaders_multitask(
        batch_size=batch_size
    )

    standardise = checkpoint["standardise"]
    ld50_mean = float(standardise["ld50_mean"])
    ld50_std = float(standardise["ld50_std"])

    X_parts = []
    y_parts = []

    with torch.no_grad():
        for old_loader in (train_loader, valid_loader, test_loader):
            loader = DataLoader(old_loader.dataset, batch_size=batch_size, shuffle=False)

            for batch, labels in loader:
                batch = batch.to(device)
                labels = labels.to(device)

                outputs = model(batch)
                mu = outputs[1]

                X_parts.append(mu.cpu().numpy())
                y_parts.append(labels[:, ld50_col].cpu().numpy())

    X = np.vstack(X_parts).astype(np.float64)
    y = np.concatenate(y_parts).astype(np.float64)
    y_raw = y * ld50_std + ld50_mean

    return BOProblem(
        X=X,
        y=y,
        y_raw=y_raw,
        ld50_mean=ld50_mean,
        ld50_std=ld50_std,
    )