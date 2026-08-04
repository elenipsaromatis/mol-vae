"""Problem definition for BO of LD50 toxicity.

Loads a trained VAE checkpoint, encodes the overlap molecules into latent space,
and returns LD50 labels for candidate selection BO.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import build_dataloaders_multitask, build_ld50_full_dataframe, encode_smiles
from model import VAE


@dataclass
class BOProblem:
    X: np.ndarray              # VAE latent embeddings, shape: (n_molecules, latent_dim)
    y: np.ndarray              # standardised LD50 labels as observed by BO, shape: (n_molecules,)
    y_raw: np.ndarray          # raw LD50 labels as observed by BO, shape: (n_molecules,)
    y_raw_true: np.ndarray     # ground-truth raw LD50 labels, never perturbed, shape: (n_molecules,)
    smiles: np.ndarray         # canonical SMILES, dtype object, shape: (n_molecules,)
    ld50_mean: float
    ld50_std: float
    in_overlap: np.ndarray     # True where molecule was in the solubility x LD50 overlap
                                # the VAE encoder was trained on, shape: (n_molecules,)


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
    smiles_parts = []

    with torch.no_grad():
        for old_loader in (train_loader, valid_loader, test_loader):
            smiles_parts.extend(list(old_loader.dataset.smiles))

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
    smiles = np.array(smiles_parts, dtype=object)

    assert smiles.shape[0] == X.shape[0], (
        f"SMILES ({smiles.shape[0]}) and X ({X.shape[0]}) are misaligned"
    )

    return BOProblem(
        X=X,
        y=y,
        y_raw=y_raw,
        y_raw_true=y_raw.copy(),
        smiles=smiles,
        ld50_mean=ld50_mean,
        ld50_std=ld50_std,
        in_overlap=np.ones(smiles.shape[0], dtype=bool),
    )


def load_bo_problem_full_ld50(
    checkpoint_path: str | Path,
    batch_size: int = 64,
    device: str | None = None,
) -> BOProblem:
    """Build the BO candidate pool from the full LD50_Zhu dataset.

    The VAE encoder is frozen (trained on the solubility x LD50 overlap
    only, see `load_bo_problem`). Encoding the rest of LD50_Zhu tests how
    well that overlap-trained latent space extrapolates to molecules the
    encoder never reconstructed during training.

    The training-time `char2idx` is rebuilt from the overlap data (not from
    the full LD50 set) so token indices line up exactly with the trained
    embedding weights. Full-LD50 molecules containing a character absent
    from that vocab cannot be encoded and are dropped. LD50 values are
    standardised with the checkpoint's own `ld50_mean`/`ld50_std` (computed
    on the overlap at training time), not recomputed on the full set, so
    results stay on the same scale as the overlap-pool runs.
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

    # Rebuild the exact training-time vocab (deterministic: same TDC split,
    # same sorted-character construction as in build_dataloaders_multitask).
    train_loader, valid_loader, test_loader, *_rest = build_dataloaders_multitask(
        batch_size=batch_size
    )
    overlap_smiles = set(train_loader.dataset.smiles)
    overlap_smiles.update(valid_loader.dataset.smiles)
    overlap_smiles.update(test_loader.dataset.smiles)

    all_overlap_chars = set("".join(overlap_smiles))
    char2idx = {c: i for i, c in enumerate(
        ["<pad>", "<sos>", "<eos>"] + sorted(all_overlap_chars)
    )}
    if len(char2idx) != build["vocab_size"]:
        raise RuntimeError(
            f"Rebuilt vocab size ({len(char2idx)}) does not match checkpoint "
            f"vocab_size ({build['vocab_size']}); char2idx no longer lines up "
            "with the trained embedding weights."
        )

    standardise = checkpoint["standardise"]
    ld50_mean = float(standardise["ld50_mean"])
    ld50_std = float(standardise["ld50_std"])

    ld50_full = build_ld50_full_dataframe()

    known_chars = set(char2idx)
    encodable = ld50_full["Drug"].apply(lambda s: set(s).issubset(known_chars))
    n_dropped = int((~encodable).sum())
    if n_dropped:
        print(
            f"[full-ld50] dropping {n_dropped}/{len(ld50_full)} molecules "
            "with characters absent from the overlap-trained vocab"
        )
    ld50_full = ld50_full[encodable].reset_index(drop=True)

    smiles = ld50_full["Drug"].to_numpy(dtype=object)
    y_raw = ld50_full["Y_ld50"].to_numpy(dtype=np.float64)
    in_overlap = np.array(
        [s in overlap_smiles for s in smiles], dtype=bool
    )

    encoded = torch.tensor(
        [encode_smiles(s, char2idx, build["seq_len"]) for s in smiles],
        dtype=torch.long,
    )

    X_parts = []
    with torch.no_grad():
        for start in range(0, encoded.shape[0], batch_size):
            batch = encoded[start:start + batch_size].to(device)
            mu, _ = model.encoder(batch)
            X_parts.append(mu.cpu().numpy())

    X = np.vstack(X_parts).astype(np.float64)
    y = (y_raw - ld50_mean) / ld50_std

    assert smiles.shape[0] == X.shape[0], (
        f"SMILES ({smiles.shape[0]}) and X ({X.shape[0]}) are misaligned"
    )

    return BOProblem(
        X=X,
        y=y,
        y_raw=y_raw,
        y_raw_true=y_raw.copy(),
        smiles=smiles,
        ld50_mean=ld50_mean,
        ld50_std=ld50_std,
        in_overlap=in_overlap,
    )