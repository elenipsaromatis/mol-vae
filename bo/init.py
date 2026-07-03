"""One-time setup helpers for the latent-space solubility/LD50 BO+PFGS search.

Loads a trained VAE checkpoint, re-derives the vocab and the solubility/LD50
overlap set, encodes that overlap set into the VAE's (already-trained-on-
solubility) latent space, fits a GP surrogate for LD50 on those latents, and
builds the InvestigationSpace the search will run over.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from data import MAX_LENGTH, build_dataloaders_multitask
from model import VAE
from paretodo import InvestigationSpace, InvestigationSpaceVariable


def get_device():
    """CPU only, deliberately.

    The optimization loop repeatedly runs a small MLP (reg_predictor) over
    the NSGA2 population every generation. That's cheap enough on CPU that
    MPS/CUDA buys nothing, and repeated small tensor allocation on MPS
    inside a tight loop like this crashes with SIGABRT on Apple Silicon
    (observed during testing).
    """
    return torch.device("cpu")


def load_vae_checkpoint(checkpoint_path, seq_len, device):
    """Reconstruct a VAE from a state_dict-only checkpoint.

    Architecture hyperparameters aren't saved in the checkpoint, so they're
    inferred from tensor shapes (same approach as training.ipynb's
    build_model_from_state_dict).
    """
    state_dict = torch.load(checkpoint_path, map_location=device)

    vocab_size = state_dict["encoder.conv.0.weight"].shape[1]
    hidden_dim = state_dict["encoder.fc.weight"].shape[0]
    latent_dim = state_dict["encoder.fc_mu.weight"].shape[0]
    n_layers = state_dict["decoder.fc_z.weight"].shape[0] // hidden_dim
    prop_hidden_size = state_dict["reg_predictor.net.0.weight"].shape[0]

    model = VAE(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        prop_hidden_size=prop_hidden_size,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_overlap_data(batch_size=64):
    """Rebuild the solubility/LD50 overlap dataloaders and vocab."""
    (
        train_loader,
        valid_loader,
        test_loader,
        vocab_size,
        char2idx,
        idx2char,
        _train_labels,
        sol_mean,
        sol_std,
        ld50_mean,
        ld50_std,
    ) = build_dataloaders_multitask(batch_size=batch_size)
    return SimpleNamespace(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        vocab_size=vocab_size,
        char2idx=char2idx,
        idx2char=idx2char,
        sol_mean=sol_mean,
        sol_std=sol_std,
        ld50_mean=ld50_mean,
        ld50_std=ld50_std,
    )


def encode_to_latent(model, loaders, device):
    """Encode every molecule in the given loaders to latent mu.

    Returns (mu, sol_std_units, ld50_std_units) as numpy arrays; the labels
    are still in standardized units (caller destandardizes as needed).
    """
    model.eval()
    mus, sol_std_units, ld50_std_units = [], [], []
    with torch.no_grad():
        for loader in loaders:
            for batch, labels in loader:
                batch = batch.to(device)
                mu, _ = model.encoder(batch)
                mus.append(mu.cpu())
                sol_std_units.append(labels[:, 0])
                ld50_std_units.append(labels[:, 1])
    mu = torch.cat(mus).numpy()
    sol_std_units = torch.cat(sol_std_units).numpy()
    ld50_std_units = torch.cat(ld50_std_units).numpy()
    return mu, sol_std_units, ld50_std_units


def fit_pca(latent, n_components):
    """Compress the VAE latent space before handing it to NSGA2.

    pymoo's default multi-objective termination periodically computes a
    generational-distance indicator via the `moocore` C extension, whose
    `gd_common` routine overflows a fixed-size stack buffer once n_var
    exceeds ~32 (confirmed empirically: crashes with SIGABRT for n_var>=35,
    fine at n_var<=32, on pymoo 0.6.2 / moocore 0.3.1). Searching in a PCA
    subspace keeps n_var comfortably under that limit; the full latent
    vector is only reconstructed (pca.inverse_transform) right before
    calling the VAE. This also makes the LD50 GP surrogate less
    overparameterized relative to the ~2k overlap points available.
    """
    pca = PCA(n_components=n_components, random_state=11)
    pca.fit(latent)
    return pca


def fit_ld50_gp(latent, ld50_raw):
    """Fit a GP surrogate mapping latent z -> raw LD50 value.

    Uses an isotropic (single shared length-scale) Matern kernel rather than
    ARD: with a few thousand overlap points in a ~100-dim latent space, a
    per-dimension length-scale is both statistically overparameterized and
    prohibitively slow to fit (each L-BFGS step over 100+ hyperparameters
    still needs an O(n^3) Cholesky decomposition).
    """
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
        noise_level=1.0
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=11,
    )
    gp.fit(latent, ld50_raw)
    train_r2 = gp.score(latent, ld50_raw)
    return gp, train_r2


def build_investigation_space(latent, margin_frac=0.1):
    """Build an InvestigationSpace covering the overlap-encoded latents.

    Bounds come from the empirical min/max of the observed latents (padded
    by margin_frac of the range), keeping the search inside the region the
    GP surrogate actually has support over.
    """
    lo = latent.min(axis=0)
    hi = latent.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    pad = margin_frac * span
    variables = [
        InvestigationSpaceVariable(
            name=f"z{i}",
            min=float(lo[i] - pad[i]),
            max=float(hi[i] + pad[i]),
        )
        for i in range(latent.shape[1])
    ]
    return InvestigationSpace(variables=variables)


def _token_id(idx2char, token):
    for idx, ch in idx2char.items():
        if ch == token:
            return idx
    raise ValueError(f"Token {token!r} not found in vocab.")


def generate_smiles_from_latent(model, z, idx2char, device, max_length=MAX_LENGTH):
    """Greedily decode latent vectors z into SMILES strings.

    model.decoder.forward() is teacher-forced (needs the target sequence as
    input), so this steps the same GRU one token at a time instead, seeding
    hidden state exactly as Decoder.forward does.
    """
    model.eval()
    decoder = model.decoder
    z = torch.as_tensor(z, dtype=torch.float32, device=device)
    batch_size = z.size(0)

    sos_idx = _token_id(idx2char, "<sos>")
    eos_idx = _token_id(idx2char, "<eos>")

    with torch.no_grad():
        hidden = decoder.fc_z(z)
        hidden = (
            hidden.reshape(-1, decoder.n_layers, decoder.hidden_dim)
            .permute(1, 0, 2)
            .contiguous()
        )

        token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = [[] for _ in range(batch_size)]

        for _ in range(max_length):
            token_onehot = F.one_hot(token, num_classes=decoder.vocab_size).float()
            gru_input = torch.cat([token_onehot, z], dim=-1).unsqueeze(1)
            output, hidden = decoder.gru(gru_input, hidden)
            logits = decoder.fc_out(output.squeeze(1))
            token = logits.argmax(dim=-1)

            for i in range(batch_size):
                if not finished[i]:
                    sequences[i].append(token[i].item())
            finished = finished | (token == eos_idx)
            if finished.all():
                break

    smiles_list = []
    for seq in sequences:
        chars = [idx2char[idx] for idx in seq]
        smiles = ""
        for ch in chars:
            if ch == "<eos>":
                break
            if ch not in ("<pad>", "<sos>"):
                smiles += ch
        smiles_list.append(smiles)
    return smiles_list


def setup(config):
    """Run the full one-time setup described in bo/config.toml."""
    device = get_device()
    data_ns = load_overlap_data()

    model = load_vae_checkpoint(config["checkpoint_path"], MAX_LENGTH, device)
    if model.encoder.vocab_size != data_ns.vocab_size:
        raise ValueError(
            "Checkpoint vocab_size "
            f"({model.encoder.vocab_size}) does not match freshly-built "
            f"vocab_size ({data_ns.vocab_size}); the TDC dataset or "
            "tokenization may have changed since this checkpoint was trained."
        )

    latent, _sol_std_units, ld50_std_units = encode_to_latent(
        model,
        [data_ns.train_loader, data_ns.valid_loader, data_ns.test_loader],
        device,
    )
    ld50_raw = ld50_std_units * data_ns.ld50_std.item() + data_ns.ld50_mean.item()

    pca = fit_pca(latent, n_components=config["pca_components"])
    latent_pca = pca.transform(latent)

    ld50_gp, ld50_gp_train_r2 = fit_ld50_gp(latent_pca, ld50_raw)
    investigation_space = build_investigation_space(
        latent_pca, margin_frac=config["bounds_margin_frac"]
    )

    return SimpleNamespace(
        device=device,
        model=model,
        idx2char=data_ns.idx2char,
        sol_mean=data_ns.sol_mean.item(),
        sol_std=data_ns.sol_std.item(),
        pca=pca,
        ld50_gp=ld50_gp,
        ld50_gp_train_r2=ld50_gp_train_r2,
        investigation_space=investigation_space,
    )
