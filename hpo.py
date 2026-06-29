import random
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch

from data import MAX_LENGTH, build_dataloaders
from evaluate import evaluate_regression
from model import VAE
from train import evaluate_loss, vae_loss


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.backends.mps.is_available():
    torch.backends.mps.deterministic = True

torch.use_deterministic_algorithms(True, warn_only=True)


HIDDEN_DIM = 512
BATCH_SIZE = 64
ROOT = Path(__file__).resolve().parent

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_objective(
    train_loader,
    valid_loader,
    vocab_size,
    reg_mean,
    reg_std,
    device,
    n_epochs,
):
    def objective(trial):
        beta_max = trial.suggest_float(
            "beta_max",
            1e-4,
            0.1,
            log=True,
        )

        gamma = trial.suggest_float(
            "gamma",
            0.01,
            5.0,
            log=True,
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-4,
            1e-3,
            log=True,
        )

        weight_decay = trial.suggest_float(
            "weight_decay",
            1e-5,
            1e-3,
            log=True,
        )

        dropout = trial.suggest_float(
            "dropout",
            0.1,
            0.6,
        )

        kl_free_bits = trial.suggest_float(
            "kl_free_bits",
            0.3,
            0.8,
        )

        latent_dim = trial.suggest_categorical(
            "latent_dim",
            [64, 100, 128, 256],
        )

        n_layers = trial.suggest_categorical(
            "n_layers",
            [1, 2, 3, 4],
        )

        prop_hidden_size = trial.suggest_categorical(
            "prop_hidden_size",
            [32, 36, 64, 128],
        )

        model = VAE(
            vocab_size=vocab_size,
            seq_len=MAX_LENGTH,
            hidden_dim=HIDDEN_DIM,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout=dropout,
            prop_hidden_size=prop_hidden_size,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=1e-5,
        )

        best_val_score = float("inf")
        anneal_epochs = max(1, n_epochs // 2)

        with mlflow.start_run(
            run_name=f"trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_params(
                {
                    "beta_max": beta_max,
                    "gamma": gamma,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "kl_free_bits": kl_free_bits,
                    "latent_dim": latent_dim,
                    "n_layers": n_layers,
                    "prop_hidden_size": prop_hidden_size,
                    "hidden_dim": HIDDEN_DIM,
                    "batch_size": BATCH_SIZE,
                    "reg_mean": reg_mean.item(),
                    "reg_std": reg_std.item(),
                    "dataset": "Solubility_AqSolDB",
                    "split": "TDC Solubility scaffold",
                }
            )

            for epoch in range(n_epochs):
                beta = min(
                    beta_max,
                    beta_max * epoch / anneal_epochs,
                )

                model.train()

                for batch, labels in train_loader:
                    batch = batch.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    (
                        logits,
                        mu,
                        log_var,
                        reg_pred,
                    ) = model(batch)

                    (
                        loss,
                        recon_loss,
                        kl_loss,
                        reg_loss,
                    ) = vae_loss(
                        logits=logits,
                        targets=batch[:, 1:],
                        mu=mu,
                        log_var=log_var,
                        reg_pred=reg_pred,
                        labels=labels,
                        beta=beta,
                        gamma=gamma,
                        kl_free_bits=kl_free_bits,
                    )

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )

                    optimizer.step()

                scheduler.step()

                (
                    valid_total,
                    valid_recon,
                    valid_kl,
                    valid_reg_loss,
                ) = evaluate_loss(
                    model=model,
                    loader=valid_loader,
                    beta=beta,
                    gamma=gamma,
                    device=device,
                    kl_free_bits=kl_free_bits,
                )

                valid_reg_metrics = evaluate_regression(
                    model,
                    valid_loader,
                    device,
                    reg_mean,
                    reg_std,
                )
                valid_rmse = valid_reg_metrics["rmse"]
                valid_mae = valid_reg_metrics["mae"]

                val_score = valid_recon + valid_reg_loss

                mlflow.log_metrics(
                    {
                        "valid_total": valid_total,
                        "valid_recon": valid_recon,
                        "valid_kl": valid_kl,
                        "valid_reg_loss": valid_reg_loss,
                        "valid_rmse": valid_rmse,
                        "valid_mae": valid_mae,
                        "val_score": val_score,
                        "beta": beta,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

                trial.report(val_score, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                best_val_score = min(
                    best_val_score,
                    val_score,
                )

        return best_val_score

    return objective


def run_hpo(n_trials=50, n_epochs=20):
    device = get_device()
    print(f"Using device: {device}")

    (
        train_loader,
        valid_loader,
        test_loader,
        vocab_size,
        char2idx,
        idx2char,
        train_labels,
        reg_mean,
        reg_std,
    ) = build_dataloaders(
        batch_size=BATCH_SIZE
    )

    print(f"vocab_size: {vocab_size}")
    print(f"training labels shape: {tuple(train_labels.shape)}")

    mlflow.set_tracking_uri(
        (ROOT / "mlruns").resolve().as_uri()
    )

    mlflow.set_experiment(
        "vae-solubility-hpo"
    )

    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="vae_solubility_hpo",
    )

    with mlflow.start_run(
        run_name="hpo_solubility"
    ):
        study.optimize(
            make_objective(
                train_loader=train_loader,
                valid_loader=valid_loader,
                vocab_size=vocab_size,
                reg_mean=reg_mean,
                reg_std=reg_std,
                device=device,
                n_epochs=n_epochs,
            ),
            n_trials=n_trials,
        )

    print("Best params:", study.best_params)
    print("Best score:", study.best_value)

    return study


if __name__ == "__main__":
    study = run_hpo()