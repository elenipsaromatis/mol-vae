import random
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch

from data import MAX_LENGTH, build_dataloaders
from evaluate import evaluate_auprc, evaluate_auroc
from model import VAE
from train import compute_pos_weights, evaluate_loss, vae_loss


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.backends.mps.is_available():
    torch.backends.mps.deterministic = True

torch.use_deterministic_algorithms(True, warn_only=True)


HIDDEN_DIM = 512
BATCH_SIZE = 64
ROOT = Path(__file__).resolve().parent


def make_objective(
    train_loader,
    valid_loader,
    vocab_size,
    train_labels,
    device,
    n_epochs,
):
    pos_weights = compute_pos_weights(train_labels, device)

    def objective(trial):
        beta_max = trial.suggest_float(
            "beta_max",
            0.004,
            0.5,
            log=True,
        )

        gamma = trial.suggest_float(
            "gamma",
            0.005,
            1,
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
                    "cyp2d6_gamma": gamma,
                    "cyp2c19_gamma": gamma,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "kl_free_bits": kl_free_bits,
                    "latent_dim": latent_dim,
                    "n_layers": n_layers,
                    "prop_hidden_size": prop_hidden_size,
                    "hidden_dim": HIDDEN_DIM,
                    "batch_size": BATCH_SIZE,
                    "cyp2d6_pos_weight": pos_weights[0].item(),
                    "cyp2c19_pos_weight": pos_weights[1].item(),
                    "dataset": "CYP2D6/CYP2C19 overlap",
                    "split": "TDC CYP2D6 scaffold",
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
                        cyp2d6_logit,
                        cyp2c19_logit,
                    ) = model(batch)

                    (
                        loss,
                        recon_loss,
                        kl_loss,
                        cyp2d6_loss,
                        cyp2c19_loss,
                    ) = vae_loss(
                        logits=logits,
                        targets=batch[:, 1:],
                        mu=mu,
                        log_var=log_var,
                        cyp2d6_logit=cyp2d6_logit,
                        cyp2c19_logit=cyp2c19_logit,
                        labels=labels,
                        beta=beta,
                        gamma=gamma,
                        kl_free_bits=kl_free_bits,
                        pos_weights=pos_weights,
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
                    valid_cyp2d6_loss,
                    valid_cyp2c19_loss,
                ) = evaluate_loss(
                    model=model,
                    loader=valid_loader,
                    beta=beta,
                    gamma=gamma,
                    device=device,
                    kl_free_bits=kl_free_bits,
                    pos_weights=pos_weights,
                )

                valid_cyp2d6_auroc = evaluate_auroc(
                    model,
                    valid_loader,
                    device,
                    "cyp2d6",
                )

                valid_cyp2c19_auroc = evaluate_auroc(
                    model,
                    valid_loader,
                    device,
                    "cyp2c19",
                )

                valid_cyp2d6_auprc = evaluate_auprc(
                    model,
                    valid_loader,
                    device,
                    "cyp2d6",
                )

                valid_cyp2c19_auprc = evaluate_auprc(
                    model,
                    valid_loader,
                    device,
                    "cyp2c19",
                )

                mean_valid_prop_loss = 0.5 * (
                    valid_cyp2d6_loss
                    + valid_cyp2c19_loss
                )

                val_score = (
                    valid_recon
                    + mean_valid_prop_loss
                )

                mlflow.log_metrics(
                    {
                        "valid_total": valid_total,
                        "valid_recon": valid_recon,
                        "valid_kl": valid_kl,
                        "valid_cyp2d6_loss": valid_cyp2d6_loss,
                        "valid_cyp2c19_loss": valid_cyp2c19_loss,
                        "valid_mean_prop_loss": mean_valid_prop_loss,
                        "valid_cyp2d6_auroc": valid_cyp2d6_auroc,
                        "valid_cyp2c19_auroc": valid_cyp2c19_auroc,
                        "valid_cyp2d6_auprc": valid_cyp2d6_auprc,
                        "valid_cyp2c19_auprc": valid_cyp2c19_auprc,
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
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    (
        train_loader,
        valid_loader,
        test_loader,
        vocab_size,
        char2idx,
        idx2char,
        train_labels,
    ) = build_dataloaders(
        batch_size=BATCH_SIZE
    )

    print(f"vocab_size: {vocab_size}")
    print(f"overlap training labels shape: {tuple(train_labels.shape)}")

    mlflow.set_tracking_uri(
        (ROOT / "mlruns").resolve().as_uri()
    )

    mlflow.set_experiment(
        "vae-overlap-multitask-hpo"
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
        study_name="vae_overlap_multitask_hpo",
    )

    with mlflow.start_run(
        run_name="hpo_overlap_multitask"
    ):
        study.optimize(
            make_objective(
                train_loader=train_loader,
                valid_loader=valid_loader,
                vocab_size=vocab_size,
                train_labels=train_labels,
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