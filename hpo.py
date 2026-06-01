import random
import numpy as np
import torch
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow

from model import VAE
from data import build_dataloaders, MAX_LENGTH
from train import vae_loss, evaluate_loss
from evaluate import evaluate_auroc

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.mps.deterministic = True
torch.use_deterministic_algorithms(True)

HIDDEN_DIM = 512
LATENT_DIM = 128
N_LAYERS = 2
KL_FREE_BITS = 0.5
ANNEAL_EPOCHS = 15  # scaled from 25 to match 30-epoch runs


def make_objective(train_loader, valid_loader, vocab_size, train_labels, device):
    def objective(trial):
        beta_max = trial.suggest_float("beta_max", 0.001, 2.0, log=True)
        gamma = trial.suggest_float("gamma", 0.005, 0.5, log=True)

        model = VAE(vocab_size, MAX_LENGTH, HIDDEN_DIM, LATENT_DIM, N_LAYERS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

        best_val_score = float("inf")

        for epoch in range(30):
            beta = min(beta_max, beta_max * epoch / ANNEAL_EPOCHS)

            model.train()
            for batch, labels in train_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits, mu, log_var, prop_logit = model(batch)
                loss, _, _, _ = vae_loss(
                    logits, batch[:, 1:], mu, log_var, beta,
                    prop_logit=prop_logit, labels=labels,
                    gamma=gamma, kl_free_bits=KL_FREE_BITS,
                    train_labels=train_labels
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            val_recon, _, _ = evaluate_loss(model, valid_loader, beta, gamma, device, KL_FREE_BITS, train_labels)
            val_auroc = evaluate_auroc(model, valid_loader, device)
            val_score = val_recon - val_auroc

            trial.report(val_score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_val_score = min(best_val_score, val_score)

        return best_val_score

    return objective


def run_hpo(n_trials=30):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, _, vocab_size, _, _, train_labels = build_dataloaders()
    print(f"vocab_size: {vocab_size}")

    mlflow.set_tracking_uri("file:///Users/elenipsaromatis/Documents/mol-vae/notebooks/mlruns")
    mlflow.set_experiment("vae_hpo")

    mlflowcb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="val_score",
    )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="vae_loss_weights",
    )

    study.optimize(
        make_objective(train_loader, valid_loader, vocab_size, train_labels, device),
        n_trials=n_trials,
        callbacks=[mlflowcb],
    )

    print("Best params:", study.best_params)
    print("Best score: ", study.best_value)
    return study


if __name__ == "__main__":
    study = run_hpo()