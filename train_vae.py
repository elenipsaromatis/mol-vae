"""VAE trained on the Solubility x LD50 overlap for BO on LD50.

Trains reconstruction, KL, and a solubility head on the overlap molecules.
LD50 is not a training target. Its labels and standardisation stats ride
along in the loader and checkpoint so the downstream BO stage can fit a GP
on LD50 over the latent z of these same molecules.

The VAE still builds an ld50_predictor head, but it receives no supervised
gradient here and stays untrained. BO reads LD50 from its own GP, not from
that head. Any LD50 metrics printed by evaluate_test come from the untrained
head and are not meaningful.
"""

import random
from pathlib import Path
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import MAX_LENGTH, build_dataloaders_multitask
from evaluate import evaluate, evaluate_regression, evaluate_test
from model import VAE


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.backends.mps.is_available():
    torch.backends.mps.deterministic = True

torch.use_deterministic_algorithms(True, warn_only=True)


HIDDEN_DIM = 512
LATENT_DIM = 64
N_LAYERS = 2
PROP_HIDDEN_SIZE = 64
WEIGHT_DECAY = 0.00010877641809866005
EPOCHS = 50
BETA_MAX = 0.0002910947584036602
ANNEAL_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.0008933357073260318
KL_FREE_BITS = 0.4956230181587681
GAMMA = 0.049215108608914884
DROPOUT = 0.29635645365390845
PATIENCE = 3
MIN_DELTA = 1e-3

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def vae_loss(logits, targets, mu, log_var, reg_pred, labels, beta, gamma, kl_free_bits):
    recon_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    ) / (targets != 0).sum()

    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=-1))

    reg_loss = nn.MSELoss()(reg_pred, labels[:, 0])

    loss = recon_loss + beta * kl_loss + gamma * reg_loss
    return loss, recon_loss, kl_loss, reg_loss


def evaluate_loss(model, loader, beta, gamma, device, kl_free_bits):
    model.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "reg": 0.0}
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            labels = labels.to(device)
            outputs = model(batch)
            values = vae_loss(
                logits=outputs[0],
                targets=batch[:, 1:],
                mu=outputs[1],
                log_var=outputs[2],
                reg_pred=outputs[3],
                labels=labels,
                beta=beta,
                gamma=gamma,
                kl_free_bits=kl_free_bits,
            )
            for key, value in zip(totals, values):
                totals[key] += value.item()
    n = len(loader)
    return tuple(v / n for v in totals.values())


def train():
    device = get_device()
    print(f"Using device: {device}")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

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
        ld50_mean,
        ld50_std,
    ) = build_dataloaders_multitask(batch_size=BATCH_SIZE)

    MLFLOW_DIR = ROOT / "notebooks" / "mlruns"
    mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
    experiment = mlflow.get_experiment_by_name("mol-vae")
    if experiment is None:
        raise RuntimeError(f"MLflow experiment 'mol-vae' not found in {MLFLOW_DIR.resolve()}")
    mlflow.set_experiment(experiment_id=experiment.experiment_id)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        run_name = f"vae_solubility_{run_id[:8]}"
        mlflow.set_tag("mlflow.runName", run_name)

        model_path = CHECKPOINT_DIR / f"{run_name}_final.pth"
        best_model_path = CHECKPOINT_DIR / f"{run_name}_best.pth"
        writer = SummaryWriter(log_dir=str(ROOT / "runs" / run_id[:8]))

        mlflow.log_params({
            "vocab_size": vocab_size,
            "max_length": MAX_LENGTH,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
            "prop_hidden_size": PROP_HIDDEN_SIZE,
            "epochs": EPOCHS,
            "anneal_epochs": ANNEAL_EPOCHS,
            "beta_max": BETA_MAX,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "kl_free_bits": KL_FREE_BITS,
            "gamma": GAMMA,
            "reg_mean": reg_mean.item(),
            "reg_std": reg_std.item(),
            "ld50_mean": ld50_mean.item(),
            "ld50_std": ld50_std.item(),
            "dataset": "Solubility_AqSolDB x LD50_Zhu overlap",
            "split": "TDC Solubility scaffold, LD50 inner-joined",
            "trained_head": "solubility only (LD50 reserved for BO)",
            "scheduler": "CosineAnnealingLR",
        })

        model = VAE(
            vocab_size=vocab_size,
            seq_len=MAX_LENGTH,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
            prop_hidden_size=PROP_HIDDEN_SIZE,
        ).to(device)

        def build_checkpoint():
            return {
                "state_dict": model.state_dict(),
                "build": {
                    "vocab_size": vocab_size,
                    "seq_len": MAX_LENGTH,
                    "hidden_dim": HIDDEN_DIM,
                    "latent_dim": LATENT_DIM,
                    "n_layers": N_LAYERS,
                    "dropout": DROPOUT,
                    "prop_hidden_size": PROP_HIDDEN_SIZE,
                },
                "standardise": {
                    "reg_mean": reg_mean.item(),
                    "reg_std": reg_std.item(),
                    "ld50_mean": ld50_mean.item(),
                    "ld50_std": ld50_std.item(),
                },
            }

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        best_val_loss = float("inf")
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "reg": 0.0, "grad": 0.0}
            beta = min(BETA_MAX, BETA_MAX * epoch / ANNEAL_EPOCHS)

            for batch, labels in train_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = model(batch)
                values = vae_loss(
                    logits=outputs[0],
                    targets=batch[:, 1:],
                    mu=outputs[1],
                    log_var=outputs[2],
                    reg_pred=outputs[3],
                    labels=labels,
                    beta=beta,
                    gamma=GAMMA,
                    kl_free_bits=KL_FREE_BITS,
                )

                values[0].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                for key, value in zip(["loss", "recon", "kl", "reg"], values):
                    totals[key] += value.item()
                totals["grad"] += grad_norm.item()

            n = len(train_loader)
            train_metrics = {k: v / n for k, v in totals.items()}
            scheduler.step()

            valid_total, valid_recon, valid_kl, valid_reg = evaluate_loss(
                model, valid_loader, beta, GAMMA, device, KL_FREE_BITS
            )
            reg_metrics = evaluate_regression(
                model, valid_loader, device, reg_mean, reg_std, head="reg_predictor"
            )
            exact_acc, token_acc, validity = evaluate(
                model, train_loader.dataset, vocab_size, device, idx2char
            )
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:>2}/{EPOCHS}, beta: {beta:.4f}, "
                f"Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon']:.4f}, "
                f"KL: {train_metrics['kl']:.4f}, Reg: {train_metrics['reg']:.4f}, LR: {lr:.1e}"
            )
            print(
                f"         Valid loss: {valid_total:.4f}, Sol RMSE: {reg_metrics['rmse']:.4f}, "
                f"MAE: {reg_metrics['mae']:.4f}, Recon acc: {exact_acc:.3f}, "
                f"Token acc: {token_acc:.3f}, Validity: {validity:.3f}"
            )

            metrics = {
                "train_loss": train_metrics["loss"],
                "train_recon": train_metrics["recon"],
                "train_kl": train_metrics["kl"],
                "train_reg_loss": train_metrics["reg"],
                "valid_loss": valid_total,
                "valid_recon": valid_recon,
                "valid_kl": valid_kl,
                "valid_reg_loss": valid_reg,
                "valid_rmse": reg_metrics["rmse"],
                "valid_mae": reg_metrics["mae"],
                "grad_norm": train_metrics["grad"],
                "beta": beta,
                "learning_rate": lr,
                "recon_acc": exact_acc,
                "token_acc": token_acc,
                "validity": validity,
            }
            mlflow.log_metrics(metrics, step=epoch + 1)
            for name, value in metrics.items():
                writer.add_scalar(name, value, epoch + 1)

            if epoch >= ANNEAL_EPOCHS:
                if valid_total < best_val_loss - MIN_DELTA:
                    best_val_loss = valid_total
                    best_epoch = epoch
                    epochs_no_improve = 0
                    torch.save(build_checkpoint(), best_model_path)
                    mlflow.log_metric("best_valid_loss", best_val_loss, step=epoch + 1)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        print(
                            f"Early stopping at epoch {epoch + 1}, "
                            f"best epoch {best_epoch + 1} (valid_loss {best_val_loss:.4f})"
                        )
                        break

        torch.save(build_checkpoint(), model_path)

        if best_epoch >= 0:
            model.load_state_dict(torch.load(best_model_path)["state_dict"])
            print(f"Reloaded best model from epoch {best_epoch + 1}")
            eval_checkpoint = best_model_path
        else:
            eval_checkpoint = model_path

        test_metrics = evaluate_test(
            model, test_loader, vocab_size, idx2char, device,
            reg_mean, reg_std, ld50_mean, ld50_std,
        )
        print(
            "\nTest: "
            f"Recon acc: {test_metrics['recon_acc']:.3f}, "
            f"Validity: {test_metrics['validity']:.3f}, "
            f"Sol RMSE: {test_metrics['rmse']:.3f}, MAE: {test_metrics['mae']:.3f}"
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.set_tag("final_checkpoint", str(model_path.resolve()))
        if best_epoch >= 0:
            mlflow.set_tag("best_checkpoint", str(best_model_path.resolve()))
        mlflow.set_tag("eval_checkpoint", str(eval_checkpoint.resolve()))
        writer.close()

        print(f"\nFinal model saved as {model_path}")
        if best_epoch >= 0:
            print(f"Best model saved as {best_model_path}")
        print(f"Run ID: {run_id}")


if __name__ == "__main__":
    train()