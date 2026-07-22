"""VAE trained on the Solubility x LD50 overlap with both properties supervised.

Trains reconstruction, KL, a solubility head, and an LD50 head jointly on the
overlap molecules. Both property losses enter the objective through
gamma * (reg_loss + ld50_loss).

LD50 labels and standardisation stats are used directly as training targets
here, unlike train_vae.py, which reserves LD50 for a downstream BO stage. Use
this file when you want the model to predict LD50 from z directly rather than
fitting a separate GP.
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
LATENT_DIM = 100
N_LAYERS = 3
PROP_HIDDEN_SIZE = 32
WEIGHT_DECAY = 4.3632866875951974e-05
EPOCHS = 50
BETA_MAX = 0.00039913734909661076
ANNEAL_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.0008526577085759847
KL_FREE_BITS = 0.7067880262850101
GAMMA = 0.06929960590143402
DROPOUT = 0.20784956926076428
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


def vae_loss(
    logits,
    targets,
    mu,
    log_var,
    reg_pred,
    ld50_pred,
    labels,
    beta,
    gamma,
    kl_free_bits,
):
    recon_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    ) / (targets != 0).sum()

    kl_per_dim = -0.5 * (
        1 + log_var - mu.pow(2) - log_var.exp()
    )
    kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=-1))

    reg_loss = nn.MSELoss()(reg_pred, labels[:, 0])
    ld50_loss = nn.MSELoss()(ld50_pred, labels[:, 1])

    loss = recon_loss + beta * kl_loss + gamma * (reg_loss + ld50_loss)

    return loss, recon_loss, kl_loss, reg_loss, ld50_loss


def evaluate_loss(
    model,
    loader,
    beta,
    gamma,
    device,
    kl_free_bits,
):
    model.eval()
    totals = {
        "total": 0.0,
        "recon": 0.0,
        "kl": 0.0,
        "reg": 0.0,
        "ld50": 0.0,
    }

    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            labels = labels.to(device)

            outputs = model(batch)
            loss_values = vae_loss(
                logits=outputs[0],
                targets=batch[:, 1:],
                mu=outputs[1],
                log_var=outputs[2],
                reg_pred=outputs[3],
                ld50_pred=outputs[4],
                labels=labels,
                beta=beta,
                gamma=gamma,
                kl_free_bits=kl_free_bits,
            )

            for key, value in zip(totals, loss_values):
                totals[key] += value.item()

    number_of_batches = len(loader)
    return tuple(value / number_of_batches for value in totals.values())


def train():
    if mlflow is None:
        raise ImportError("MLflow is required. Install it with `pip install mlflow`.")
    if SummaryWriter is None:
        raise ImportError(
            "TensorBoard is required. Install it with `pip install tensorboard`."
        )

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
        raise RuntimeError(
            f"Existing MLflow experiment 'mol-vae' was not found in "
            f"{MLFLOW_DIR.resolve()}"
        )
    mlflow.set_experiment(experiment_id=experiment.experiment_id)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(
        f"MLflow experiment: {experiment.name} "
        f"(ID: {experiment.experiment_id})"
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        run_name = f"vae_solubility_ld50_{run_id[:8]}"
        mlflow.set_tag("mlflow.runName", run_name)

        model_path = CHECKPOINT_DIR / f"{run_name}_final.pth"
        best_model_path = CHECKPOINT_DIR / f"{run_name}_best.pth"
        writer = SummaryWriter(log_dir=str(ROOT / "runs" / run_id[:8]))

        mlflow.log_params(
            {
                "vocab_size": vocab_size,
                "max_length": MAX_LENGTH,
                "hidden_dim": HIDDEN_DIM,
                "latent_dim": LATENT_DIM,
                "n_layers": N_LAYERS,
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
                "dataset": "Solubility_AqSolDB + LD50_Zhu",
                "split": "TDC Solubility scaffold, LD50 inner-joined",
                "scheduler": "CosineAnnealingLR",
            }
        )

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

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            totals = {
                "loss": 0.0,
                "recon": 0.0,
                "kl": 0.0,
                "reg": 0.0,
                "ld50": 0.0,
                "grad": 0.0,
            }

            beta = min(
                BETA_MAX,
                BETA_MAX * epoch / ANNEAL_EPOCHS,
            )

            for batch, labels in train_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = model(batch)
                loss_values = vae_loss(
                    logits=outputs[0],
                    targets=batch[:, 1:],
                    mu=outputs[1],
                    log_var=outputs[2],
                    reg_pred=outputs[3],
                    ld50_pred=outputs[4],
                    labels=labels,
                    beta=beta,
                    gamma=GAMMA,
                    kl_free_bits=KL_FREE_BITS,
                )

                loss_values[0].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

                for key, value in zip(
                    ["loss", "recon", "kl", "reg", "ld50"],
                    loss_values,
                ):
                    totals[key] += value.item()
                totals["grad"] += grad_norm.item()

            number_of_batches = len(train_loader)
            train_metrics = {
                key: value / number_of_batches
                for key, value in totals.items()
            }

            scheduler.step()

            (
                valid_total,
                valid_recon,
                valid_kl,
                valid_reg,
                valid_ld50,
            ) = evaluate_loss(
                model,
                valid_loader,
                beta,
                GAMMA,
                device,
                KL_FREE_BITS,
            )

            valid_reg_metrics = evaluate_regression(
                model, valid_loader, device, reg_mean, reg_std, head="reg_predictor"
            )
            valid_rmse = valid_reg_metrics["rmse"]
            valid_mae = valid_reg_metrics["mae"]

            valid_ld50_metrics = evaluate_regression(
                model, valid_loader, device, ld50_mean, ld50_std, head="ld50_predictor"
            )
            valid_ld50_rmse = valid_ld50_metrics["rmse"]
            valid_ld50_mae = valid_ld50_metrics["mae"]

            train_exact_acc, train_token_acc, train_validity = evaluate(
                model,
                train_loader.dataset,
                vocab_size,
                device,
                idx2char,
            )
            valid_exact_acc, valid_token_acc, valid_validity = evaluate(
                model,
                valid_loader.dataset,
                vocab_size,
                device,
                idx2char,
            )
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:>2}/{EPOCHS}, beta: {beta:.4f}, "
                f"Loss: {train_metrics['loss']:.4f}, "
                f"Recon: {train_metrics['recon']:.4f}, "
                f"KL: {train_metrics['kl']:.4f}, "
                f"Reg: {train_metrics['reg']:.4f}, "
                f"LD50: {train_metrics['ld50']:.4f}, "
                f"Recon acc: {train_exact_acc:.3f}, "
                f"Token acc: {train_token_acc:.3f}, "
                f"Validity: {train_validity:.3f}, "
                f"LR: {current_lr:.1e}"
            )
            print(
                f"         Valid loss: {valid_total:.4f}, "
                f"Sol RMSE: {valid_rmse:.4f}, MAE: {valid_mae:.4f}, "
                f"LD50 RMSE: {valid_ld50_rmse:.4f}, MAE: {valid_ld50_mae:.4f}, "
                f"Recon acc: {valid_exact_acc:.3f}, "
                f"Token acc: {valid_token_acc:.3f}, "
                f"Validity: {valid_validity:.3f}"
            )

            metrics = {
                "train_loss": train_metrics["loss"],
                "train_recon": train_metrics["recon"],
                "train_kl": train_metrics["kl"],
                "train_reg_loss": train_metrics["reg"],
                "train_ld50_loss": train_metrics["ld50"],
                "train_recon_acc": train_exact_acc,
                "train_token_acc": train_token_acc,
                "train_validity": train_validity,
                "valid_loss": valid_total,
                "valid_recon": valid_recon,
                "valid_kl": valid_kl,
                "valid_reg_loss": valid_reg,
                "valid_ld50_loss": valid_ld50,
                "valid_rmse": valid_rmse,
                "valid_mae": valid_mae,
                "valid_ld50_rmse": valid_ld50_rmse,
                "valid_ld50_mae": valid_ld50_mae,
                "valid_recon_acc": valid_exact_acc,
                "valid_token_acc": valid_token_acc,
                "valid_validity": valid_validity,
                "grad_norm": train_metrics["grad"],
                "beta": beta,
                "learning_rate": current_lr,
            }

            mlflow.log_metrics(metrics, step=epoch + 1)
            for name, value in metrics.items():
                writer.add_scalar(name, value, epoch + 1)

            if epoch >= ANNEAL_EPOCHS:
                improved = valid_total < best_val_loss - MIN_DELTA
                if improved:
                    best_val_loss = valid_total
                    best_epoch = epoch
                    epochs_no_improve = 0
                    torch.save(build_checkpoint(), best_model_path)
                    mlflow.log_metric(
                        "best_valid_loss", best_val_loss, step=epoch + 1
                    )
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        print(
                            f"Early stopping at epoch {epoch + 1}, "
                            f"best epoch {best_epoch + 1} "
                            f"(valid_loss {best_val_loss:.4f})"
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
            f"Sol RMSE: {test_metrics['rmse']:.3f}, MAE: {test_metrics['mae']:.3f}, "
            f"LD50 RMSE: {test_metrics['ld50_rmse']:.3f}, MAE: {test_metrics['ld50_mae']:.3f}"
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

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