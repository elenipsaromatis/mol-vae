import random
from pathlib import Path
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import MAX_LENGTH, build_dataloaders
from evaluate import evaluate, evaluate_regression, evaluate_test
from model import VAE


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.backends.mps.is_available():
    torch.backends.mps.deterministic = True

torch.use_deterministic_algorithms(True, warn_only=True)


HIDDEN_DIM = 512
LATENT_DIM = 256
N_LAYERS = 3
PROP_HIDDEN_SIZE = 32
WEIGHT_DECAY = 5.414669339199985e-05
EPOCHS = 50
BETA_MAX = 0.48748000781458084
ANNEAL_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.0009178680324771136
KL_FREE_BITS = 0.7917042596673599
GAMMA = 0.030010000333227073
DROPOUT = 0.37030217137983146

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"


def vae_loss(
    logits,
    targets,
    mu,
    log_var,
    reg_pred,
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

    loss = recon_loss + beta * kl_loss + gamma * reg_loss

    return loss, recon_loss, kl_loss, reg_loss


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

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
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
    ) = build_dataloaders(batch_size=BATCH_SIZE)

    MLFLOW_DIR = ROOT / "notebooks" / "mlruns"
    mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())

    # Reuse the existing experiment. Do not silently create another one.
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
        run_name = f"vae_solubility_{run_id[:8]}"
        mlflow.set_tag("mlflow.runName", run_name)

        model_path = CHECKPOINT_DIR / f"{run_name}.pth"
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
                "dataset": "Solubility_AqSolDB",
                "split": "TDC Solubility scaffold",
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

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5
        )

        for epoch in range(EPOCHS):
            model.train()
            totals = {
                "loss": 0.0,
                "recon": 0.0,
                "kl": 0.0,
                "reg": 0.0,
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
                    ["loss", "recon", "kl", "reg"],
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
            ) = evaluate_loss(
                model,
                valid_loader,
                beta,
                GAMMA,
                device,
                KL_FREE_BITS,
            )

            valid_reg_metrics = evaluate_regression(
                model, valid_loader, device, reg_mean, reg_std
            )
            valid_rmse = valid_reg_metrics["rmse"]
            valid_mae = valid_reg_metrics["mae"]

            exact_acc, token_acc, validity = evaluate(
                model,
                train_loader.dataset,
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
                f"LR: {current_lr:.1e}"
            )
            print(
                f"         Valid loss: {valid_total:.4f}, "
                f"RMSE: {valid_rmse:.4f}, "
                f"MAE: {valid_mae:.4f}, "
                f"Recon acc: {exact_acc:.3f}, "
                f"Token acc: {token_acc:.3f}, "
                f"Validity: {validity:.3f}"
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
                "valid_rmse": valid_rmse,
                "valid_mae": valid_mae,
                "grad_norm": train_metrics["grad"],
                "beta": beta,
                "learning_rate": current_lr,
                "recon_acc": exact_acc,
                "token_acc": token_acc,
                "validity": validity,
            }

            mlflow.log_metrics(metrics, step=epoch + 1)
            for name, value in metrics.items():
                writer.add_scalar(name, value, epoch + 1)

        test_metrics = evaluate_test(
            model, test_loader, vocab_size, idx2char, device, reg_mean, reg_std
        )

        print(
            "\nTest: "
            f"Recon acc: {test_metrics['recon_acc']:.3f}, "
            f"Validity: {test_metrics['validity']:.3f}, "
            f"RMSE: {test_metrics['rmse']:.3f}, "
            f"MAE: {test_metrics['mae']:.3f}"
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

        torch.save(model.state_dict(), model_path)
        mlflow.set_tag("checkpoint", str(model_path.resolve()))
        writer.close()

        print(f"\nModel saved as {model_path}")
        print(f"Run ID: {run_id}")


if __name__ == "__main__":
    train()