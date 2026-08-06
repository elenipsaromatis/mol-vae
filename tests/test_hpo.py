"""Tests for hpo.py against the current train_vae.py objective.

train_vae.py trains reconstruction + KL + the solubility (reg_predictor)
head only; the ld50_predictor head is untrained (LD50 is reserved for the
downstream GP/BO stage). hpo.py's objective must mirror that: no ld50_pred
unpacked from the model, no ld50_loss folded into val_score.
"""
from contextlib import nullcontext
import sys
import types

import optuna
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

if "mlflow" not in sys.modules:
    try:
        import mlflow
    except ImportError:
        mlflow = types.ModuleType("mlflow")
        mlflow.start_run = lambda *args, **kwargs: nullcontext()
        mlflow.log_params = lambda *args, **kwargs: None
        mlflow.log_metrics = lambda *args, **kwargs: None
        mlflow.set_tracking_uri = lambda *args, **kwargs: None
        mlflow.set_experiment = lambda *args, **kwargs: None
        sys.modules["mlflow"] = mlflow

import hpo
from data import MAX_LENGTH


VOCAB_SIZE = 8


def make_overlap_loader(number_of_samples=4, batch_size=2):
    data = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(number_of_samples, MAX_LENGTH),
        dtype=torch.long,
    )

    labels = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float,
    )[:number_of_samples]

    dataset = TensorDataset(data, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def configure_mlflow_mocks(monkeypatch):
    monkeypatch.setattr(
        hpo.mlflow,
        "start_run",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        hpo.mlflow,
        "log_params",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hpo.mlflow,
        "log_metrics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hpo.mlflow,
        "set_tracking_uri",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hpo.mlflow,
        "set_experiment",
        lambda *args, **kwargs: None,
    )


def make_fixed_trial():
    return optuna.trial.FixedTrial(
        {
            "beta_max": 0.01,
            "gamma": 0.02,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "dropout": 0.2,
            "kl_free_bits": 0.5,
            "latent_dim": 64,
            "n_layers": 1,
            "prop_hidden_size": 32,
        }
    )


def make_objective_kwargs(train_loader, valid_loader, n_epochs=1):
    return dict(
        train_loader=train_loader,
        valid_loader=valid_loader,
        vocab_size=VOCAB_SIZE,
        reg_mean=torch.tensor(0.0),
        reg_std=torch.tensor(1.0),
        ld50_mean=torch.tensor(0.0),
        ld50_std=torch.tensor(1.0),
        device=torch.device("cpu"),
        n_epochs=n_epochs,
    )


def test_hpo_imports_loss_from_train_vae_not_train():
    # train.py is retired; the objective must come from train_vae.py, whose
    # vae_loss/evaluate_loss never train the ld50 head.
    import train_vae

    assert hpo.vae_loss is train_vae.vae_loss
    assert hpo.evaluate_loss is train_vae.evaluate_loss


def test_make_objective_returns_callable(monkeypatch):
    configure_mlflow_mocks(monkeypatch)
    monkeypatch.setattr(hpo, "HIDDEN_DIM", 16)

    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()

    objective = hpo.make_objective(
        **make_objective_kwargs(train_loader, valid_loader)
    )

    assert callable(objective)


def test_objective_runs_end_to_end_and_returns_finite_score(monkeypatch):
    configure_mlflow_mocks(monkeypatch)
    monkeypatch.setattr(hpo, "HIDDEN_DIM", 16)

    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()

    objective = hpo.make_objective(
        **make_objective_kwargs(train_loader, valid_loader)
    )

    score = objective(make_fixed_trial())

    assert isinstance(score, float)
    assert score == score  # not NaN
    assert score != float("inf")


def test_vae_loss_called_without_ld50_pred(monkeypatch):
    configure_mlflow_mocks(monkeypatch)
    monkeypatch.setattr(hpo, "HIDDEN_DIM", 16)

    captured_kwargs = []
    original_vae_loss = hpo.vae_loss

    def capture_vae_loss(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return original_vae_loss(*args, **kwargs)

    monkeypatch.setattr(hpo, "vae_loss", capture_vae_loss)

    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()

    objective = hpo.make_objective(
        **make_objective_kwargs(train_loader, valid_loader)
    )

    objective(make_fixed_trial())

    assert captured_kwargs
    for kwargs in captured_kwargs:
        assert "ld50_pred" not in kwargs
        assert "ld50_loss" not in kwargs


def test_val_score_excludes_ld50_loss(monkeypatch):
    configure_mlflow_mocks(monkeypatch)
    monkeypatch.setattr(hpo, "HIDDEN_DIM", 16)

    # evaluate_loss now returns a 4-tuple (total, recon, kl, reg) — no
    # ld50 term. val_score must be exactly recon + reg.
    monkeypatch.setattr(
        hpo,
        "evaluate_loss",
        lambda **kwargs: (3.0, 2.0, 1.0, 0.4),
    )

    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()

    objective = hpo.make_objective(
        **make_objective_kwargs(train_loader, valid_loader)
    )

    score = objective(make_fixed_trial())

    assert score == pytest.approx(2.4)


def test_run_hpo_builds_and_returns_study(monkeypatch, tmp_path):
    configure_mlflow_mocks(monkeypatch)
    monkeypatch.setattr(hpo, "ROOT", tmp_path)

    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()
    test_loader = make_overlap_loader()

    monkeypatch.setattr(
        hpo,
        "build_dataloaders_multitask",
        lambda batch_size: (
            train_loader,
            valid_loader,
            test_loader,
            VOCAB_SIZE,
            {"<pad>": 0},
            {0: "<pad>"},
            train_loader.dataset.tensors[1],
            torch.tensor(0.0),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        ),
    )

    monkeypatch.setattr(
        hpo.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    monkeypatch.setattr(
        hpo,
        "JournalStorage",
        lambda backend: None,
    )
    monkeypatch.setattr(
        hpo,
        "JournalFileBackend",
        lambda path: None,
    )

    class FakeTrial:
        number = 0

    class FakeStudy:
        def __init__(self):
            self.best_params = {"gamma": 0.02}
            self.best_value = 1.25
            self.best_trial = FakeTrial()
            self.objective = None
            self.n_trials = None
            self.n_jobs = None

        def optimize(self, objective, n_trials, n_jobs=None):
            self.objective = objective
            self.n_trials = n_trials
            self.n_jobs = n_jobs

    fake_study = FakeStudy()

    monkeypatch.setattr(
        hpo.optuna,
        "create_study",
        lambda **kwargs: fake_study,
    )

    study = hpo.run_hpo(
        n_trials=3,
        n_epochs=2,
    )

    assert study is fake_study
    assert fake_study.n_trials == 3
    assert callable(fake_study.objective)

    best_path = tmp_path / "best_params.json"
    assert best_path.exists()
