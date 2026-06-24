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


def test_make_objective_returns_callable(monkeypatch):
    configure_mlflow_mocks(monkeypatch)

    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()
    train_labels = train_loader.dataset.tensors[1]

    objective = hpo.make_objective(
        train_loader=train_loader,
        valid_loader=valid_loader,
        vocab_size=VOCAB_SIZE,
        train_labels=train_labels,
        device=torch.device("cpu"),
        n_epochs=1,
    )

    assert callable(objective)


def test_make_objective_requires_two_label_columns():
    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()
    invalid_train_labels = torch.tensor(
        [0.0, 1.0, 0.0, 1.0],
        dtype=torch.float,
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[N, 2\]",
    ):
        hpo.make_objective(
            train_loader=train_loader,
            valid_loader=valid_loader,
            vocab_size=VOCAB_SIZE,
            train_labels=invalid_train_labels,
            device=torch.device("cpu"),
            n_epochs=1,
        )


def test_objective_uses_one_shared_gamma_for_both_properties(monkeypatch):
    configure_mlflow_mocks(monkeypatch)
    monkeypatch.setattr(hpo, "HIDDEN_DIM", 16)

    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()
    train_labels = train_loader.dataset.tensors[1]

    captured_gammas = []
    original_vae_loss = hpo.vae_loss

    def capture_vae_loss(*args, **kwargs):
        captured_gammas.append(kwargs["gamma"])
        return original_vae_loss(*args, **kwargs)

    monkeypatch.setattr(
        hpo,
        "vae_loss",
        capture_vae_loss,
    )

    monkeypatch.setattr(
        hpo,
        "evaluate_loss",
        lambda **kwargs: (
            3.0,
            2.0,
            1.0,
            0.4,
            0.6,
        ),
    )

    monkeypatch.setattr(
        hpo,
        "evaluate_auroc",
        lambda *args, **kwargs: 0.75,
    )

    monkeypatch.setattr(
        hpo,
        "evaluate_auprc",
        lambda *args, **kwargs: 0.70,
    )

    objective = hpo.make_objective(
        train_loader=train_loader,
        valid_loader=valid_loader,
        vocab_size=VOCAB_SIZE,
        train_labels=train_labels,
        device=torch.device("cpu"),
        n_epochs=1,
    )

    trial = optuna.trial.FixedTrial(
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

    score = objective(trial)

    assert score == pytest.approx(2.5)
    assert captured_gammas
    assert all(
        gamma == pytest.approx(0.02)
        for gamma in captured_gammas
    )


def test_run_hpo_builds_and_returns_study(monkeypatch):
    configure_mlflow_mocks(monkeypatch)

    train_loader = make_overlap_loader()
    valid_loader = make_overlap_loader()
    test_loader = make_overlap_loader()
    train_labels = train_loader.dataset.tensors[1]

    monkeypatch.setattr(
        hpo,
        "build_dataloaders",
        lambda batch_size: (
            train_loader,
            valid_loader,
            test_loader,
            VOCAB_SIZE,
            {"<pad>": 0},
            {0: "<pad>"},
            train_labels,
        ),
    )

    monkeypatch.setattr(
        hpo.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    class FakeStudy:
        def __init__(self):
            self.best_params = {"gamma": 0.02}
            self.best_value = 1.25
            self.objective = None
            self.n_trials = None

        def optimize(self, objective, n_trials):
            self.objective = objective
            self.n_trials = n_trials

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
