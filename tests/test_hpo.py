import optuna
import torch
from model import VAE
from data import MAX_LENGTH

VOCAB_SIZE = 41
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def test_optuna_study_creation():
    study = optuna.create_study(direction="minimize")
    assert study.direction.name == "MINIMIZE"


def test_trial_suggest_beta():
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        beta_max = trial.suggest_float("beta_max", 0.001, 2.0, log=True)
        gamma = trial.suggest_float("gamma", 0.005, 0.5, log=True)
        assert 0.001 <= beta_max <= 2.0
        assert 0.005 <= gamma <= 0.5
        return 0.0

    study.optimize(objective, n_trials=3)
    assert len(study.trials) == 3


def test_single_trial_forward_pass():
    model = VAE(VOCAB_SIZE, MAX_LENGTH, 512, 128, 2).to(DEVICE)
    batch = torch.randint(0, VOCAB_SIZE, (4, MAX_LENGTH)).to(DEVICE)
    logits, mu, log_var, prop_logit = model(batch)
    assert logits.shape == (4, MAX_LENGTH - 1, VOCAB_SIZE)
