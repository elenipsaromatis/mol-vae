import torch
import pytest
from model import VAE
from evaluate import evaluate_auroc
from data import build_dataloaders, MAX_LENGTH

VOCAB_SIZE = 41
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@pytest.fixture
def model():
    return VAE(VOCAB_SIZE, MAX_LENGTH, 512, 128, 2).to(DEVICE)


def test_evaluate_auroc_returns_float(model):
    _, valid_loader, _, _, _, _, _ = build_dataloaders()
    auroc = evaluate_auroc(model, valid_loader, DEVICE)
    assert isinstance(auroc, float)


def test_evaluate_auroc_in_valid_range(model):
    _, valid_loader, _, _, _, _, _ = build_dataloaders()
    auroc = evaluate_auroc(model, valid_loader, DEVICE)
    assert 0.0 <= auroc <= 1.0


def test_evaluate_auroc_untrained_near_random(model):
    _, valid_loader, _, _, _, _, _ = build_dataloaders()
    auroc = evaluate_auroc(model, valid_loader, DEVICE)
    # untrained model should be near 0.5, allow some slack
    assert 0.3 <= auroc <= 0.7
