import torch
import pytest
from model import VAE
from train import vae_loss, evaluate_loss
from data import build_dataloaders, MAX_LENGTH

VOCAB_SIZE = 41
BATCH_SIZE = 4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@pytest.fixture
def model():
    return VAE(VOCAB_SIZE, MAX_LENGTH, 512, 128, 2).to(DEVICE)


@pytest.fixture
def dummy_inputs():
    batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LENGTH)).to(DEVICE)
    labels = torch.randint(0, 2, (BATCH_SIZE,)).float().to(DEVICE)
    train_labels = torch.randint(0, 2, (100,)).float()
    return batch, labels, train_labels


def test_vae_loss_returns_four_values(model, dummy_inputs):
    batch, labels, train_labels = dummy_inputs
    logits, mu, log_var, prop_logit = model(batch)
    result = vae_loss(
        logits, batch[:, 1:], mu, log_var, beta=0.008,
        prop_logit=prop_logit, labels=labels,
        gamma=0.05, kl_free_bits=0.5, train_labels=train_labels
    )
    assert len(result) == 4


def test_vae_loss_is_positive(model, dummy_inputs):
    batch, labels, train_labels = dummy_inputs
    logits, mu, log_var, prop_logit = model(batch)
    loss, recon, kl, prop = vae_loss(
        logits, batch[:, 1:], mu, log_var, beta=0.008,
        prop_logit=prop_logit, labels=labels,
        gamma=0.05, kl_free_bits=0.5, train_labels=train_labels
    )
    assert loss.item() > 0
    assert recon.item() > 0
    assert kl.item() > 0
    assert prop.item() > 0


def test_vae_loss_zero_beta_zero_gamma(model, dummy_inputs):
    batch, labels, train_labels = dummy_inputs
    logits, mu, log_var, prop_logit = model(batch)
    loss, recon, kl, prop = vae_loss(
        logits, batch[:, 1:], mu, log_var, beta=0.0,
        prop_logit=prop_logit, labels=labels,
        gamma=0.0, kl_free_bits=0.5, train_labels=train_labels
    )
    assert torch.isclose(loss, recon)


def test_vae_loss_is_scalar(model, dummy_inputs):
    batch, labels, train_labels = dummy_inputs
    logits, mu, log_var, prop_logit = model(batch)
    loss, _, _, _ = vae_loss(
        logits, batch[:, 1:], mu, log_var, beta=0.008,
        prop_logit=prop_logit, labels=labels,
        gamma=0.05, kl_free_bits=0.5, train_labels=train_labels
    )
    assert loss.shape == torch.Size([])


def test_evaluate_loss_returns_three_floats(model):
    _, valid_loader, _, _, _, _, train_labels = build_dataloaders()
    val_recon, val_kl, val_prop = evaluate_loss(
        model, valid_loader, beta=0.008, gamma=0.05,
        device=DEVICE, kl_free_bits=0.5, train_labels=train_labels
    )
    assert isinstance(val_recon, float)
    assert isinstance(val_kl, float)
    assert isinstance(val_prop, float)
