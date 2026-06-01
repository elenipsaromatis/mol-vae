import torch
import pytest
from model import VAE
from data import MAX_LENGTH

VOCAB_SIZE = 41
BATCH_SIZE = 4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@pytest.fixture
def model():
    return VAE(VOCAB_SIZE, MAX_LENGTH, 512, 128, 2).to(DEVICE)


@pytest.fixture
def dummy_batch():
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LENGTH)).to(DEVICE)


def test_forward_output_shapes(model, dummy_batch):
    logits, mu, log_var, prop_logit = model(dummy_batch)
    assert logits.shape == (BATCH_SIZE, MAX_LENGTH - 1, VOCAB_SIZE)
    assert mu.shape == (BATCH_SIZE, 128)
    assert log_var.shape == (BATCH_SIZE, 128)
    assert prop_logit.shape == (BATCH_SIZE,)


def test_encoder_output_shapes(model, dummy_batch):
    mu, log_var = model.encoder(dummy_batch)
    assert mu.shape == (BATCH_SIZE, 128)
    assert log_var.shape == (BATCH_SIZE, 128)


def test_decoder_output_shape(model, dummy_batch):
    mu, log_var = model.encoder(dummy_batch)
    logits = model.decoder(mu, dummy_batch)
    assert logits.shape == (BATCH_SIZE, MAX_LENGTH - 1, VOCAB_SIZE)


def test_property_predictor_output_shape(model, dummy_batch):
    mu, _ = model.encoder(dummy_batch)
    prop_logit = model.predictor(mu)
    assert prop_logit.shape == (BATCH_SIZE,)


def test_model_on_device(model):
    for param in model.parameters():
        assert param.device.type == DEVICE.type


def test_model_parameter_count(model):
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
