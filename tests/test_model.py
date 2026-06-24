import pytest
import torch
from data import MAX_LENGTH
from model import VAE, reparameterise
VOCAB_SIZE = 41
BATCH_SIZE = 4
HIDDEN_DIM = 128
LATENT_DIM = 64
N_LAYERS = 2
PROP_HIDDEN_SIZE = 32
DEVICE = torch.device("cpu")


@pytest.fixture
def model():
    return VAE(
        vocab_size=VOCAB_SIZE,
        seq_len=MAX_LENGTH,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        n_layers=N_LAYERS,
        dropout=0.5,
        prop_hidden_size=PROP_HIDDEN_SIZE,
    ).to(DEVICE)


@pytest.fixture
def dummy_batch():
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LENGTH), device=DEVICE)


def test_forward_output_shapes(model, dummy_batch):
    logits, mu, log_var, cyp2d6_logit, cyp2c19_logit = model(dummy_batch)
    assert logits.shape == (BATCH_SIZE, MAX_LENGTH - 1, VOCAB_SIZE)
    assert mu.shape == (BATCH_SIZE, LATENT_DIM)
    assert log_var.shape == (BATCH_SIZE, LATENT_DIM)
    assert cyp2d6_logit.shape == (BATCH_SIZE,)
    assert cyp2c19_logit.shape == (BATCH_SIZE,)


def test_encoder_output_shapes(model, dummy_batch):
    mu, log_var = model.encoder(dummy_batch)
    assert mu.shape == (BATCH_SIZE, LATENT_DIM)
    assert log_var.shape == (BATCH_SIZE, LATENT_DIM)


def test_decoder_output_shape(model, dummy_batch):
    mu, log_var = model.encoder(dummy_batch)
    z = reparameterise(mu, log_var)
    logits = model.decoder(z, dummy_batch)
    assert logits.shape == (BATCH_SIZE, MAX_LENGTH - 1, VOCAB_SIZE)


def test_property_predictor_output_shapes(model, dummy_batch):
    mu, _ = model.encoder(dummy_batch)
    assert model.cyp2d6_predictor(mu).shape == (BATCH_SIZE,)
    assert model.cyp2c19_predictor(mu).shape == (BATCH_SIZE,)


def test_predictors_are_separate(model):
    assert model.cyp2d6_predictor is not model.cyp2c19_predictor
    for d6_parameter, c19_parameter in zip(
        model.cyp2d6_predictor.parameters(),
        model.cyp2c19_predictor.parameters(),
    ):
        assert d6_parameter is not c19_parameter


def test_model_parameter_count(model):
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
