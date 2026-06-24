import torch
from data import MAX_LENGTH
from model import VAE
from train import compute_pos_weights, vae_loss


def test_compute_pos_weights():
    labels = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    weights = compute_pos_weights(labels, torch.device("cpu"))
    assert torch.allclose(weights, torch.tensor([2.0, 1.0]))


def test_vae_loss_uses_one_reconstruction_and_equal_property_weights():
    batch_size = 3
    seq_len = 8
    vocab_size = 7
    latent_dim = 4
    gamma = 0.25
    beta = 0.1

    logits = torch.randn(batch_size, seq_len - 1, vocab_size, requires_grad=True)
    targets = torch.randint(1, vocab_size, (batch_size, seq_len - 1))
    mu = torch.randn(batch_size, latent_dim, requires_grad=True)
    log_var = torch.randn(batch_size, latent_dim, requires_grad=True)
    d6_logit = torch.randn(batch_size, requires_grad=True)
    c19_logit = torch.randn(batch_size, requires_grad=True)
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    pos_weights = torch.tensor([1.0, 1.0])

    loss, recon, kl, d6_loss, c19_loss = vae_loss(
        logits=logits,
        targets=targets,
        mu=mu,
        log_var=log_var,
        cyp2d6_logit=d6_logit,
        cyp2c19_logit=c19_logit,
        labels=labels,
        beta=beta,
        gamma=gamma,
        kl_free_bits=0.0,
        pos_weights=pos_weights,
    )

    expected = recon + beta * kl + gamma * d6_loss + gamma * c19_loss
    assert torch.allclose(loss, expected)


def test_both_predictors_receive_gradients():
    vocab_size = 10
    model = VAE(
        vocab_size=vocab_size,
        seq_len=MAX_LENGTH,
        hidden_dim=64,
        latent_dim=16,
        n_layers=1,
        dropout=0.0,
        prop_hidden_size=16,
    )
    batch = torch.randint(1, vocab_size, (4, MAX_LENGTH))
    labels = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    )
    outputs = model(batch)
    loss, *_ = vae_loss(
        logits=outputs[0],
        targets=batch[:, 1:],
        mu=outputs[1],
        log_var=outputs[2],
        cyp2d6_logit=outputs[3],
        cyp2c19_logit=outputs[4],
        labels=labels,
        beta=0.01,
        gamma=0.1,
        kl_free_bits=0.0,
        pos_weights=torch.tensor([1.0, 1.0]),
    )
    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.cyp2d6_predictor.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in model.cyp2c19_predictor.parameters()
    )
