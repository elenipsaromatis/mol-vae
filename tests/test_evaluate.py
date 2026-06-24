import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import SMILESDataset, encode_smiles
from evaluate import evaluate, evaluate_auprc, evaluate_auroc, evaluate_test


class DummyEncoder(nn.Module):
    def forward(self, x):
        mu = x[:, 1].float().unsqueeze(1)
        log_var = torch.zeros_like(mu)
        return mu, log_var


class PerfectDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, z, x):
        del z
        targets = x[:, 1:]
        logits = torch.zeros(
            targets.size(0), targets.size(1), self.vocab_size, device=x.device
        )
        logits.scatter_(2, targets.unsqueeze(-1), 10.0)
        return logits


class IdentityPredictor(nn.Module):
    def forward(self, mu):
        return mu.squeeze(-1)


class NegativePredictor(nn.Module):
    def forward(self, mu):
        return -mu.squeeze(-1)


class DummyModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = DummyEncoder()
        self.decoder = PerfectDecoder(vocab_size)
        self.cyp2d6_predictor = IdentityPredictor()
        self.cyp2c19_predictor = NegativePredictor()


def make_test_components():
    vocab = ["<pad>", "<sos>", "<eos>", "C", "O", "N", "F"]
    char2idx = {character: index for index, character in enumerate(vocab)}
    idx2char = {index: character for character, index in char2idx.items()}
    smiles = ["C", "O", "N", "F"]
    encoded = torch.tensor(
        [encode_smiles(smi, char2idx, 8) for smi in smiles], dtype=torch.long
    )
    labels = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    )
    dataset = SMILESDataset(encoded, labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = DummyModel(len(vocab))
    return model, dataset, loader, idx2char, len(vocab)


def test_property_metrics_use_correct_label_columns():
    model, _, loader, _, _ = make_test_components()
    device = torch.device("cpu")
    assert evaluate_auroc(model, loader, device, "cyp2d6") == 1.0
    assert evaluate_auroc(model, loader, device, "cyp2c19") == 1.0
    assert evaluate_auprc(model, loader, device, "cyp2d6") == 1.0
    assert evaluate_auprc(model, loader, device, "cyp2c19") == 1.0


def test_reconstruction_evaluation():
    model, dataset, _, idx2char, vocab_size = make_test_components()
    exact_acc, token_acc, validity = evaluate(
        model,
        dataset,
        vocab_size,
        torch.device("cpu"),
        idx2char,
        n_samples=4,
    )
    assert exact_acc == 1.0
    assert token_acc == 1.0
    assert validity == 1.0


def test_evaluate_test_returns_both_properties():
    model, _, loader, idx2char, vocab_size = make_test_components()
    metrics = evaluate_test(
        model,
        loader,
        vocab_size,
        idx2char,
        torch.device("cpu"),
    )
    assert metrics["recon_acc"] == 1.0
    assert metrics["validity"] == 1.0
    assert metrics["cyp2d6_auroc"] == 1.0
    assert metrics["cyp2c19_auroc"] == 1.0
    assert metrics["cyp2d6_auprc"] == 1.0
    assert metrics["cyp2c19_auprc"] == 1.0
