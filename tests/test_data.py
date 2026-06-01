import torch
import pytest
from data import build_dataloaders, MAX_LENGTH, encode_smiles, keep_largest_fragment, SMILESDataset


def test_keep_largest_fragment_no_dot():
    smi = "CCO"
    assert keep_largest_fragment(smi) == "CCO"


def test_keep_largest_fragment_with_dot():
    smi = "CCO.C"
    result = keep_largest_fragment(smi)
    assert result == "CCO"


def test_keep_largest_fragment_invalid():
    result = keep_largest_fragment("INVALID.XYZ")
    assert result is None or isinstance(result, str)


def test_encode_smiles_length():
    char2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "C": 3, "O": 4}
    encoded = encode_smiles("CCO", char2idx, MAX_LENGTH)
    assert len(encoded) == MAX_LENGTH


def test_encode_smiles_starts_with_sos():
    char2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "C": 3, "O": 4}
    encoded = encode_smiles("CCO", char2idx, MAX_LENGTH)
    assert encoded[0] == 1  # <sos> token


def test_smiles_dataset_len():
    data = torch.randint(0, 41, (10, MAX_LENGTH))
    labels = torch.zeros(10)
    dataset = SMILESDataset(data, labels)
    assert len(dataset) == 10


def test_smiles_dataset_getitem():
    data = torch.randint(0, 41, (10, MAX_LENGTH))
    labels = torch.ones(10)
    dataset = SMILESDataset(data, labels)
    x, y = dataset[0]
    assert x.shape == (MAX_LENGTH,)
    assert y.item() == 1.0


def test_build_dataloaders_returns_correct_types():
    train_loader, valid_loader, test_loader, vocab_size, char2idx, idx2char, train_labels = build_dataloaders()
    assert vocab_size == 41
    assert isinstance(char2idx, dict)
    assert isinstance(idx2char, dict)
    assert isinstance(train_labels, torch.Tensor)


def test_build_dataloaders_batch_shape():
    train_loader, _, _, vocab_size, _, _, _ = build_dataloaders()
    batch, labels = next(iter(train_loader))
    assert batch.shape[1] == MAX_LENGTH
    assert labels.shape[0] == batch.shape[0]


def test_vocab_size():
    _, _, _, vocab_size, _, _, _ = build_dataloaders()
    assert vocab_size == 41
