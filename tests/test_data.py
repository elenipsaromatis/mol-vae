import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import data as data_module
from data import (
    MAX_LENGTH,
    SMILESDataset,
    build_dataloaders,
    canonicalize_smiles,
    encode_smiles,
    keep_largest_fragment,
)


class FakeADME:
    def __init__(self, name):
        self.name = name

    def get_split(self, method):
        assert self.name == "CYP2D6_Veith"
        assert method == "scaffold"
        return {
            "train": pd.DataFrame(
                {
                    "Drug": ["CCO", "CCC", "CCN", "CO"],
                    "Y": [1, 0, 1, 0],
                }
            ),
            "valid": pd.DataFrame(
                {
                    "Drug": ["CCCl", "COC"],
                    "Y": [0, 1],
                }
            ),
            "test": pd.DataFrame(
                {
                    "Drug": ["CCBr", "CNC"],
                    "Y": [1, 0],
                }
            ),
        }

    def get_data(self, format="df"):
        assert self.name == "CYP2C19_Veith"
        assert format == "df"
        return pd.DataFrame(
            {
                "Drug": [
                    "CCO",
                    "CCC",
                    "CCN",
                    "CCCl",
                    "COC",
                    "CCBr",
                    "CNC",
                    "CCCCCC",
                ],
                "Y": [0, 1, 0, 1, 0, 0, 1, 1],
            }
        )


@pytest.fixture(scope="module")
def dataloader_results():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(data_module, "ADME", FakeADME)
    results = build_dataloaders(batch_size=2)
    yield results
    monkeypatch.undo()


def test_keep_largest_fragment_no_dot():
    assert keep_largest_fragment("CCO") == "CCO"


def test_keep_largest_fragment_with_dot():
    assert keep_largest_fragment("CCO.C") == "CCO"


def test_keep_largest_fragment_invalid():
    assert keep_largest_fragment("INVALID.XYZ") is None


def test_canonicalize_smiles():
    assert canonicalize_smiles("OCC") == "CCO"


def test_encode_smiles_length_and_tokens():
    char2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "C": 3, "O": 4}
    encoded = encode_smiles("CCO", char2idx, MAX_LENGTH)
    assert len(encoded) == MAX_LENGTH
    assert encoded[0] == char2idx["<sos>"]
    assert encoded[4] == char2idx["<eos>"]


def test_smiles_dataset_returns_two_labels():
    encoded = torch.randint(0, 5, (3, MAX_LENGTH))
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dataset = SMILESDataset(encoded, labels)
    x, y = dataset[0]
    assert len(dataset) == 3
    assert x.shape == (MAX_LENGTH,)
    assert y.shape == (2,)


def test_build_dataloaders_return_types(dataloader_results):
    train_loader, valid_loader, test_loader, vocab_size, char2idx, idx2char, train_labels = dataloader_results
    assert isinstance(train_loader, DataLoader)
    assert isinstance(valid_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert isinstance(vocab_size, int)
    assert isinstance(char2idx, dict)
    assert isinstance(idx2char, dict)
    assert isinstance(train_labels, torch.Tensor)


def test_overlap_only_and_label_shape(dataloader_results):
    train_loader, valid_loader, test_loader, _, _, _, train_labels = dataloader_results
    assert len(train_loader.dataset) == 3  # "CO" is absent from CYP2C19.
    assert len(valid_loader.dataset) == 2
    assert len(test_loader.dataset) == 2
    assert train_labels.shape == (3, 2)


def test_batch_shapes(dataloader_results):
    train_loader = dataloader_results[0]
    batch, labels = next(iter(train_loader))
    assert batch.ndim == 2
    assert batch.shape[1] == MAX_LENGTH
    assert labels.ndim == 2
    assert labels.shape[1] == 2


def test_shared_vocabulary(dataloader_results):
    _, _, _, vocab_size, char2idx, idx2char, _ = dataloader_results
    assert vocab_size == len(char2idx) == len(idx2char)
    assert char2idx["<pad>"] == 0
    assert char2idx["<sos>"] == 1
    assert char2idx["<eos>"] == 2
    for character, index in char2idx.items():
        assert idx2char[index] == character
