import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tdc.single_pred import ADME
from rdkit import Chem

MAX_LENGTH = 100


def keep_largest_fragment(smi):
    if '.' not in smi:
        return smi
    frags = smi.split('.')
    mols = [(Chem.MolFromSmiles(f), f) for f in frags]
    mols = [(m, f) for m, f in mols if m is not None]
    if not mols:
        return None
    return max(mols, key=lambda x: x[0].GetNumHeavyAtoms())[1]


def encode_smiles(smiles, char2idx, max_length):
    tokens = ['<sos>'] + list(smiles) + ['<eos>']
    tokens = tokens[:max_length]
    tokens += ['<pad>'] * (max_length - len(tokens))
    return [char2idx[t] for t in tokens]


class SMILESDataset(Dataset):
    def __init__(self, encoded_data, labels):
        self.data = encoded_data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def build_dataloaders(batch_size=64):
    data = ADME(name='CYP2D6_Veith')
    split = data.get_split()
    train_df = split['train']
    valid_df = split['valid']
    test_df = split['test']

    for df in [train_df, valid_df, test_df]:
        df['Drug'] = df['Drug'].apply(keep_largest_fragment)
        df.dropna(subset=['Drug'], inplace=True)
        df.drop(df[df['Drug'].str.len() > 98].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    train_smiles = train_df['Drug'].tolist()
    valid_smiles = valid_df['Drug'].tolist()
    test_smiles = test_df['Drug'].tolist()

    import pandas as pd
    df_all = pd.concat([train_df, valid_df, test_df])
    all_smiles = df_all['Drug'].tolist()
    vocab = sorted(set(''.join(all_smiles)))
    vocab = ['<pad>', '<sos>', '<eos>'] + vocab
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    vocab_size = len(vocab)

    encoded_train = torch.tensor([encode_smiles(s, char2idx, MAX_LENGTH) for s in train_smiles], dtype=torch.long)
    encoded_valid = torch.tensor([encode_smiles(s, char2idx, MAX_LENGTH) for s in valid_smiles], dtype=torch.long)
    encoded_test = torch.tensor([encode_smiles(s, char2idx, MAX_LENGTH) for s in test_smiles], dtype=torch.long)

    train_labels = torch.tensor(train_df['Y'].tolist(), dtype=torch.float)
    valid_labels = torch.tensor(valid_df['Y'].tolist(), dtype=torch.float)
    test_labels = torch.tensor(test_df['Y'].tolist(), dtype=torch.float)

    train_dataset = SMILESDataset(encoded_train, train_labels)
    valid_dataset = SMILESDataset(encoded_valid, valid_labels)
    test_dataset = SMILESDataset(encoded_test, test_labels)

    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, vocab_size, char2idx, idx2char, train_labels