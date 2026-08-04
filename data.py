import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from tdc.single_pred import ADME
from tdc.single_pred import Tox

MAX_LENGTH = 100

def keep_largest_fragment(smi):
    """Return the valid fragment with the most heavy atoms."""
    if not isinstance(smi, str) or not smi:
        return None

    fragments = smi.split(".")
    valid_fragments = []

    for fragment in fragments:
        mol = Chem.MolFromSmiles(fragment)
        if mol is not None:
            valid_fragments.append((mol.GetNumHeavyAtoms(), fragment))

    if not valid_fragments:
        return None

    return max(valid_fragments, key=lambda item: item[0])[1]


def canonicalize_smiles(smi):
    """Convert a valid SMILES string to RDKit canonical SMILES."""
    if not isinstance(smi, str) or not smi:
        return None

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def encode_smiles(smiles, char2idx, max_length):
    tokens = ["<sos>"] + list(smiles) + ["<eos>"]
    tokens = tokens[:max_length]
    tokens += ["<pad>"] * (max_length - len(tokens))
    return [char2idx[token] for token in tokens]


class SMILESDataset(Dataset):
    def __init__(self, encoded_data, labels, smiles=None):
        self.data = encoded_data
        self.labels = labels
        # Raw canonical SMILES aligned to encoded_data by index.
        # None for callers that do not need them (e.g. single-task loader).
        self.smiles = smiles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _clean_property_dataframe(df, label_name):
    required_columns = {"Drug", "Y"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    cleaned = df[["Drug", "Y"]].copy()
    cleaned["Drug"] = cleaned["Drug"].apply(keep_largest_fragment)
    cleaned["Drug"] = cleaned["Drug"].apply(canonicalize_smiles)
    cleaned["Y"] = pd.to_numeric(cleaned["Y"], errors="coerce")
    # Zhu LD50 values are in log10(mol/kg) units, but we want to work with mol/kg
    if label_name == "Y_ld50":
        cleaned["Y"] = 1 / (10 ** cleaned["Y"])

    cleaned.dropna(subset=["Drug", "Y"], inplace=True)
    cleaned = cleaned[cleaned["Drug"].str.len() <= MAX_LENGTH - 2]

    cleaned.rename(columns={"Y": label_name}, inplace=True)
    cleaned.drop_duplicates(subset=["Drug"], keep="first", inplace=True)
    cleaned.reset_index(drop=True, inplace=True)

    return cleaned


def _get_tdc_dataframe(dataset):
    try:
        return dataset.get_data(format="df")
    except TypeError:
        return dataset.get_data()


def build_dataloaders(batch_size=64):
    """Single-task solubility (AqSolDB) regression loaders"""
    sol_data = ADME(name="Solubility_AqSolDB")
    sol_split = sol_data.get_split(method="scaffold")

    train_df = _clean_property_dataframe(sol_split["train"], "Y_sol")
    valid_df = _clean_property_dataframe(sol_split["valid"], "Y_sol")
    test_df = _clean_property_dataframe(sol_split["test"], "Y_sol")

    train_smiles = train_df["Drug"].tolist()
    valid_smiles = valid_df["Drug"].tolist()
    test_smiles = test_df["Drug"].tolist()

    all_smiles = train_smiles + valid_smiles + test_smiles
    vocab = ["<pad>", "<sos>", "<eos>"] + sorted(set("".join(all_smiles)))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(vocab)

    encoded_train = torch.tensor(
        [encode_smiles(s, char2idx, MAX_LENGTH) for s in train_smiles],
        dtype=torch.long,
    )
    encoded_valid = torch.tensor(
        [encode_smiles(s, char2idx, MAX_LENGTH) for s in valid_smiles],
        dtype=torch.long,
    )
    encoded_test = torch.tensor(
        [encode_smiles(s, char2idx, MAX_LENGTH) for s in test_smiles],
        dtype=torch.long,
    )

    train_labels = torch.tensor(
        train_df[["Y_sol"]].to_numpy(dtype=np.float32), dtype=torch.float
    )
    valid_labels = torch.tensor(
        valid_df[["Y_sol"]].to_numpy(dtype=np.float32), dtype=torch.float
    )
    test_labels = torch.tensor(
        test_df[["Y_sol"]].to_numpy(dtype=np.float32), dtype=torch.float
    )

    reg_mean = train_labels[:, 0].mean()
    reg_std = train_labels[:, 0].std()
    for t in (train_labels, valid_labels, test_labels):
        t[:, 0] = (t[:, 0] - reg_mean) / reg_std

    train_dataset = SMILESDataset(encoded_train, train_labels, train_smiles)
    valid_dataset = SMILESDataset(encoded_valid, valid_labels, valid_smiles)
    test_dataset = SMILESDataset(encoded_test, test_labels, test_smiles)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    print(
        "[split] sizes train/valid/test: "
        f"{len(train_df)}/{len(valid_df)}/{len(test_df)}"
    )
    print(f"[split] vocab_size: {vocab_size}")
    print(f"[target] mean/std: {reg_mean:.3f}/{reg_std:.3f}")

    return (
        train_loader,
        valid_loader,
        test_loader,
        vocab_size,
        char2idx,
        idx2char,
        train_labels,
        reg_mean,
        reg_std,
    )

def build_dataloaders_multitask(batch_size=64):
    """Solubility (AqSolDB, scaffold split) + LD50_Zhu overlap"""
    sol_data = ADME(name="Solubility_AqSolDB")
    sol_split = sol_data.get_split(method="scaffold")

    ld50_df = _get_tdc_dataframe(Tox(name="LD50_Zhu"))
    ld50_clean = _clean_property_dataframe(ld50_df, "Y_ld50")

    def build_partition(df):
        cleaned = _clean_property_dataframe(df, "Y_sol")
        merged = cleaned.merge(
            ld50_clean[["Drug", "Y_ld50"]], on="Drug", how="inner"
        )
        return merged

    train_df = build_partition(sol_split["train"])
    valid_df = build_partition(sol_split["valid"])
    test_df = build_partition(sol_split["test"])

    train_smiles = train_df["Drug"].tolist()
    valid_smiles = valid_df["Drug"].tolist()
    test_smiles = test_df["Drug"].tolist()

    all_smiles = train_smiles + valid_smiles + test_smiles
    vocab = ["<pad>", "<sos>", "<eos>"] + sorted(set("".join(all_smiles)))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(vocab)

    encoded_train = torch.tensor(
        [encode_smiles(s, char2idx, MAX_LENGTH) for s in train_smiles],
        dtype=torch.long,
    )
    encoded_valid = torch.tensor(
        [encode_smiles(s, char2idx, MAX_LENGTH) for s in valid_smiles],
        dtype=torch.long,
    )
    encoded_test = torch.tensor(
        [encode_smiles(s, char2idx, MAX_LENGTH) for s in test_smiles],
        dtype=torch.long,
    )

    train_labels = torch.tensor(
        train_df[["Y_sol", "Y_ld50"]].to_numpy(dtype=np.float32), dtype=torch.float
    )
    valid_labels = torch.tensor(
        valid_df[["Y_sol", "Y_ld50"]].to_numpy(dtype=np.float32), dtype=torch.float
    )
    test_labels = torch.tensor(
        test_df[["Y_sol", "Y_ld50"]].to_numpy(dtype=np.float32), dtype=torch.float
    )

    # standardize both targets independently
    sol_mean = train_labels[:, 0].mean()
    sol_std = train_labels[:, 0].std()
    ld50_mean = train_labels[:, 1].mean()
    ld50_std = train_labels[:, 1].std()

    for t in (train_labels, valid_labels, test_labels):
        t[:, 0] = (t[:, 0] - sol_mean) / sol_std
        t[:, 1] = (t[:, 1] - ld50_mean) / ld50_std

    train_dataset = SMILESDataset(encoded_train, train_labels, train_smiles)
    valid_dataset = SMILESDataset(encoded_valid, valid_labels, valid_smiles)
    test_dataset = SMILESDataset(encoded_test, test_labels, test_smiles)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        "[split] sizes train/valid/test: "
        f"{len(train_df)}/{len(valid_df)}/{len(test_df)}"
    )
    print(f"[split] vocab_size: {vocab_size}")
    print(f"[target] solubility mean/std: {sol_mean:.3f}/{sol_std:.3f}")
    print(f"[target] LD50 mean/std: {ld50_mean:.3f}/{ld50_std:.3f}")

    return (
        train_loader,
        valid_loader,
        test_loader,
        vocab_size,
        char2idx,
        idx2char,
        train_labels,
        sol_mean,
        sol_std,
        ld50_mean,
        ld50_std,
    )


def build_ld50_full_dataframe():
    """Full LD50_Zhu, cleaned the same way as the overlap pipeline but not
    merged against Solubility_AqSolDB. Used to test how well a VAE trained
    on the solubility x LD50 overlap extrapolates to the rest of LD50_Zhu.
    """
    ld50_df = _get_tdc_dataframe(Tox(name="LD50_Zhu"))
    return _clean_property_dataframe(ld50_df, "Y_ld50")