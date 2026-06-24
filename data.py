import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from tdc.single_pred import ADME

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
    def __init__(self, encoded_data, labels):
        self.data = encoded_data
        self.labels = labels

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
    """
    Build overlap-only CYP2D6/CYP2C19 loaders.

    Each sample has the form:
        encoded SMILES -> [CYP2D6 label, CYP2C19 label]

    TDC's CYP2D6 scaffold split is used as the master split. CYP2C19
    labels are inner-joined into each partition, so every retained
    molecule has both labels and stays in its CYP2D6 scaffold partition.
    """
    if ADME is None:
        raise ImportError(
            "PyTDC is required to build the real dataloaders. "
            "Install it with `pip install PyTDC`."
        )

    cyp2d6_data = ADME(name="CYP2D6_Veith")
    cyp2d6_split = cyp2d6_data.get_split(method="scaffold")

    cyp2c19_data = ADME(name="CYP2C19_Veith")
    cyp2c19_df = _get_tdc_dataframe(cyp2c19_data)

    cyp2d6_train_df = _clean_property_dataframe(
        cyp2d6_split["train"], "Y_cyp2d6"
    )
    cyp2d6_valid_df = _clean_property_dataframe(
        cyp2d6_split["valid"], "Y_cyp2d6"
    )
    cyp2d6_test_df = _clean_property_dataframe(
        cyp2d6_split["test"], "Y_cyp2d6"
    )
    cyp2c19_df = _clean_property_dataframe(cyp2c19_df, "Y_cyp2c19")

    train_df = cyp2d6_train_df.merge(cyp2c19_df, on="Drug", how="inner")
    valid_df = cyp2d6_valid_df.merge(cyp2c19_df, on="Drug", how="inner")
    test_df = cyp2d6_test_df.merge(cyp2c19_df, on="Drug", how="inner")

    for split_name, split_df in {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }.items():
        if split_df.empty:
            raise ValueError(
                f"No overlapping CYP2D6/CYP2C19 molecules were found "
                f"in the {split_name} split."
            )

    train_smiles = train_df["Drug"].tolist()
    valid_smiles = valid_df["Drug"].tolist()
    test_smiles = test_df["Drug"].tolist()

    all_smiles = train_smiles + valid_smiles + test_smiles
    vocab = ["<pad>", "<sos>", "<eos>"] + sorted(set("".join(all_smiles)))
    char2idx = {character: index for index, character in enumerate(vocab)}
    idx2char = {index: character for character, index in char2idx.items()}
    vocab_size = len(vocab)

    encoded_train = torch.tensor(
        [encode_smiles(smi, char2idx, MAX_LENGTH) for smi in train_smiles],
        dtype=torch.long,
    )
    encoded_valid = torch.tensor(
        [encode_smiles(smi, char2idx, MAX_LENGTH) for smi in valid_smiles],
        dtype=torch.long,
    )
    encoded_test = torch.tensor(
        [encode_smiles(smi, char2idx, MAX_LENGTH) for smi in test_smiles],
        dtype=torch.long,
    )

    label_columns = ["Y_cyp2d6", "Y_cyp2c19"]
    train_labels = torch.tensor(
        train_df[label_columns].to_numpy(dtype=np.float32), dtype=torch.float
    )
    valid_labels = torch.tensor(
        valid_df[label_columns].to_numpy(dtype=np.float32), dtype=torch.float
    )
    test_labels = torch.tensor(
        test_df[label_columns].to_numpy(dtype=np.float32), dtype=torch.float
    )

    train_dataset = SMILESDataset(encoded_train, train_labels)
    valid_dataset = SMILESDataset(encoded_valid, valid_labels)
    test_dataset = SMILESDataset(encoded_test, test_labels)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    cyp2d6_train_labels = train_labels[:, 0]
    cyp2c19_train_labels = train_labels[:, 1]

    d6_pos = (cyp2d6_train_labels == 1).sum().item()
    d6_neg = (cyp2d6_train_labels == 0).sum().item()
    c19_pos = (cyp2c19_train_labels == 1).sum().item()
    c19_neg = (cyp2c19_train_labels == 0).sum().item()

    print("[split] method=TDC CYP2D6 scaffold")
    print("[split] dataset=CYP2D6/CYP2C19 overlap")
    print(
        "[split] sizes train/valid/test: "
        f"{len(train_df)}/{len(valid_df)}/{len(test_df)}"
    )
    print(f"[split] vocab_size: {vocab_size}")
    print(
        f"[CYP2D6] train pos/neg: {d6_pos}/{d6_neg}  "
        f"pos_weight: {d6_neg / d6_pos:.3f}"
    )
    print(
        f"[CYP2C19] train pos/neg: {c19_pos}/{c19_neg}  "
        f"pos_weight: {c19_neg / c19_pos:.3f}"
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
        vocab_size,
        char2idx,
        idx2char,
        train_labels,
    )
