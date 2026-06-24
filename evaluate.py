import torch
from rdkit import Chem


def _collect_regression(model, loader, device):
    """Return (labels, preds) tensors for the regression head. Predictor reads mu."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch, batch_labels in loader:
            batch = batch.to(device)
            mu, _ = model.encoder(batch)
            preds.append(model.reg_predictor(mu).reshape(-1).cpu())
            labels.append(batch_labels[:, 0])
    return torch.cat(labels), torch.cat(preds)


def evaluate_regression(model, loader, device, reg_mean, reg_std):
    """RMSE and MAE in original target units (inverts standardization)."""
    labels, preds = _collect_regression(model, loader, device)

    reg_mean = reg_mean.cpu()
    reg_std = reg_std.cpu()
    preds = preds * reg_std + reg_mean
    labels = labels * reg_std + reg_mean

    rmse = torch.sqrt(((preds - labels) ** 2).mean()).item()
    mae = (preds - labels).abs().mean().item()
    return {"rmse": rmse, "mae": mae}


def evaluate(model, dataset, vocab_size, device, idx2char, n_samples=500):
    """Reconstruction metric on a random subset. Decoder reads mu (deterministic)."""
    model.eval()
    n_samples = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:n_samples]
    batch = torch.stack([dataset[i][0] for i in indices]).to(device)

    with torch.no_grad():
        mu, _ = model.encoder(batch)
        logits = model.decoder(mu, batch)
        pred_tokens = logits.argmax(dim=-1)

    targets = batch[:, 1:]
    exact_acc = (pred_tokens == targets).all(dim=1).float().mean().item()
    mask = targets != 0
    token_acc = (pred_tokens[mask] == targets[mask]).float().mean().item()

    valid_count = 0
    for seq in pred_tokens:
        chars = [idx2char[idx.item()] for idx in seq]
        smiles = ''
        for ch in chars:
            if ch == '<eos>':
                break
            if ch not in ['<pad>', '<sos>']:
                smiles += ch
        if smiles and Chem.MolFromSmiles(smiles) is not None:
            valid_count += 1
    validity = valid_count / n_samples

    return exact_acc, token_acc, validity


def evaluate_test(model, loader, vocab_size, idx2char, device, reg_mean, reg_std):
    """Full test metrics as a dict. Decoder reads mu (deterministic), predictor reads mu."""
    model.eval()
    all_logits, all_targets = [], []
    preds, labels = [], []

    with torch.no_grad():
        for batch, batch_labels in loader:
            batch = batch.to(device)
            mu, _ = model.encoder(batch)
            logits = model.decoder(mu, batch)

            all_logits.append(logits)
            all_targets.append(batch[:, 1:])

            preds.append(model.reg_predictor(mu).reshape(-1).cpu())
            labels.append(batch_labels[:, 0])

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    pred_tokens = all_logits.argmax(dim=-1)
    recon_acc = (pred_tokens == all_targets).all(dim=1).float().mean().item()

    valid_count = 0
    for seq in pred_tokens:
        chars = [idx2char[idx.item()] for idx in seq]
        smiles = ''
        for ch in chars:
            if ch == '<eos>':
                break
            if ch not in ['<pad>', '<sos>']:
                smiles += ch
        if smiles and Chem.MolFromSmiles(smiles) is not None:
            valid_count += 1
    validity = valid_count / len(pred_tokens)

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    reg_mean = reg_mean.cpu()
    reg_std = reg_std.cpu()
    preds = preds * reg_std + reg_mean
    labels = labels * reg_std + reg_mean

    rmse = torch.sqrt(((preds - labels) ** 2).mean()).item()
    mae = (preds - labels).abs().mean().item()

    return {
        "recon_acc": recon_acc,
        "validity": validity,
        "rmse": rmse,
        "mae": mae,
    }