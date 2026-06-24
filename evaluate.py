import torch
from rdkit import Chem
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from model import reparameterise


TASK_COLUMN = {
    "cyp2d6": 0,
    "cyp2c19": 1,
}


def _predictor_for(model, task):
    return getattr(model, f"{task}_predictor")


def _collect_property(model, loader, device, task):
    """Return (labels, logits) numpy arrays for one task. Predictor reads mu."""
    column = TASK_COLUMN[task]
    predictor = _predictor_for(model, task)
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch, batch_labels in loader:
            batch = batch.to(device)
            mu, _ = model.encoder(batch)
            logits.append(predictor(mu).cpu())
            labels.append(batch_labels[:, column])
    return torch.cat(labels).numpy(), torch.cat(logits).numpy()


def evaluate_auroc(model, loader, device, task):
    labels, logits = _collect_property(model, loader, device, task)
    return roc_auc_score(labels, logits)


def evaluate_auprc(model, loader, device, task):
    labels, logits = _collect_property(model, loader, device, task)
    precision, recall, _ = precision_recall_curve(labels, logits)
    return auc(recall, precision)


def evaluate(model, dataset, vocab_size, device, idx2char, n_samples=500):
    """Reconstruction metric on a random subset. Decoder reads mu (deterministic)."""
    model.eval()
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


def evaluate_test(model, loader, vocab_size, idx2char, device):
    """Full test metrics as a dict. Decoder reads mu (deterministic), predictors read mu."""
    model.eval()
    all_logits, all_targets = [], []
    prop_logits = {task: [] for task in TASK_COLUMN}
    prop_labels = {task: [] for task in TASK_COLUMN}

    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            mu, log_var = model.encoder(batch)
            z = reparameterise(mu, log_var)
            logits = model.decoder(mu, batch)

            all_logits.append(logits)
            all_targets.append(batch[:, 1:])

            for task, column in TASK_COLUMN.items():
                predictor = _predictor_for(model, task)
                prop_logits[task].append(predictor(mu).cpu())
                prop_labels[task].append(labels[:, column])

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

    metrics = {"recon_acc": recon_acc, "validity": validity}
    for task in TASK_COLUMN:
        y_true = torch.cat(prop_labels[task]).numpy()
        y_score = torch.cat(prop_logits[task]).numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics[f"{task}_auroc"] = roc_auc_score(y_true, y_score)
        metrics[f"{task}_auprc"] = auc(recall, precision)

    return metrics