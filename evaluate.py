import torch
from sklearn.metrics import roc_auc_score


def evaluate_auroc(model, loader, device):
    model.eval()
    all_prop_logits, all_labels = [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            mu, _ = model.encoder(batch)
            prop_logit = model.predictor(mu)
            all_prop_logits.append(prop_logit.cpu())
            all_labels.append(labels)
    prop_logits = torch.cat(all_prop_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return roc_auc_score(labels, prop_logits)