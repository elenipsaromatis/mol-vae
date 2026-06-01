import torch
import torch.nn as nn


def vae_loss(logits, targets, mu, log_var, beta, prop_logit=None, labels=None, gamma=0.0, kl_free_bits=0.5, train_labels=None):
    recon_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')(
        logits.view(-1, logits.size(-1)),
        targets.reshape(-1)
    ) / (targets != 0).sum()

    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=-1))

    prop_loss = torch.tensor(0.0, device=logits.device)
    if prop_logit is not None and labels is not None and gamma > 0 and train_labels is not None:
        pos_weight = torch.tensor(
            (train_labels == 0).sum().item() / (train_labels == 1).sum().item(),
            device=logits.device
        )
        prop_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(prop_logit, labels)

    loss = recon_loss + beta * kl_loss + gamma * prop_loss
    return loss, recon_loss, kl_loss, prop_loss


def evaluate_loss(model, loader, beta, gamma, device, kl_free_bits, train_labels):
    model.eval()
    total_recon, total_kl, total_prop, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            labels = labels.to(device)
            logits, mu, log_var, prop_logit = model(batch)
            _, recon, kl, prop = vae_loss(
                logits, batch[:, 1:], mu, log_var, beta,
                prop_logit=prop_logit, labels=labels,
                gamma=gamma, kl_free_bits=kl_free_bits,
                train_labels=train_labels
            )
            total_recon += recon.item()
            total_kl += kl.item()
            total_prop += prop.item()
            n += 1
    return total_recon / n, total_kl / n, total_prop / n