"""Training loop and per-epoch evaluation logic."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError(
        "torch is required. Install training extras: uv sync --extra training"
    ) from exc


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch. Returns {'loss': float, 'top1': float, 'top5': float}."""
    model.train()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    for features, labels, masks in loader:
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(features)

        masked_logits = logits.masked_fill(masks == 0, float("-inf"))
        loss = F.cross_entropy(masked_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        top1_correct += _top_k_correct(masked_logits, labels, k=1)
        top5_correct += _top_k_correct(masked_logits, labels, k=5)
        total += len(labels)

    return {
        "loss": total_loss / total if total else 0.0,
        "top1": top1_correct / total if total else 0.0,
        "top5": top5_correct / total if total else 0.0,
    }


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a DataLoader. Returns loss, top1, top5 accuracy."""
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for features, labels, masks in loader:
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(features)
            masked_logits = logits.masked_fill(masks == 0, float("-inf"))
            loss = F.cross_entropy(masked_logits, labels)

            total_loss += loss.item() * len(labels)
            top1_correct += _top_k_correct(masked_logits, labels, k=1)
            top5_correct += _top_k_correct(masked_logits, labels, k=5)
            total += len(labels)

    return {
        "loss": total_loss / total if total else 0.0,
        "top1": top1_correct / total if total else 0.0,
        "top5": top5_correct / total if total else 0.0,
    }


def _top_k_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    """Count how many predictions have the label in the top-k logits."""
    top_k = logits.topk(min(k, logits.size(-1)), dim=-1).indices  # (batch, k)
    correct = top_k.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
    return int(correct)
