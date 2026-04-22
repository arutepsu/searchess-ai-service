"""Training loop and per-epoch evaluation for the move-scoring architecture.

Loss function: in-context cross-entropy over legal moves.

  For each position, the model scores all N legal candidate moves.
  The loss is CE(-log softmax(scores)[played_idx]) where the softmax
  denominator is only over the legal moves for that position.

  This is equivalent to training the model to assign the highest score
  to the move a strong player chose, relative to the other legal moves
  available in that exact position.

  Padding positions are masked out (filled with -inf) before softmax so they
  don't affect the loss or accuracy.
"""

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
    optimizer: "torch.optim.Optimizer",
    device: "torch.device",
) -> dict[str, float]:
    """Run one training epoch. Returns {'loss': float, 'top1': float}."""
    model.train()
    total_loss = 0.0
    top1_correct = 0
    total = 0

    for pos_features, move_features, move_mask, played_idx in loader:
        pos_features = pos_features.to(device)
        move_features = move_features.to(device)
        move_mask = move_mask.to(device)
        played_idx = played_idx.to(device)

        optimizer.zero_grad()
        scores = model(pos_features, move_features)                    # (B, N)
        scores = scores.masked_fill(~move_mask, float("-inf"))         # mask padding
        loss = F.cross_entropy(scores, played_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(played_idx)
        top1_correct += (scores.argmax(dim=-1) == played_idx).sum().item()
        total += len(played_idx)

    return {
        "loss": total_loss / total if total else 0.0,
        "top1": top1_correct / total if total else 0.0,
    }


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: "torch.device",
) -> dict[str, float]:
    """Evaluate model on a DataLoader. Returns loss and top1 accuracy."""
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    total = 0

    with torch.no_grad():
        for pos_features, move_features, move_mask, played_idx in loader:
            pos_features = pos_features.to(device)
            move_features = move_features.to(device)
            move_mask = move_mask.to(device)
            played_idx = played_idx.to(device)

            scores = model(pos_features, move_features)
            scores = scores.masked_fill(~move_mask, float("-inf"))
            loss = F.cross_entropy(scores, played_idx)

            total_loss += loss.item() * len(played_idx)
            top1_correct += (scores.argmax(dim=-1) == played_idx).sum().item()
            total += len(played_idx)

    return {
        "loss": total_loss / total if total else 0.0,
        "top1": top1_correct / total if total else 0.0,
    }
