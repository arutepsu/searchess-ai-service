"""Offline evaluation report for the move-scoring architecture.

Computes simple metrics over a test DataLoader and produces a dict
that is persisted inside the model artifact as evaluation.json.

The report is tied to a specific (dataset_id, model_version, encoder_version,
move_encoder_version) tuple so that any future comparison can be traced back
to its exact provenance.

Metrics:
  top1_accuracy — fraction of samples where model's argmax = played move
  avg_loss      — average in-context cross-entropy loss on test set
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError(
        "torch is required. Install training extras: uv sync --extra training"
    ) from exc

if TYPE_CHECKING:
    import torch.nn as nn


def compute_report(
    model: "nn.Module",
    test_loader: DataLoader,
    device: "torch.device",
    dataset_sizes: dict[str, int],
    *,
    dataset_id: str,
    model_version: str,
    encoder_version: str,
    move_encoder_version: str,
) -> dict:
    """Evaluate model on test_loader and return a serialisable metrics dict."""
    model.eval()

    total_loss = 0.0
    top1_correct = 0
    total = 0

    with torch.no_grad():
        for pos_features, move_features, move_mask, played_idx in test_loader:
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

    avg_loss = total_loss / total if total > 0 else 0.0
    top1 = top1_correct / total if total > 0 else 0.0

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "dataset_id": dataset_id,
            "model_version": model_version,
            "encoder_version": encoder_version,
            "move_encoder_version": move_encoder_version,
        },
        "test_samples": total,
        "avg_loss": round(avg_loss, 6),
        "top1_accuracy": round(top1, 6),
        "dataset_sizes": dataset_sizes,
        "notes": "top1 = model argmax over legal moves matches played move",
    }
