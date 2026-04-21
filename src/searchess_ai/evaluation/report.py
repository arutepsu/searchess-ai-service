"""Offline evaluation report generation.

Computes simple metrics over a test DataLoader and produces a dict
that is persisted inside the model artifact as evaluation.json.

The report is tied to a specific (dataset_id, model_version, encoder_version)
triple so that any future comparison can be traced back to its exact provenance.

Metrics:
  top1_accuracy — fraction of samples where model's best move = played move
  top5_accuracy — fraction of samples where played move is in top-5 predictions
  avg_loss      — average masked cross-entropy loss on test set
  legal_move_selection_rate — sanity check: should always be 1.0 with masking
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
    device: torch.device,
    dataset_sizes: dict[str, int],
    *,
    dataset_id: str,
    model_version: str,
    encoder_version: str,
) -> dict:
    """Evaluate model on test_loader and return a serialisable metrics dict.

    Args:
      dataset_id      — ties the report to the exact dataset used for training
      model_version   — ties the report to the exact artifact version
      encoder_version — confirms encoding contract in effect during evaluation
    """
    model.eval()

    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    legal_correct = 0
    total = 0

    with torch.no_grad():
        for features, labels, masks in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(features)
            masked_logits = logits.masked_fill(masks == 0, float("-inf"))

            loss = F.cross_entropy(masked_logits, labels)
            total_loss += loss.item() * len(labels)

            top1_preds = masked_logits.argmax(dim=-1)
            top1_correct += (top1_preds == labels).sum().item()

            k = min(5, masked_logits.size(-1))
            top5_preds = masked_logits.topk(k, dim=-1).indices
            top5_correct += top5_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            # Sanity: predicted move must be legal (mask bit = 1)
            legal_correct += masks[range(len(labels)), top1_preds].sum().item()

            total += len(labels)

    avg_loss = total_loss / total if total > 0 else 0.0
    top1 = top1_correct / total if total > 0 else 0.0
    top5 = top5_correct / total if total > 0 else 0.0
    legal_rate = legal_correct / total if total > 0 else 0.0

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "dataset_id": dataset_id,
            "model_version": model_version,
            "encoder_version": encoder_version,
        },
        "test_samples": total,
        "avg_loss": round(avg_loss, 6),
        "top1_accuracy": round(top1, 6),
        "top5_accuracy": round(top5, 6),
        "legal_move_selection_rate": round(legal_rate, 6),
        "dataset_sizes": dataset_sizes,
        "notes": (
            "top1 = model best move matches played move; "
            "legal_move_selection_rate should be 1.0 with masking"
        ),
    }
