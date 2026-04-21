"""Baseline supervised policy network.

Architecture: fully-connected feed-forward network.
  Input:  FEATURE_SIZE-dimensional board encoding
  Output: MOVE_VOCAB_SIZE logits (one per (from_sq, to_sq) pair)

At training time, illegal moves are masked out before cross-entropy loss.
At inference time, the runtime masks by the legal moves provided by Game Service.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "torch is required. Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.model_runtime.encoder import FEATURE_SIZE, MOVE_VOCAB_SIZE


class PolicyNetwork(nn.Module):
    """Simple MLP policy network for supervised move prediction."""

    def __init__(
        self,
        feature_size: int = FEATURE_SIZE,
        hidden_size: int = 512,
        num_hidden_layers: int = 2,
        move_vocab_size: int = MOVE_VOCAB_SIZE,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_size = feature_size
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = hidden_size
        layers.append(nn.Linear(in_size, move_vocab_size))

        self.net = nn.Sequential(*layers)

        self._feature_size = feature_size
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._move_vocab_size = move_vocab_size
        self._dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, feature_size) → logits: (batch, move_vocab_size)"""
        return self.net(x)

    def config(self) -> dict:
        """Return serialisable architecture config for artifact packaging."""
        return {
            "architecture": "PolicyNetwork",
            "feature_size": self._feature_size,
            "hidden_size": self._hidden_size,
            "num_hidden_layers": self._num_hidden_layers,
            "move_vocab_size": self._move_vocab_size,
            "dropout": self._dropout,
        }

    @staticmethod
    def from_config(cfg: dict) -> "PolicyNetwork":
        return PolicyNetwork(
            feature_size=cfg["feature_size"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            move_vocab_size=cfg["move_vocab_size"],
            dropout=cfg["dropout"],
        )
