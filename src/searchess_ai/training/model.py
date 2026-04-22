"""Move-scoring policy network.

Architecture: per-move scoring via independent position and move branches.

  Given:
    position_features  — float32 (FEATURE_SIZE,)   board encoding
    move_features      — float32 (N, MOVE_FEATURE_SIZE)  N candidate moves

  The model:
    1. Encodes the position through a small MLP → position hidden vector
    2. Encodes each candidate move through a shared MLP → move hidden vector
    3. Concatenates position and move hidden vectors
    4. Scores the pair with a linear head → scalar score per move

  Output: float32 (N,) — one score per candidate move

At inference the score vector is softmax-normalised and argmax is taken over
the legal moves.  At training it is used with cross-entropy over the legal
move set (softmax denominator = legal moves only), so the model learns to rank
the played move above all other legal moves in the same position.

This replaces the old PolicyNetwork (global 4096-logit vocabulary) entirely.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "torch is required. Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.model_runtime.encoder import FEATURE_SIZE
from searchess_ai.model_runtime.move_encoder import MOVE_FEATURE_SIZE


def _make_mlp(in_size: int, out_size: int, num_layers: int, dropout: float) -> nn.Sequential:
    """Build a ReLU MLP with num_layers hidden layers of size out_size."""
    layers: list[nn.Module] = []
    current = in_size
    for _ in range(num_layers):
        layers += [nn.Linear(current, out_size), nn.ReLU(), nn.Dropout(dropout)]
        current = out_size
    return nn.Sequential(*layers)


class MoveScoringNetwork(nn.Module):
    """Scores each candidate legal move given the board position."""

    def __init__(
        self,
        position_feature_size: int = FEATURE_SIZE,
        move_feature_size: int = MOVE_FEATURE_SIZE,
        hidden_size: int = 256,
        num_hidden_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self._position_feature_size = position_feature_size
        self._move_feature_size = move_feature_size
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._dropout = dropout

        # Each branch encodes its input to hidden_size dimensions.
        self.pos_branch = _make_mlp(position_feature_size, hidden_size, num_hidden_layers, dropout)
        self.move_branch = _make_mlp(move_feature_size, hidden_size, num_hidden_layers, dropout)

        # Score head: takes [pos_hidden || move_hidden] → scalar
        self.score_head = nn.Linear(2 * hidden_size, 1)

    def forward(
        self,
        pos_features: "torch.Tensor",
        move_features: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute one score per candidate move.

        Args:
          pos_features:  (B, position_feature_size) or (1, position_feature_size)
          move_features: (B, N, move_feature_size)

        Returns:
          scores: (B, N) — higher score = model prefers this move
        """
        B, N, _ = move_features.shape

        # Encode position: (B, hidden_size)
        pos_h = self.pos_branch(pos_features)

        # Encode each move independently: reshape to (B*N, F), apply MLP, reshape back
        move_h = self.move_branch(move_features.view(B * N, -1)).view(B, N, -1)

        # Broadcast position encoding across all N moves
        pos_h_expanded = pos_h.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_size)

        combined = torch.cat([pos_h_expanded, move_h], dim=-1)  # (B, N, 2*hidden_size)
        scores = self.score_head(combined).squeeze(-1)           # (B, N)
        return scores

    def config(self) -> dict:
        """Serialisable architecture config for artifact packaging."""
        return {
            "architecture": "MoveScoringNetwork",
            "position_feature_size": self._position_feature_size,
            "move_feature_size": self._move_feature_size,
            "hidden_size": self._hidden_size,
            "num_hidden_layers": self._num_hidden_layers,
            "dropout": self._dropout,
        }

    @staticmethod
    def from_config(cfg: dict) -> "MoveScoringNetwork":
        return MoveScoringNetwork(
            position_feature_size=cfg["position_feature_size"],
            move_feature_size=cfg["move_feature_size"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            dropout=cfg["dropout"],
        )
