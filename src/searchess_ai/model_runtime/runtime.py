"""Move-scoring runtime — scores and selects from legal moves directly.

Every call to select_move() returns a RuntimeDecision.
No path is silent: every fallback carries a FallbackReason and optional
error_detail for logs.

The inference procedure
-----------------------
1. Encode the position (FEN → float32 vector via encoder.py).
2. Encode each legal move (UCI → float32 vector via move_encoder.py).
3. Score all legal moves jointly (MoveScoringNetwork forward pass).
4. Return the highest-scoring legal move with its softmax confidence.

No fixed vocabulary. No masking. Each legal move is scored directly.
Underpromotions are distinct from queen promotions.
"""

from __future__ import annotations

from searchess_ai.model_runtime.decision import (
    DecisionMode,
    FallbackReason,
    RuntimeDecision,
)
from searchess_ai.model_runtime.encoder import encode_fen
from searchess_ai.model_runtime.loader import LoadedModel
from searchess_ai.model_runtime.move_encoder import encode_moves

_ERROR_DETAIL_MAX_LEN = 300


class SupervisedModelRuntime:
    """Wraps a LoadedModel and provides explicit, classified move selection."""

    def __init__(self, loaded: LoadedModel) -> None:
        self._loaded = loaded
        try:
            import torch
            import torch.nn.functional as F
            self._torch = torch
            self._F = F
        except ImportError as exc:
            raise ImportError(
                "torch is required. Install training extras: uv sync --extra training"
            ) from exc

    @property
    def model_version(self) -> str:
        return self._loaded.model_version

    @property
    def artifact_id(self) -> str:
        return self._loaded.artifact_id

    def select_move(self, fen: str, legal_uci_moves: list[str]) -> RuntimeDecision:
        """Score legal_uci_moves given the board position in fen.

        Always returns a RuntimeDecision — never raises.
        decision_mode is MODEL when the model successfully scored moves,
        FALLBACK or ERROR_RECOVERED otherwise (with explicit fallback_reason).
        """
        if not legal_uci_moves:
            return RuntimeDecision(
                selected_uci="",
                decision_mode=DecisionMode.FALLBACK,
                fallback_reason=FallbackReason.NO_LEGAL_MOVES,
                error_detail="legal_uci_moves was empty; Game Service should prevent this",
            )

        # Step 1: encode position
        try:
            pos_features = encode_fen(fen)
        except ValueError as exc:
            return _fallback(
                legal_uci_moves,
                FallbackReason.MODEL_FAILURE,
                f"FEN encoding failed: {exc}",
            )
        except Exception as exc:
            return _fallback(
                legal_uci_moves,
                FallbackReason.RUNTIME_EXCEPTION,
                f"Unexpected error during FEN encoding: {type(exc).__name__}: {exc}",
            )

        # Step 2: encode legal moves
        try:
            move_features = encode_moves(legal_uci_moves)
        except ValueError as exc:
            return _fallback(
                legal_uci_moves,
                FallbackReason.MODEL_FAILURE,
                f"Move encoding failed: {exc}",
            )
        except Exception as exc:
            return _fallback(
                legal_uci_moves,
                FallbackReason.RUNTIME_EXCEPTION,
                f"Unexpected error during move encoding: {type(exc).__name__}: {exc}",
            )

        # Step 3: score and select
        try:
            torch = self._torch
            F = self._F

            # pos: (1, FEATURE_SIZE), moves: (1, N, MOVE_FEATURE_SIZE)
            pos_t = torch.from_numpy(pos_features).unsqueeze(0)
            move_t = torch.from_numpy(move_features).unsqueeze(0)

            with torch.no_grad():
                scores = self._loaded.model(pos_t, move_t).squeeze(0)  # (N,)

            probs = F.softmax(scores, dim=0)
            best_idx = int(probs.argmax().item())
            confidence = float(probs[best_idx].item())
            confidence = max(0.0, min(1.0, confidence))

            return RuntimeDecision(
                selected_uci=legal_uci_moves[best_idx],
                decision_mode=DecisionMode.MODEL,
                confidence=confidence,
            )

        except Exception as exc:
            return _fallback(
                legal_uci_moves,
                FallbackReason.MODEL_FAILURE,
                f"Scoring failed: {type(exc).__name__}: {exc}",
            )


def _fallback(legal_moves: list[str], reason: FallbackReason, detail: str) -> RuntimeDecision:
    """Return the first legal move with explicit fallback classification."""
    return RuntimeDecision(
        selected_uci=legal_moves[0],
        decision_mode=DecisionMode.FALLBACK,
        confidence=None,
        fallback_reason=reason,
        error_detail=detail[:_ERROR_DETAIL_MAX_LEN],
    )
