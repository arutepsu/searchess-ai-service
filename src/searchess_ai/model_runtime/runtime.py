"""Supervised model runtime — scores and selects from legal moves.

Every call to select_move() returns a RuntimeDecision.
No path is silent: every fallback carries a FallbackReason and optional
error_detail for logs. The calling engine is responsible for logging
and for deciding whether to surface the decision_mode to the API layer.
"""

from __future__ import annotations

from searchess_ai.model_runtime.decision import (
    DecisionMode,
    FallbackReason,
    RuntimeDecision,
)
from searchess_ai.model_runtime.encoder import encode_fen, encode_uci_move
from searchess_ai.model_runtime.loader import LoadedModel

# Promotion piece priority: prefer queen, then rook, then bishop, then knight.
_PROMO_PRIORITY = {"q": 0, "r": 1, "b": 2, "n": 3, "": 4}

# Maximum characters from an exception message included in error_detail.
_ERROR_DETAIL_MAX_LEN = 300


class SupervisedModelRuntime:
    """Wraps a LoadedModel and provides explicit, classified move selection."""

    def __init__(self, loaded: LoadedModel) -> None:
        self._loaded = loaded
        try:
            import torch
            self._torch = torch
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

        Raises ValueError only if legal_uci_moves is empty, which is a
        programming error (Game Service must always provide ≥1 legal move).
        """
        if not legal_uci_moves:
            return RuntimeDecision(
                selected_uci="",
                decision_mode=DecisionMode.FALLBACK,
                fallback_reason=FallbackReason.NO_LEGAL_MOVES,
                error_detail="legal_uci_moves was empty; Game Service should prevent this",
            )

        try:
            features = encode_fen(fen)
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

        try:
            x = self._torch.from_numpy(features).unsqueeze(0)  # (1, FEATURE_SIZE)
            with self._torch.no_grad():
                logits = self._loaded.model(x).squeeze(0)  # (MOVE_VOCAB_SIZE,)
        except Exception as exc:
            return _fallback(
                legal_uci_moves,
                FallbackReason.MODEL_FAILURE,
                f"Forward pass failed: {type(exc).__name__}: {exc}",
            )

        try:
            return _pick_best(self._torch, logits, legal_uci_moves)
        except Exception as exc:
            return _fallback(
                legal_uci_moves,
                FallbackReason.RUNTIME_EXCEPTION,
                f"Move selection failed: {type(exc).__name__}: {exc}",
            )


# ---------------------------------------------------------------------------
# Module-level helpers (no self state needed)
# ---------------------------------------------------------------------------


def _pick_best(torch_module, logits, legal_moves: list[str]) -> RuntimeDecision:
    """Select the highest-probability legal move.

    Returns FALLBACK/INVALID_OUTPUT if no legal move maps to a valid vocab index.
    Promotion tie-breaking: prefer queen > rook > bishop > knight.
    """
    import torch.nn.functional as F

    # Group legal moves by vocabulary index; keep preferred promotion per index.
    index_to_best_uci: dict[int, str] = {}
    skipped = 0
    for uci in legal_moves:
        try:
            idx = encode_uci_move(uci)
        except ValueError:
            skipped += 1
            continue
        if idx not in index_to_best_uci:
            index_to_best_uci[idx] = uci
        else:
            existing = index_to_best_uci[idx]
            if _promo_priority(uci) < _promo_priority(existing):
                index_to_best_uci[idx] = uci

    if not index_to_best_uci:
        return _fallback(
            legal_moves,
            FallbackReason.INVALID_OUTPUT,
            f"All {len(legal_moves)} legal moves failed UCI encoding ({skipped} skipped)",
        )

    indices = list(index_to_best_uci.keys())
    scores = logits[indices]
    probs = F.softmax(scores, dim=0)
    best_local = int(probs.argmax().item())
    best_uci = index_to_best_uci[indices[best_local]]
    confidence = float(probs[best_local].item())
    confidence = max(0.0, min(1.0, confidence))

    return RuntimeDecision(
        selected_uci=best_uci,
        decision_mode=DecisionMode.MODEL,
        confidence=confidence,
    )


def _fallback(
    legal_moves: list[str],
    reason: FallbackReason,
    error_detail: str,
) -> RuntimeDecision:
    """Return the first legal move with explicit fallback classification."""
    return RuntimeDecision(
        selected_uci=legal_moves[0],
        decision_mode=DecisionMode.FALLBACK,
        confidence=None,
        fallback_reason=reason,
        error_detail=error_detail[:_ERROR_DETAIL_MAX_LEN],
    )


def _promo_priority(uci: str) -> int:
    if len(uci) == 5:
        return _PROMO_PRIORITY.get(uci[4], 99)
    return _PROMO_PRIORITY[""]
