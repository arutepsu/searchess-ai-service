"""Runtime decision types — the primary transparency contract.

Every call to SupervisedModelRuntime.select_move() returns a RuntimeDecision.
Nothing is silent: every decision is either MODEL (a real model output) or
FALLBACK (a safe default with an explicit reason).

This module is the contract between the model runtime and the inference engine.
Do not import from training or data_pipeline here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DecisionMode(str, Enum):
    """How the move was selected."""

    MODEL = "model"
    """Model scored and ranked legal moves; returned highest-confidence choice."""

    FALLBACK = "fallback"
    """Model could not produce a usable output; safe default was returned."""

    ERROR_RECOVERED = "error_recovered"
    """An unexpected error occurred; the engine recovered with a safe default."""


class FallbackReason(str, Enum):
    """Why a fallback was triggered. Always present when decision_mode != MODEL."""

    MODEL_FAILURE = "model_failure"
    """Feature encoding or forward pass failed (e.g. invalid FEN, torch error)."""

    INVALID_OUTPUT = "invalid_output"
    """Model output could not be mapped to any legal move vocabulary index."""

    NO_LEGAL_MOVES = "no_legal_moves"
    """Legal move list was empty — should not occur if Game Service validates correctly."""

    RUNTIME_EXCEPTION = "runtime_exception"
    """Unexpected exception type not covered by MODEL_FAILURE or INVALID_OUTPUT."""


@dataclass(frozen=True, slots=True)
class RuntimeDecision:
    """The full result of a single move-selection call.

    Fields:
      selected_uci    the move to play (always a member of the legal move list)
      decision_mode   MODEL, FALLBACK, or ERROR_RECOVERED
      confidence      softmax probability of selected move (None for fallbacks)
      fallback_reason why fallback was used (None when decision_mode == MODEL)
      error_detail    truncated exception message for logs (None when no error)
    """

    selected_uci: str
    decision_mode: DecisionMode
    confidence: float | None = None
    fallback_reason: FallbackReason | None = None
    error_detail: str | None = None

    def is_model_decision(self) -> bool:
        return self.decision_mode == DecisionMode.MODEL

    def is_fallback(self) -> bool:
        return self.decision_mode in (DecisionMode.FALLBACK, DecisionMode.ERROR_RECOVERED)

    def to_log_dict(self) -> dict:
        """Serialisable snapshot for structured logging."""
        return {
            "selected_uci": self.selected_uci,
            "decision_mode": self.decision_mode.value,
            "confidence": self.confidence,
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
            "error_detail": self.error_detail,
        }
