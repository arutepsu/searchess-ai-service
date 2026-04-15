"""
OpenSpiel-backed inference engine prototype.

OpenSpiel is NOT imported at module level — it is loaded lazily inside
choose_move(). This means:
  - Constructing OpenSpielInferenceEngine() succeeds even if pyspiel is
    not installed (wiring tests can assert the backend type).
  - The ImportError only surfaces when inference is actually attempted,
    and is wrapped in OpenSpielAdapterError for a clean structured 500.

See _openspiel_mapping.py for notation boundary contracts and reconciliation
policy documentation. See _openspiel_chess.py for OpenSpiel state operations.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from searchess_ai.application.port.inference_engine import InferenceEngine
from searchess_ai.domain.inference import DecisionType, InferenceDecision, InferenceRequest
from searchess_ai.domain.model import ModelId, ModelVersion
from searchess_ai.infrastructure.inference._openspiel_chess import (
    get_legal_move_candidates,
    load_chess_state_from_fen,
)
from searchess_ai.infrastructure.inference._openspiel_decision import ordered_move_strings
from searchess_ai.infrastructure.inference._openspiel_mapping import (
    MoveReconciliationResult,
    ReconciliationPolicy,
    reconcile_moves,
    validate_fen,
)

# Re-export so existing callers (api/errors.py, api/app.py) keep working
# without any import changes.
from searchess_ai.infrastructure.inference._openspiel_mapping import (  # noqa: F401
    OpenSpielAdapterError,
)


def _require_pyspiel() -> Any:
    """Return the pyspiel module, or raise OpenSpielAdapterError if absent."""
    try:
        return importlib.import_module("pyspiel")
    except ImportError as exc:
        raise OpenSpielAdapterError(
            "pyspiel is not installed. "
            "Set INFERENCE_BACKEND=fake or INFERENCE_BACKEND=random, "
            "or install open-spiel on a supported platform."
        ) from exc


@dataclass(slots=True)
class OpenSpielInferenceEngine(InferenceEngine):
    """Prototype inference engine backed by OpenSpiel chess.

    Action selection is deterministic. Phase G enriches candidate ordering
    before reconciliation:
      - promotions first
      - then captures
      - then other moves in original OpenSpiel order

    The reconciliation_policy field controls what happens when the OpenSpiel
    move list and request.legal_moves do not overlap.
    """

    default_model_id: ModelId = ModelId("openspiel-chess-v0")
    default_model_version: ModelVersion = ModelVersion("0.1.0")
    reconciliation_policy: ReconciliationPolicy = (
        ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK
    )

    def choose_move(self, request: InferenceRequest) -> InferenceDecision:
        pyspiel = _require_pyspiel()
        fen = request.position.value

        try:
            validate_fen(fen)
            state = load_chess_state_from_fen(pyspiel, fen)
            candidates = get_legal_move_candidates(state)
            ranked_moves = ordered_move_strings(candidates)
        except OpenSpielAdapterError:
            raise
        except Exception as exc:
            raise OpenSpielAdapterError(
                f"OpenSpiel failed to process position '{fen}': {exc}"
            ) from exc

        result: MoveReconciliationResult = reconcile_moves(
            ranked_moves,
            request.legal_moves,
            self.reconciliation_policy,
        )

        return InferenceDecision(
            request_id=request.request_id,
            decision_type=DecisionType.MOVE,
            selected_move=result.selected_move,
            model_id=request.model_id or self.default_model_id,
            model_version=request.model_version or self.default_model_version,
            decision_time_millis=0,
            policy_profile=request.policy_profile,
        )