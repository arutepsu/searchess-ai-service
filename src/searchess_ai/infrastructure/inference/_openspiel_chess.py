"""
OpenSpiel chess state operations.

This module handles the OpenSpiel-side of the mapping: loading chess states
from FEN and extracting legal moves as UCI strings. Notation policy (format
contracts, reconciliation) lives in _openspiel_mapping.py.

pyspiel is passed as an explicit argument to all functions so that tests can
inject a mock without requiring the native extension to be installed.
"""
from __future__ import annotations

from typing import Any

from searchess_ai.domain.game import LegalMoveSet, Move
from searchess_ai.infrastructure.inference._openspiel_decision import (
    OpenSpielMoveCandidate,
    is_promotion_move,
)
from searchess_ai.infrastructure.inference._openspiel_mapping import (
    ReconciliationPolicy,
    reconcile_moves,
)


def load_chess_state_from_fen(pyspiel: Any, fen: str) -> Any:
    """Load an OpenSpiel chess state for the given FEN position.

    Passes the FEN to new_initial_state as supported by OpenSpiel chess >= 1.0.
    Raises whatever pyspiel raises if the FEN is invalid or unsupported.
    Call validate_fen() before this function for a clearer error message on
    structurally malformed input.
    """
    game = pyspiel.load_game("chess")
    return game.new_initial_state(fen)


def get_legal_move_strings(state: Any) -> list[str]:
    """Return OpenSpiel legal moves as UCI strings (e.g. "e2e4", "e7e8q").

    action_to_string(player, action) is the documented OpenSpiel State API
    for converting integer actions to human-readable move strings.
    """
    player = state.current_player()
    return [state.action_to_string(player, a) for a in state.legal_actions()]


def _extract_is_capture(state: Any, player: int, action: Any) -> bool:
    """
    Best-effort capture detection for Phase G.

    We keep this intentionally defensive because pyspiel/mock state APIs may vary
    across environments and tests. If capture detection is unavailable, we
    safely return False and still retain promotion-based ordering.
    """
    if hasattr(state, "is_capture"):
        try:
            return bool(state.is_capture(action))
        except TypeError:
            try:
                return bool(state.is_capture(player, action))
            except Exception:
                return False
        except Exception:
            return False
    return False


def get_legal_move_candidates(state: Any) -> list[OpenSpielMoveCandidate]:
    """
    Return enriched OpenSpiel move candidates in original OpenSpiel order.

    Phase G currently extracts:
      - move string
      - promotion flag
      - capture flag (best effort)
    """
    player = state.current_player()
    candidates: list[OpenSpielMoveCandidate] = []

    for action in state.legal_actions():
        move = state.action_to_string(player, action)
        candidates.append(
            OpenSpielMoveCandidate(
                move=move,
                is_capture=_extract_is_capture(state, player, action),
                is_promotion=is_promotion_move(move),
            )
        )

    return candidates


def select_move(openspiel_moves: list[str], request_moves: LegalMoveSet) -> Move:
    """Select a move using the default reconciliation policy.

    Delegates to reconcile_moves with INTERSECT_THEN_PLATFORM_FALLBACK.
    Preserved for backward compatibility; prefer calling reconcile_moves
    directly when the policy needs to be explicit or the full result
    metadata is required.
    """
    result = reconcile_moves(
        openspiel_moves,
        request_moves,
        ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK,
    )
    return result.selected_move