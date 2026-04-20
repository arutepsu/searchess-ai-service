"""
OpenSpiel ↔ platform notation mapping policy.

Notation authority
------------------
The Scala searchess platform is the authority for:
  - FEN position semantics
  - Move notation (UCI strings)
  - Which moves are legal in any given position

This module makes the boundary contract and reconciliation behavior explicit
and testable. It does not redefine notation — it maps from the platform
contract to what OpenSpiel expects, and resolves any disagreement using an
explicit named policy.

Boundary contracts
------------------
  POSITION_FORMAT = "FEN"  — positions arrive as FEN strings from the platform
  MOVE_FORMAT     = "UCI"  — moves are UCI strings (e.g. "e2e4", "e7e8q")

OpenSpiel chess action_to_string() also produces UCI notation, so in the
common case the formats are identical. Divergence can occur for promotion
notation or castling representations; the reconciliation policy handles
these cases safely without ever returning a move outside the platform set.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from searchess_ai.domain.game import LegalMoveSet, Move

# ── Boundary format constants ───────────────────────────────────────────────

POSITION_FORMAT: Final[str] = "FEN"
"""Position format at the AI service boundary.
FEN strings are produced and authoritative in the Scala searchess platform.
This service consumes FEN — it does not define or validate FEN semantics."""

MOVE_FORMAT: Final[str] = "UCI"
"""Move format at the AI service boundary.
UCI strings (e.g. "e2e4", "g1f3", "e7e8q") are the canonical move notation
used by both the Scala platform and OpenSpiel chess action_to_string."""


# ── Error type ──────────────────────────────────────────────────────────────

class OpenSpielAdapterError(Exception):
    """Raised when the OpenSpiel adapter cannot process a request.

    Causes: pyspiel not installed, unsupported position format,
    unparseable FEN, reconciliation policy failure, or any other
    adapter-level failure.
    """


class BadPositionAdapterError(OpenSpielAdapterError):
    """Raised specifically when the FEN string is structurally invalid or
    the position cannot be loaded — maps to BAD_POSITION (422) at the HTTP boundary."""


# ── Position format validation ──────────────────────────────────────────────

_FEN_MIN_FIELDS = 2  # board field + side-to-move; clock fields are optional


def validate_fen(fen: str) -> None:
    """Raise OpenSpielAdapterError if the FEN string is structurally unsupported.

    This is a lightweight structural guard — it does not validate chess
    legality. The Scala platform is the authority for that. We only check
    that the string looks minimally parseable before handing it to OpenSpiel,
    so failures are caught early with a clear message rather than a cryptic
    native-extension error.

    Accepts any FEN with the required fields. Does not validate piece counts,
    castling rights, or en-passant square consistency.
    """
    if not fen or not fen.strip():
        raise BadPositionAdapterError(
            "Unsupported position format: FEN string is empty. "
            "Position must be a valid FEN string produced by the platform."
        )
    parts = fen.strip().split()
    if len(parts) < _FEN_MIN_FIELDS:
        raise BadPositionAdapterError(
            f"Unsupported position format: expected at least {_FEN_MIN_FIELDS} "
            f"FEN fields (board + side-to-move), got {len(parts)}. "
            f"Input: {fen!r}"
        )
    board_field, side_field = parts[0], parts[1]
    if board_field.count("/") != 7:
        raise BadPositionAdapterError(
            f"Unsupported position format: FEN board field must contain exactly "
            f"7 rank separators ('/'), got {board_field.count('/')}. "
            f"Input: {fen!r}"
        )
    if side_field not in ("w", "b"):
        raise BadPositionAdapterError(
            f"Unsupported position format: FEN side-to-move must be 'w' or 'b', "
            f"got {side_field!r}. Input: {fen!r}"
        )


# ── Reconciliation policy ───────────────────────────────────────────────────

class ReconciliationPolicy(str, Enum):
    """How to handle disagreement between OpenSpiel moves and platform moves.

    The platform's LegalMoveSet is always authoritative — no policy may
    return a move outside it.

    Disagreement occurs when OpenSpiel move strings and platform move strings
    share no common elements. Common causes:
      - Promotion notation divergence (e.g. "e7e8q" vs "e7e8=Q")
      - Castling notation differences between platform and OpenSpiel versions
      - The platform request contains a restricted subset of board-legal moves
    """

    INTERSECT_THEN_PLATFORM_FALLBACK = "intersect_then_platform_fallback"
    """Default. Return the first OpenSpiel move that is also in the platform
    set (intersection). If the intersection is empty, fall back to the first
    move in the platform-supplied LegalMoveSet.

    Guarantees: always returns a move from the platform legal set. Never raises.
    Suitable for the prototype while notation alignment is being hardened."""

    INTERSECT_THEN_FAIL = "intersect_then_fail"
    """Return the first OpenSpiel move in the intersection. If the intersection
    is empty, raise OpenSpielAdapterError.

    Use this to surface notation mismatches immediately rather than silently
    falling back. Appropriate when notation alignment is confirmed and any
    divergence indicates a bug."""

    STRICT_FAIL = "strict_fail"
    """Fail if any OpenSpiel move is absent from the platform legal set.

    The most conservative policy — intended for notation alignment
    verification in controlled test scenarios. In practice, OpenSpiel may
    generate moves not present in the platform request (e.g. the request is
    a subset of board-legal moves), so this policy is primarily for testing."""


@dataclass(frozen=True)
class MoveReconciliationResult:
    """Outcome of a reconciliation pass between OpenSpiel and platform moves."""

    selected_move: Move
    policy: ReconciliationPolicy
    fallback_used: bool
    """True when the intersection was empty and the policy-defined fallback
    path was taken. Only possible with INTERSECT_THEN_PLATFORM_FALLBACK."""


# ── Core reconciliation logic ───────────────────────────────────────────────

def reconcile_moves(
    openspiel_moves: list[str],
    platform_moves: LegalMoveSet,
    policy: ReconciliationPolicy = ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK,
) -> MoveReconciliationResult:
    """Reconcile OpenSpiel legal moves against the platform-authoritative set.

    Arguments:
        openspiel_moves: UCI move strings from OpenSpiel action_to_string().
        platform_moves:  The authoritative LegalMoveSet from the platform request.
        policy:          How to handle the case where no intersection exists.

    Returns:
        MoveReconciliationResult with the selected move and reconciliation metadata.

    Raises:
        OpenSpielAdapterError: if INTERSECT_THEN_FAIL or STRICT_FAIL and the
            invariant is violated.

    Invariant: selected_move is always a member of platform_moves.
    """
    platform_values = {m.value for m in platform_moves.moves}
    intersection = [m for m in openspiel_moves if m in platform_values]

    if policy == ReconciliationPolicy.STRICT_FAIL:
        extra = [m for m in openspiel_moves if m not in platform_values]
        if extra:
            raise OpenSpielAdapterError(
                f"OpenSpiel returned {len(extra)} move(s) outside the platform "
                f"legal set under {policy.value!r} policy. "
                f"Unexpected moves: {extra!r}. "
                f"Platform set: {sorted(platform_values)!r}"
            )
        if not intersection:
            raise OpenSpielAdapterError(
                f"No legal moves available under {policy.value!r} policy: "
                "OpenSpiel returned no moves and platform set is non-empty "
                "but disjoint."
            )
        return MoveReconciliationResult(
            selected_move=Move(intersection[0]),
            policy=policy,
            fallback_used=False,
        )

    if intersection:
        return MoveReconciliationResult(
            selected_move=Move(intersection[0]),
            policy=policy,
            fallback_used=False,
        )

    # Empty intersection — apply policy
    if policy == ReconciliationPolicy.INTERSECT_THEN_FAIL:
        raise OpenSpielAdapterError(
            f"No OpenSpiel move matches any platform legal move "
            f"under {policy.value!r} policy. "
            f"OpenSpiel moves: {openspiel_moves!r}. "
            f"Platform moves: {sorted(platform_values)!r}"
        )

    # INTERSECT_THEN_PLATFORM_FALLBACK
    return MoveReconciliationResult(
        selected_move=platform_moves.moves[0],
        policy=policy,
        fallback_used=True,
    )
