from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenSpielMoveCandidate:
    """
    Lightweight infrastructure-only candidate model for Phase G.

    This is intentionally not a domain model:
    - it depends on OpenSpiel-facing move extraction
    - it exists only to support internal move ordering
    """
    move: str
    is_capture: bool = False
    is_promotion: bool = False


def is_promotion_move(move: str) -> bool:
    """
    Detect UCI promotion suffixes like:
      e7e8q, e7e8r, e7e8b, e7e8n

    This is intentionally simple and notation-local.
    """
    normalized = move.strip().lower()
    return len(normalized) == 5 and normalized[-1] in {"q", "r", "b", "n"}


def score_candidate(candidate: OpenSpielMoveCandidate) -> int:
    """
    Phase G heuristic:
      promotion > capture > quiet

    Higher score is better.
    """
    score = 0
    if candidate.is_promotion:
        score += 100
    if candidate.is_capture:
        score += 10
    return score


def rank_candidates(
    candidates: list[OpenSpielMoveCandidate],
) -> list[OpenSpielMoveCandidate]:
    """
    Return candidates ordered best-first.

    Stable sort preserves original OpenSpiel order among equal-scored moves,
    which keeps behavior deterministic.
    """
    return sorted(candidates, key=score_candidate, reverse=True)


def ordered_move_strings(candidates: list[OpenSpielMoveCandidate]) -> list[str]:
    """
    Convenience helper for feeding ranked moves into reconcile_moves().
    """
    return [candidate.move for candidate in rank_candidates(candidates)]