from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HeuristicProfile(str, Enum):
    STANDARD = "standard"
    PROMOTION_ONLY = "promotion_only"
    PRESERVE_ORDER = "preserve_order"


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


@dataclass(frozen=True, slots=True)
class HeuristicDecisionResult:
    """
    Internal heuristic ranking result used by the OpenSpiel inference engine.
    """
    ranked_moves: list[str]
    top_move: str | None
    top_score: int | None
    selected_reason: str | None
    profile: HeuristicProfile
    candidate_count: int


def is_promotion_move(move: str) -> bool:
    normalized = move.strip().lower()
    return len(normalized) == 5 and normalized[-1] in {"q", "r", "b", "n"}


def score_candidate(
    candidate: OpenSpielMoveCandidate,
    profile: HeuristicProfile = HeuristicProfile.STANDARD,
) -> int:
    """
    Ranking rules by profile:

    STANDARD:
      promotion > capture > quiet

    PROMOTION_ONLY:
      promotion > all others, otherwise preserve OpenSpiel order

    PRESERVE_ORDER:
      no reordering
    """
    if profile == HeuristicProfile.PRESERVE_ORDER:
        return 0

    if profile == HeuristicProfile.PROMOTION_ONLY:
        return 100 if candidate.is_promotion else 0

    score = 0
    if candidate.is_promotion:
        score += 100
    if candidate.is_capture:
        score += 10
    return score


def rank_candidates(
    candidates: list[OpenSpielMoveCandidate],
    profile: HeuristicProfile = HeuristicProfile.STANDARD,
) -> list[OpenSpielMoveCandidate]:
    """
    Stable sort preserves original OpenSpiel order among equal-scored moves.
    """
    return sorted(
        candidates,
        key=lambda candidate: score_candidate(candidate, profile),
        reverse=True,
    )


def ordered_move_strings(
    candidates: list[OpenSpielMoveCandidate],
    profile: HeuristicProfile = HeuristicProfile.STANDARD,
) -> list[str]:
    return [candidate.move for candidate in rank_candidates(candidates, profile)]


def _reason_for_candidate(
    candidate: OpenSpielMoveCandidate,
    profile: HeuristicProfile,
) -> str:
    if profile == HeuristicProfile.PRESERVE_ORDER:
        return "preserve_order"
    if candidate.is_promotion:
        return "promotion"
    if profile == HeuristicProfile.STANDARD and candidate.is_capture:
        return "capture"
    return "quiet"


def build_heuristic_decision(
    candidates: list[OpenSpielMoveCandidate],
    profile: HeuristicProfile = HeuristicProfile.STANDARD,
) -> HeuristicDecisionResult:
    ranked_candidates = rank_candidates(candidates, profile)
    ranked_moves = [candidate.move for candidate in ranked_candidates]

    if not ranked_candidates:
        return HeuristicDecisionResult(
            ranked_moves=[],
            top_move=None,
            top_score=None,
            selected_reason=None,
            profile=profile,
            candidate_count=0,
        )

    top_candidate = ranked_candidates[0]
    return HeuristicDecisionResult(
        ranked_moves=ranked_moves,
        top_move=top_candidate.move,
        top_score=score_candidate(top_candidate, profile),
        selected_reason=_reason_for_candidate(top_candidate, profile),
        profile=profile,
        candidate_count=len(candidates),
    )


def heuristic_confidence(
    decision: HeuristicDecisionResult,
    fallback_used: bool,
) -> float | None:
    if decision.top_move is None:
        return None
    if fallback_used:
        return 0.2
    if decision.selected_reason == "promotion":
        return 0.9
    if decision.selected_reason == "capture":
        return 0.7
    if decision.selected_reason == "preserve_order":
        return 0.4
    return 0.5