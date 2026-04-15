from __future__ import annotations

from searchess_ai.infrastructure.inference._openspiel_decision import (
    HeuristicDecisionResult,
    HeuristicProfile,
    OpenSpielMoveCandidate,
    build_heuristic_decision,
    heuristic_confidence,
    is_promotion_move,
    ordered_move_strings,
    rank_candidates,
    score_candidate,
)


class TestIsPromotionMove:
    def test_detects_queen_promotion(self) -> None:
        assert is_promotion_move("e7e8q")

    def test_detects_rook_promotion(self) -> None:
        assert is_promotion_move("e7e8r")

    def test_detects_bishop_promotion(self) -> None:
        assert is_promotion_move("e7e8b")

    def test_detects_knight_promotion(self) -> None:
        assert is_promotion_move("e7e8n")

    def test_returns_false_for_normal_move(self) -> None:
        assert not is_promotion_move("e2e4")

    def test_returns_false_for_longer_non_uci_string(self) -> None:
        assert not is_promotion_move("e7e8=Q")


class TestScoreCandidate:
    def test_promotion_scores_higher_than_capture(self) -> None:
        promotion = OpenSpielMoveCandidate(move="e7e8q", is_promotion=True)
        capture = OpenSpielMoveCandidate(move="d4e5", is_capture=True)
        assert score_candidate(promotion) > score_candidate(capture)

    def test_capture_scores_higher_than_quiet(self) -> None:
        capture = OpenSpielMoveCandidate(move="d4e5", is_capture=True)
        quiet = OpenSpielMoveCandidate(move="e2e4")
        assert score_candidate(capture) > score_candidate(quiet)


class TestRankCandidates:
    def test_promotion_before_capture_before_quiet(self) -> None:
        quiet = OpenSpielMoveCandidate(move="e2e4")
        capture = OpenSpielMoveCandidate(move="d4e5", is_capture=True)
        promotion = OpenSpielMoveCandidate(move="e7e8q", is_promotion=True)

        ranked = rank_candidates([quiet, capture, promotion])

        assert [c.move for c in ranked] == ["e7e8q", "d4e5", "e2e4"]

    def test_preserves_original_order_among_equal_scores(self) -> None:
        a = OpenSpielMoveCandidate(move="e2e4")
        b = OpenSpielMoveCandidate(move="d2d4")
        c = OpenSpielMoveCandidate(move="g1f3")

        ranked = rank_candidates([a, b, c])

        assert [candidate.move for candidate in ranked] == ["e2e4", "d2d4", "g1f3"]

    def test_two_captures_preserve_original_order(self) -> None:
        a = OpenSpielMoveCandidate(move="d4e5", is_capture=True)
        b = OpenSpielMoveCandidate(move="c4d5", is_capture=True)

        ranked = rank_candidates([a, b])

        assert [candidate.move for candidate in ranked] == ["d4e5", "c4d5"]


class TestOrderedMoveStrings:
    def test_returns_ranked_move_strings(self) -> None:
        candidates = [
            OpenSpielMoveCandidate(move="e2e4"),
            OpenSpielMoveCandidate(move="d4e5", is_capture=True),
            OpenSpielMoveCandidate(move="e7e8q", is_promotion=True),
        ]
        assert ordered_move_strings(candidates) == ["e7e8q", "d4e5", "e2e4"]

class TestBuildHeuristicDecision:
    def test_returns_empty_result_for_no_candidates(self) -> None:
        result = build_heuristic_decision([])
        assert result == HeuristicDecisionResult(
            ranked_moves=[],
            top_move=None,
            top_score=None,
            selected_reason=None,
        )

    def test_returns_top_move_and_reason_for_promotion(self) -> None:
        result = build_heuristic_decision([
            OpenSpielMoveCandidate(move="e2e4"),
            OpenSpielMoveCandidate(move="e7e8q", is_promotion=True),
        ])
        assert result.top_move == "e7e8q"
        assert result.top_score == 100
        assert result.selected_reason == "promotion"
        assert result.ranked_moves == ["e7e8q", "e2e4"]

    def test_returns_top_move_and_reason_for_capture(self) -> None:
        result = build_heuristic_decision([
            OpenSpielMoveCandidate(move="e2e4"),
            OpenSpielMoveCandidate(move="d4e5", is_capture=True),
        ])
        assert result.top_move == "d4e5"
        assert result.top_score == 10
        assert result.selected_reason == "capture"

    def test_returns_quiet_reason_when_no_promotion_or_capture(self) -> None:
        result = build_heuristic_decision([
            OpenSpielMoveCandidate(move="e2e4"),
            OpenSpielMoveCandidate(move="d2d4"),
        ])
        assert result.top_move == "e2e4"
        assert result.top_score == 0
        assert result.selected_reason == "quiet"


class TestHeuristicConfidence:
    def test_returns_none_when_no_top_move(self) -> None:
        decision = HeuristicDecisionResult(
            ranked_moves=[],
            top_move=None,
            top_score=None,
            selected_reason=None,
        )
        assert heuristic_confidence(decision, fallback_used=False) is None

    def test_returns_low_confidence_on_fallback(self) -> None:
        decision = HeuristicDecisionResult(
            ranked_moves=["e7e8q"],
            top_move="e7e8q",
            top_score=100,
            selected_reason="promotion",
        )
        assert heuristic_confidence(decision, fallback_used=True) == 0.2

    def test_returns_high_confidence_for_promotion(self) -> None:
        decision = HeuristicDecisionResult(
            ranked_moves=["e7e8q"],
            top_move="e7e8q",
            top_score=100,
            selected_reason="promotion",
        )
        assert heuristic_confidence(decision, fallback_used=False) == 0.9

    def test_returns_medium_confidence_for_capture(self) -> None:
        decision = HeuristicDecisionResult(
            ranked_moves=["d4e5"],
            top_move="d4e5",
            top_score=10,
            selected_reason="capture",
        )
        assert heuristic_confidence(decision, fallback_used=False) == 0.7

    def test_returns_base_confidence_for_quiet_move(self) -> None:
        decision = HeuristicDecisionResult(
            ranked_moves=["e2e4"],
            top_move="e2e4",
            top_score=0,
            selected_reason="quiet",
        )
        assert heuristic_confidence(decision, fallback_used=False) == 0.5
    def test_standard_prefers_capture_over_quiet(self) -> None:
        result = build_heuristic_decision(
            [
                OpenSpielMoveCandidate(move="e2e4"),
                OpenSpielMoveCandidate(move="d4e5", is_capture=True),
            ],
            HeuristicProfile.STANDARD,
        )
        assert result.ranked_moves == ["d4e5", "e2e4"]
        assert result.selected_reason == "capture"

    def test_promotion_only_prefers_promotion_but_not_capture(self) -> None:
        result = build_heuristic_decision(
            [
                OpenSpielMoveCandidate(move="d4e5", is_capture=True),
                OpenSpielMoveCandidate(move="e7e8q", is_promotion=True),
                OpenSpielMoveCandidate(move="e2e4"),
            ],
            HeuristicProfile.PROMOTION_ONLY,
        )
        assert result.ranked_moves == ["e7e8q", "d4e5", "e2e4"]
        assert result.selected_reason == "promotion"

    def test_promotion_only_preserves_order_for_non_promotions(self) -> None:
        result = build_heuristic_decision(
            [
                OpenSpielMoveCandidate(move="e2e4"),
                OpenSpielMoveCandidate(move="d4e5", is_capture=True),
                OpenSpielMoveCandidate(move="g1f3"),
            ],
            HeuristicProfile.PROMOTION_ONLY,
        )
        assert result.ranked_moves == ["e2e4", "d4e5", "g1f3"]

    def test_preserve_order_keeps_original_order(self) -> None:
        result = build_heuristic_decision(
            [
                OpenSpielMoveCandidate(move="e2e4"),
                OpenSpielMoveCandidate(move="d4e5", is_capture=True),
                OpenSpielMoveCandidate(move="e7e8q", is_promotion=True),
            ],
            HeuristicProfile.PRESERVE_ORDER,
        )
        assert result.ranked_moves == ["e2e4", "d4e5", "e7e8q"]
        assert result.selected_reason == "preserve_order"
        assert result.top_score == 0