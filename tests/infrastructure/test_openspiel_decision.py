from __future__ import annotations

from searchess_ai.infrastructure.inference._openspiel_decision import (
    OpenSpielMoveCandidate,
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