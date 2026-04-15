"""
Tests for _openspiel_mapping: boundary contracts, FEN validation,
and reconciliation policy behavior.

All tests run without pyspiel installed — this module is pure Python.
"""
from __future__ import annotations

import pytest

from searchess_ai.domain.game import LegalMoveSet, Move
from searchess_ai.infrastructure.inference._openspiel_mapping import (
    MOVE_FORMAT,
    POSITION_FORMAT,
    MoveReconciliationResult,
    OpenSpielAdapterError,
    ReconciliationPolicy,
    reconcile_moves,
    validate_fen,
)

_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _moves(*values: str) -> LegalMoveSet:
    return LegalMoveSet(tuple(Move(v) for v in values))


# ── Boundary constants ─────────────────────────────────────────────────────

class TestBoundaryConstants:
    def test_position_format_is_fen(self) -> None:
        assert POSITION_FORMAT == "FEN"

    def test_move_format_is_uci(self) -> None:
        assert MOVE_FORMAT == "UCI"


# ── validate_fen ───────────────────────────────────────────────────────────

class TestValidateFen:
    def test_accepts_standard_start_position(self) -> None:
        validate_fen(_START_FEN)  # must not raise

    def test_accepts_fen_without_clock_fields(self) -> None:
        validate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -")

    def test_accepts_black_to_move(self) -> None:
        validate_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(OpenSpielAdapterError, match="empty"):
            validate_fen("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(OpenSpielAdapterError, match="empty"):
            validate_fen("   ")

    def test_rejects_single_token(self) -> None:
        with pytest.raises(OpenSpielAdapterError, match="FEN fields"):
            validate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    def test_rejects_wrong_rank_separator_count(self) -> None:
        # Only 6 rank separators instead of 7
        with pytest.raises(OpenSpielAdapterError, match="rank separators"):
            validate_fen("rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -")

    def test_rejects_invalid_side_to_move(self) -> None:
        with pytest.raises(OpenSpielAdapterError, match="side-to-move"):
            validate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq -")

    def test_rejects_plaintext_garbage(self) -> None:
        with pytest.raises(OpenSpielAdapterError):
            validate_fen("not a fen at all")

    def test_error_type_is_openspiel_adapter_error(self) -> None:
        with pytest.raises(OpenSpielAdapterError):
            validate_fen("")


# ── reconcile_moves — exact overlap ───────────────────────────────────────

class TestReconcileMovesExactOverlap:
    def test_returns_first_openspiel_move_when_all_match(self) -> None:
        result = reconcile_moves(["e2e4", "d2d4"], _moves("e2e4", "d2d4"))
        assert result.selected_move == Move("e2e4")

    def test_openspiel_order_determines_selection(self) -> None:
        # OpenSpiel returns d2d4 first; that should win
        result = reconcile_moves(["d2d4", "e2e4"], _moves("e2e4", "d2d4"))
        assert result.selected_move == Move("d2d4")

    def test_fallback_not_used_on_exact_overlap(self) -> None:
        result = reconcile_moves(["g1f3"], _moves("g1f3"))
        assert not result.fallback_used

    def test_single_move_overlap(self) -> None:
        result = reconcile_moves(["g1f3"], _moves("g1f3"))
        assert result.selected_move == Move("g1f3")


# ── reconcile_moves — partial overlap ─────────────────────────────────────

class TestReconcileMovesPartialOverlap:
    def test_ignores_openspiel_moves_outside_platform_set(self) -> None:
        result = reconcile_moves(["a2a3", "e2e4"], _moves("e2e4"))
        assert result.selected_move == Move("e2e4")
        assert not result.fallback_used

    def test_skips_openspiel_moves_until_intersection_found(self) -> None:
        result = reconcile_moves(["a2a3", "b2b3", "g1f3"], _moves("g1f3"))
        assert result.selected_move == Move("g1f3")


# ── reconcile_moves — no overlap ──────────────────────────────────────────

class TestReconcileMovesNoOverlap:
    def test_fallback_returns_first_platform_move(self) -> None:
        result = reconcile_moves(
            ["a2a3", "b2b3"],
            _moves("e2e4", "d2d4"),
            ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK,
        )
        assert result.selected_move == Move("e2e4")
        assert result.fallback_used

    def test_fallback_never_returns_openspiel_only_move(self) -> None:
        platform = _moves("g1f3")
        result = reconcile_moves(
            ["a2a3", "b2b3"],
            platform,
            ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK,
        )
        assert result.selected_move in platform.moves

    def test_empty_openspiel_list_triggers_fallback(self) -> None:
        result = reconcile_moves(
            [],
            _moves("e2e4"),
            ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK,
        )
        assert result.selected_move == Move("e2e4")
        assert result.fallback_used

    def test_intersect_then_fail_raises_on_empty_intersection(self) -> None:
        with pytest.raises(OpenSpielAdapterError, match="intersect_then_fail"):
            reconcile_moves(
                ["a2a3", "b2b3"],
                _moves("e2e4", "d2d4"),
                ReconciliationPolicy.INTERSECT_THEN_FAIL,
            )

    def test_intersect_then_fail_error_names_the_policy(self) -> None:
        with pytest.raises(OpenSpielAdapterError, match="intersect_then_fail"):
            reconcile_moves([], _moves("e2e4"), ReconciliationPolicy.INTERSECT_THEN_FAIL)


# ── reconcile_moves — STRICT_FAIL policy ──────────────────────────────────

class TestReconcileMovesStrictFail:
    def test_raises_when_openspiel_has_extra_moves(self) -> None:
        with pytest.raises(OpenSpielAdapterError, match="strict_fail"):
            reconcile_moves(
                ["e2e4", "a2a3"],  # a2a3 not in platform
                _moves("e2e4"),
                ReconciliationPolicy.STRICT_FAIL,
            )

    def test_passes_when_all_openspiel_moves_in_platform_superset(self) -> None:
        # Platform is a superset — all OpenSpiel moves are present
        result = reconcile_moves(
            ["e2e4"],
            _moves("e2e4", "d2d4"),
            ReconciliationPolicy.STRICT_FAIL,
        )
        assert result.selected_move == Move("e2e4")
        assert not result.fallback_used

    def test_error_names_the_policy(self) -> None:
        with pytest.raises(OpenSpielAdapterError, match="strict_fail"):
            reconcile_moves(
                ["e2e4", "extra"],
                _moves("e2e4"),
                ReconciliationPolicy.STRICT_FAIL,
            )


# ── Platform-authoritative invariant ──────────────────────────────────────

class TestPlatformAuthoritativeInvariant:
    """The selected move must always be a member of the platform legal set."""

    @pytest.mark.parametrize("openspiel_moves,platform_values", [
        (["e2e4", "d2d4"], ["e2e4"]),          # exact overlap
        (["a2a3"], ["e2e4", "d2d4"]),           # no overlap → fallback
        ([], ["e2e4"]),                          # empty OpenSpiel → fallback
        (["a2a3", "b2b3", "e2e4"], ["e2e4"]),  # extra OpenSpiel moves
    ])
    def test_selected_move_always_in_platform_set(
        self, openspiel_moves: list[str], platform_values: list[str]
    ) -> None:
        platform = _moves(*platform_values)
        result = reconcile_moves(
            openspiel_moves,
            platform,
            ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK,
        )
        assert result.selected_move in platform.moves


# ── Result metadata ────────────────────────────────────────────────────────

class TestMoveReconciliationResult:
    def test_result_carries_policy(self) -> None:
        result = reconcile_moves(["e2e4"], _moves("e2e4"))
        assert result.policy == ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK

    def test_fallback_false_when_intersection_exists(self) -> None:
        result = reconcile_moves(["e2e4"], _moves("e2e4"))
        assert not result.fallback_used

    def test_fallback_true_when_no_intersection(self) -> None:
        result = reconcile_moves(["a2a3"], _moves("e2e4"))
        assert result.fallback_used

    def test_result_is_a_move_instance(self) -> None:
        result = reconcile_moves(["e2e4"], _moves("e2e4"))
        assert isinstance(result.selected_move, Move)

    def test_result_is_frozen(self) -> None:
        result = reconcile_moves(["e2e4"], _moves("e2e4"))
        with pytest.raises((AttributeError, TypeError)):
            result.fallback_used = True  # type: ignore[misc]


# ── Promotion / notation mismatch scenarios ────────────────────────────────

class TestPromotionNotationMismatch:
    """Exercises the notation gap between platform and OpenSpiel for promotions."""

    def test_promotion_mismatch_triggers_fallback(self) -> None:
        """Platform uses 'e7e8=Q' style; OpenSpiel uses 'e7e8q' without '='.
        No intersection → fallback to platform_moves[0]."""
        platform = _moves("e7e8=Q", "e7e8=R")
        result = reconcile_moves(
            ["e7e8q", "e7e8r"],
            platform,
            ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK,
        )
        assert result.fallback_used
        assert result.selected_move in platform.moves

    def test_matching_promotion_notation_succeeds_without_fallback(self) -> None:
        """When both sides use the same notation, intersection exists."""
        result = reconcile_moves(
            ["e7e8q"],
            _moves("e7e8q", "e7e8r"),
        )
        assert result.selected_move == Move("e7e8q")
        assert not result.fallback_used

    def test_promotion_mismatch_with_intersect_then_fail_raises(self) -> None:
        """Notation mismatch surfaces as an error under the strict policy."""
        with pytest.raises(OpenSpielAdapterError, match="intersect_then_fail"):
            reconcile_moves(
                ["e7e8q", "e7e8r"],
                _moves("e7e8=Q", "e7e8=R"),
                ReconciliationPolicy.INTERSECT_THEN_FAIL,
            )
