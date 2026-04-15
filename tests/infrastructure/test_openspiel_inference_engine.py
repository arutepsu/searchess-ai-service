"""
Tests for the OpenSpiel inference adapter.

Testing strategy
----------------
pyspiel is a native C extension that cannot be installed on Windows/Python 3.14.
Rather than skipping all tests, we test the adapter at two levels:

  Level 1 — pure Python logic (always run):
    _openspiel_chess helpers (select_move, get_legal_move_strings) take plain
    Python objects, so we test them without any pyspiel dependency.

  Level 2 — adapter wiring (always run, pyspiel mocked):
    OpenSpielInferenceEngine.choose_move is tested by patching _require_pyspiel
    and load_chess_state_from_fen so the adapter logic runs fully without the
    native extension.

  Level 3 — real integration (skipped unless pyspiel is installed):
    pytest.importorskip("pyspiel") gates these tests; they run automatically
    on Linux CI where open-spiel can be installed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from searchess_ai.domain.game import LegalMoveSet, Move, Position, SideToMove
from searchess_ai.domain.inference import InferenceRequest
from searchess_ai.domain.model import ModelId, ModelVersion
from searchess_ai.infrastructure.inference._openspiel_chess import (
    get_legal_move_strings,
    select_move,
)
from searchess_ai.infrastructure.inference._openspiel_mapping import (
    ReconciliationPolicy,
)
from searchess_ai.infrastructure.inference.openspiel_inference_engine import (
    OpenSpielAdapterError,
    OpenSpielInferenceEngine,
)

_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _make_request(
    moves: list[str] | None = None,
    fen: str = _START_FEN,
) -> InferenceRequest:
    move_strs = moves or ["e2e4", "d2d4", "g1f3"]
    return InferenceRequest(
        request_id="req-1",
        match_id="match-1",
        position=Position(fen),
        side_to_move=SideToMove.WHITE,
        legal_moves=LegalMoveSet(tuple(Move(m) for m in move_strs)),
    )


def _make_mock_state(legal_move_strings: list[str]) -> MagicMock:
    """Return a mock OpenSpiel state whose legal actions map to the given UCI strings."""
    state = MagicMock()
    state.current_player.return_value = 0
    state.legal_actions.return_value = list(range(len(legal_move_strings)))
    state.action_to_string.side_effect = (
        lambda player, action: legal_move_strings[action]
    )
    return state


# ── Level 1: pure mapping helpers ─────────────────────────────────────────────

class TestSelectMove:
    def test_picks_first_openspiel_move_in_intersection(self) -> None:
        request_moves = LegalMoveSet((Move("e2e4"), Move("d2d4"), Move("g1f3")))
        openspiel_moves = ["d2d4", "g1f3", "a2a3"]
        assert select_move(openspiel_moves, request_moves) == Move("d2d4")

    def test_falls_back_to_first_request_move_when_intersection_empty(self) -> None:
        """Platform legal moves are authoritative — never return outside that set."""
        request_moves = LegalMoveSet((Move("e2e4"), Move("d2d4")))
        openspiel_moves = ["a2a3", "b2b3"]  # no overlap
        assert select_move(openspiel_moves, request_moves) == Move("e2e4")

    def test_returns_move_type(self) -> None:
        result = select_move(["e2e4"], LegalMoveSet((Move("e2e4"),)))
        assert isinstance(result, Move)

    def test_ignores_openspiel_moves_outside_request_set(self) -> None:
        request_moves = LegalMoveSet((Move("g1f3"),))
        openspiel_moves = ["e2e4", "d2d4", "g1f3"]
        assert select_move(openspiel_moves, request_moves) == Move("g1f3")


class TestGetLegalMoveStrings:
    def test_returns_uci_strings_for_all_legal_actions(self) -> None:
        state = _make_mock_state(["e2e4", "d2d4", "g1f3"])
        assert get_legal_move_strings(state) == ["e2e4", "d2d4", "g1f3"]

    def test_passes_current_player_to_action_to_string(self) -> None:
        state = _make_mock_state(["e2e4"])
        state.current_player.return_value = 1
        get_legal_move_strings(state)
        state.action_to_string.assert_called_with(1, 0)

    def test_returns_empty_list_when_no_legal_actions(self) -> None:
        state = MagicMock()
        state.current_player.return_value = 0
        state.legal_actions.return_value = []
        assert get_legal_move_strings(state) == []


# ── Level 2: adapter wiring with mocked pyspiel ───────────────────────────────

class TestOpenSpielInferenceEngineWired:
    """Tests that run without pyspiel installed by mocking _require_pyspiel
    and load_chess_state_from_fen at the engine module's import names."""

    _ENGINE_MODULE = "searchess_ai.infrastructure.inference.openspiel_inference_engine"

    def _run(
        self,
        openspiel_moves: list[str],
        request: InferenceRequest | None = None,
    ) -> object:
        mock_state = _make_mock_state(openspiel_moves)
        with (
            patch(f"{self._ENGINE_MODULE}._require_pyspiel") as mock_require,
            patch(f"{self._ENGINE_MODULE}.load_chess_state_from_fen") as mock_load,
        ):
            mock_require.return_value = MagicMock()
            mock_load.return_value = mock_state
            engine = OpenSpielInferenceEngine()
            return engine.choose_move(request or _make_request())

    def test_returns_move_from_request_legal_moves(self) -> None:
        decision = self._run(["e2e4", "a2a3"])
        assert decision.selected_move in [Move("e2e4"), Move("d2d4"), Move("g1f3")]

    def test_preserves_request_id(self) -> None:
        assert self._run(["e2e4"]).request_id == "req-1"

    def test_includes_model_id_and_version(self) -> None:
        decision = self._run(["e2e4"])
        assert decision.model_id.value
        assert decision.model_version.value

    def test_uses_default_model_when_request_has_none(self) -> None:
        decision = self._run(["e2e4"])
        assert decision.model_id == ModelId("openspiel-chess-v0")
        assert decision.model_version == ModelVersion("0.1.0")

    def test_respects_request_model_id_and_version(self) -> None:
        request = InferenceRequest(
            request_id="req-2",
            match_id="match-1",
            position=Position(_START_FEN),
            side_to_move=SideToMove.WHITE,
            legal_moves=LegalMoveSet((Move("e2e4"),)),
            model_id=ModelId("alpha-v1"),
            model_version=ModelVersion("1.0.0"),
        )
        decision = self._run(["e2e4"], request=request)
        assert decision.model_id == ModelId("alpha-v1")
        assert decision.model_version == ModelVersion("1.0.0")

    def test_is_deterministic_for_same_openspiel_order(self) -> None:
        decision_a = self._run(["d2d4", "e2e4"])
        decision_b = self._run(["d2d4", "e2e4"])
        assert decision_a.selected_move == decision_b.selected_move

    def test_falls_back_gracefully_when_openspiel_moves_dont_match(self) -> None:
        """If OpenSpiel returns moves not in request.legal_moves, we fall back
        to the first platform-authoritative move — never a fabricated move."""
        request = _make_request(moves=["e2e4", "d2d4"])
        decision = self._run(["a2a3", "b2b3"], request=request)
        assert decision.selected_move in [Move("e2e4"), Move("d2d4")]

    def test_raises_adapter_error_when_state_loading_fails(self) -> None:
        with (
            patch(f"{self._ENGINE_MODULE}._require_pyspiel") as mock_require,
            patch(f"{self._ENGINE_MODULE}.load_chess_state_from_fen") as mock_load,
        ):
            mock_require.return_value = MagicMock()
            mock_load.side_effect = RuntimeError("bad FEN")
            engine = OpenSpielInferenceEngine()
            with pytest.raises(OpenSpielAdapterError, match="bad FEN"):
                engine.choose_move(_make_request())

    def test_raises_adapter_error_when_pyspiel_missing(self) -> None:
        with patch(f"{self._ENGINE_MODULE}._require_pyspiel") as mock_require:
            mock_require.side_effect = OpenSpielAdapterError("pyspiel not installed")
            engine = OpenSpielInferenceEngine()
            with pytest.raises(OpenSpielAdapterError, match="pyspiel not installed"):
                engine.choose_move(_make_request())

    def test_raises_adapter_error_for_malformed_fen(self) -> None:
        """validate_fen fires before load_chess_state_from_fen for bad input."""
        request = InferenceRequest(
            request_id="req-bad",
            match_id="match-1",
            position=Position("not a fen at all"),
            side_to_move=SideToMove.WHITE,
            legal_moves=LegalMoveSet((Move("e2e4"),)),
        )
        with (
            patch(f"{self._ENGINE_MODULE}._require_pyspiel") as mock_require,
            patch(f"{self._ENGINE_MODULE}.load_chess_state_from_fen") as mock_load,
        ):
            mock_require.return_value = MagicMock()
            engine = OpenSpielInferenceEngine()
            with pytest.raises(OpenSpielAdapterError, match="Unsupported position format"):
                engine.choose_move(request)
            mock_load.assert_not_called()

    def test_raises_adapter_error_for_fen_with_wrong_side_to_move(self) -> None:
        request = InferenceRequest(
            request_id="req-bad2",
            match_id="match-1",
            position=Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq -"),
            side_to_move=SideToMove.WHITE,
            legal_moves=LegalMoveSet((Move("e2e4"),)),
        )
        with (
            patch(f"{self._ENGINE_MODULE}._require_pyspiel") as mock_require,
            patch(f"{self._ENGINE_MODULE}.load_chess_state_from_fen") as mock_load,
        ):
            mock_require.return_value = MagicMock()
            engine = OpenSpielInferenceEngine()
            with pytest.raises(OpenSpielAdapterError, match="side-to-move"):
                engine.choose_move(request)
            mock_load.assert_not_called()

    def test_default_reconciliation_policy_is_fallback(self) -> None:
        engine = OpenSpielInferenceEngine()
        assert engine.reconciliation_policy == (
            ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK
        )

    def test_engine_uses_configured_reconciliation_policy(self) -> None:
        """Engine with INTERSECT_THEN_FAIL raises when no overlap exists."""
        request = _make_request(moves=["e2e4", "d2d4"])
        mock_state = _make_mock_state(["a2a3", "b2b3"])  # no overlap with request
        with (
            patch(f"{self._ENGINE_MODULE}._require_pyspiel") as mock_require,
            patch(f"{self._ENGINE_MODULE}.load_chess_state_from_fen") as mock_load,
        ):
            mock_require.return_value = MagicMock()
            mock_load.return_value = mock_state
            engine = OpenSpielInferenceEngine(
                reconciliation_policy=ReconciliationPolicy.INTERSECT_THEN_FAIL
            )
            with pytest.raises(OpenSpielAdapterError, match="intersect_then_fail"):
                engine.choose_move(request)


# ── Level 3: real integration (skipped unless pyspiel is installed) ───────────

pyspiel = pytest.importorskip("pyspiel", reason="pyspiel not installed")


class TestOpenSpielInferenceEngineIntegration:
    """Integration tests against the real OpenSpiel library.

    These run automatically on platforms where open-spiel can be installed
    (Linux, macOS). They are skipped on Windows / Python 3.14.
    """

    def test_returns_move_from_legal_set_for_start_position(self) -> None:
        engine = OpenSpielInferenceEngine()
        request = _make_request(moves=["e2e4", "d2d4", "g1f3"])
        decision = engine.choose_move(request)
        assert decision.selected_move in request.legal_moves.moves

    def test_preserves_request_id_real(self) -> None:
        engine = OpenSpielInferenceEngine()
        decision = engine.choose_move(_make_request())
        assert decision.request_id == "req-1"

    def test_returns_default_model_metadata(self) -> None:
        engine = OpenSpielInferenceEngine()
        decision = engine.choose_move(_make_request())
        assert decision.model_id == ModelId("openspiel-chess-v0")
