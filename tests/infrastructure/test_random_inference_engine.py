import random

from searchess_ai.domain.game import LegalMoveSet, Move, Position, SideToMove
from searchess_ai.domain.inference import InferenceRequest
from searchess_ai.domain.model import ModelId, ModelVersion
from searchess_ai.infrastructure.inference.random_inference_engine import RandomInferenceEngine
from searchess_ai.infrastructure.inference._openspiel_decision import (
    HeuristicDecisionResult,
    HeuristicProfile,
)
from searchess_ai.infrastructure.inference.openspiel_inference_engine import (
    OpenSpielAdapterError,
    OpenSpielInferenceEngine,
    _build_decision_log_payload,
)

def _make_request(moves: list[str] = None) -> InferenceRequest:
    move_strs = moves or ["e2e4", "d2d4", "g1f3"]
    return InferenceRequest(
        request_id="req-1",
        match_id="match-1",
        position=Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        side_to_move=SideToMove.WHITE,
        legal_moves=LegalMoveSet(tuple(Move(m) for m in move_strs)),
    )


def test_random_engine_returns_move_from_legal_set() -> None:
    engine = RandomInferenceEngine(rng=random.Random(0))
    request = _make_request()
    decision = engine.choose_move(request)
    assert decision.selected_move in request.legal_moves.moves


def test_random_engine_preserves_request_id() -> None:
    engine = RandomInferenceEngine(rng=random.Random(0))
    decision = engine.choose_move(_make_request())
    assert decision.request_id == "req-1"


def test_random_engine_includes_model_id_and_version() -> None:
    engine = RandomInferenceEngine(rng=random.Random(0))
    decision = engine.choose_move(_make_request())
    assert decision.model_id.value
    assert decision.model_version.value


def test_random_engine_uses_default_model_when_request_has_none() -> None:
    engine = RandomInferenceEngine(rng=random.Random(0))
    decision = engine.choose_move(_make_request())
    assert decision.model_id == ModelId("random-engine")
    assert decision.model_version == ModelVersion("0.1.0")


def test_random_engine_respects_request_model_id_and_version() -> None:
    engine = RandomInferenceEngine(rng=random.Random(0))
    request = InferenceRequest(
        request_id="req-2",
        match_id="match-1",
        position=Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        side_to_move=SideToMove.WHITE,
        legal_moves=LegalMoveSet((Move("e2e4"),)),
        model_id=ModelId("alpha-v1"),
        model_version=ModelVersion("1.0.0"),
    )
    decision = engine.choose_move(request)
    assert decision.model_id == ModelId("alpha-v1")
    assert decision.model_version == ModelVersion("1.0.0")


def test_random_engine_is_deterministic_when_seeded() -> None:
    request = _make_request()
    decision_a = RandomInferenceEngine(rng=random.Random(42)).choose_move(request)
    decision_b = RandomInferenceEngine(rng=random.Random(42)).choose_move(request)
    assert decision_a.selected_move == decision_b.selected_move


def test_random_engine_can_produce_different_moves_across_seeds() -> None:
    """Sanity-check that the engine is not always returning the same move."""
    request = _make_request(["a2a3", "b2b3", "c2c3", "d2d4", "e2e4", "f2f3", "g2g3", "h2h3"])
    moves_seen = {
        RandomInferenceEngine(rng=random.Random(seed)).choose_move(request).selected_move
        for seed in range(20)
    }
    assert len(moves_seen) > 1

    def test_prefers_promotion_over_quiet_move(self) -> None:
        request = _make_request(moves=["e7e8q", "e2e4"])
        decision = self._run(["e2e4", "e7e8q"], request=request)
        assert decision.selected_move == Move("e7e8q")

    def test_prefers_capture_over_quiet_move_when_state_exposes_capture_info(self) -> None:
        class CustomState:
            def current_player(self) -> int:
                return 0

            def legal_actions(self) -> list[int]:
                return [0, 1]

            def action_to_string(self, player: int, action: int) -> str:
                return ["e2e4", "d4e5"][action]

            def is_capture(self, player: int, action: int) -> bool:
                return action == 1

        request = _make_request(moves=["e2e4", "d4e5"])

        with (
            patch(f"{self._ENGINE_MODULE}._require_pyspiel") as mock_require,
            patch(f"{self._ENGINE_MODULE}.load_chess_state_from_fen") as mock_load,
        ):
            mock_require.return_value = MagicMock()
            mock_load.return_value = CustomState()
            engine = OpenSpielInferenceEngine()
            decision = engine.choose_move(request)

        assert decision.selected_move == Move("d4e5")

    def test_preserves_old_behavior_for_equal_score_moves(self) -> None:
        request = _make_request(moves=["d2d4", "e2e4"])
        decision = self._run(["d2d4", "e2e4"], request=request)
        assert decision.selected_move == Move("d2d4")

        def test_sets_high_confidence_for_promotion_choice(self) -> None:
            request = _make_request(moves=["e7e8q", "e2e4"])
            decision = self._run(["e2e4", "e7e8q"], request=request)
            assert decision.selected_move == Move("e7e8q")
            assert decision.confidence == 0.9

    def test_sets_base_confidence_for_quiet_choice(self) -> None:
        request = _make_request(moves=["d2d4", "e2e4"])
        decision = self._run(["d2d4", "e2e4"], request=request)
        assert decision.selected_move == Move("d2d4")
        assert decision.confidence == 0.5

    def test_sets_low_confidence_when_fallback_is_used(self) -> None:
        request = _make_request(moves=["e2e4", "d2d4"])
        decision = self._run(["a2a3", "b2b3"], request=request)
        assert decision.selected_move == Move("e2e4")
        assert decision.confidence == 0.2

    def test_sets_capture_confidence_when_capture_is_preferred(self) -> None:
        class CustomState:
            def current_player(self) -> int:
                return 0

            def legal_actions(self) -> list[int]:
                return [0, 1]

            def action_to_string(self, player: int, action: int) -> str:
                return ["e2e4", "d4e5"][action]

            def is_capture(self, player: int, action: int) -> bool:
                return action == 1

        request = _make_request(moves=["e2e4", "d4e5"])

        with (
            patch(f"{self._ENGINE_MODULE}._require_pyspiel") as mock_require,
            patch(f"{self._ENGINE_MODULE}.load_chess_state_from_fen") as mock_load,
        ):
            mock_require.return_value = MagicMock()
            mock_load.return_value = CustomState()
            engine = OpenSpielInferenceEngine()
            decision = engine.choose_move(request)

        assert decision.selected_move == Move("d4e5")
        assert decision.confidence == 0.7
        
    class TestDecisionLogPayload:
        def test_builds_structured_payload(self) -> None:
            request = _make_request(moves=["e2e4", "d2d4"])
            heuristic = HeuristicDecisionResult(
                ranked_moves=["e2e4", "d2d4"],
                top_move="e2e4",
                top_score=0,
                selected_reason="quiet",
                profile=HeuristicProfile.STANDARD,
                candidate_count=2,
            )
            reconciliation_result = MagicMock()
            reconciliation_result.selected_move = Move("e2e4")
            reconciliation_result.fallback_used = False

            payload = _build_decision_log_payload(
                request=request,
                heuristic=heuristic,
                reconciliation_result=reconciliation_result,
                confidence=0.5,
                reconciliation_policy=ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK,
            )

            assert payload["event"] == "openspiel_inference_decision"
            assert payload["request_id"] == "req-1"
            assert payload["match_id"] == "match-1"
            assert payload["heuristic_profile"] == "standard"
            assert payload["candidate_count"] == 2
            assert payload["top_move"] == "e2e4"
            assert payload["top_score"] == 0
            assert payload["selected_reason"] == "quiet"
            assert payload["selected_move"] == "e2e4"
            assert payload["fallback_used"] is False
            assert payload["confidence"] == 0.5
            assert (
                payload["reconciliation_policy"]
                == ReconciliationPolicy.INTERSECT_THEN_PLATFORM_FALLBACK.value
            )