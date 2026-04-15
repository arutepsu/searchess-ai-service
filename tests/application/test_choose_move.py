import random

import pytest

from searchess_ai.application.port.inference_engine import InferenceEngine
from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.domain.game import LegalMoveSet, Move, Position, SideToMove
from searchess_ai.domain.inference import DecisionType, InferenceDecision, InferenceRequest
from searchess_ai.domain.model import ModelId, ModelVersion
from searchess_ai.infrastructure.inference.random_inference_engine import RandomInferenceEngine


def _make_request() -> InferenceRequest:
    return InferenceRequest(
        request_id="req-1",
        match_id="match-1",
        position=Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        side_to_move=SideToMove.WHITE,
        legal_moves=LegalMoveSet((Move("e2e4"), Move("d2d4"))),
    )


def _make_decision(request: InferenceRequest, move: Move) -> InferenceDecision:
    return InferenceDecision(
        request_id=request.request_id,
        decision_type=DecisionType.MOVE,
        selected_move=move,
        model_id=ModelId("stub"),
        model_version=ModelVersion("0.1"),
        decision_time_millis=1,
    )


class _ValidEngine(InferenceEngine):
    def choose_move(self, request: InferenceRequest) -> InferenceDecision:
        return _make_decision(request, request.legal_moves.moves[0])


class _IllegalMoveEngine(InferenceEngine):
    def choose_move(self, request: InferenceRequest) -> InferenceDecision:
        return _make_decision(request, Move("a1a8"))  # not in legal_moves


def test_use_case_returns_decision_for_valid_move() -> None:
    use_case = ChooseMoveUseCase(inference_engine=_ValidEngine())
    decision = use_case.execute(_make_request())
    assert decision.selected_move == Move("e2e4")
    assert decision.request_id == "req-1"


def test_use_case_raises_for_illegal_move() -> None:
    use_case = ChooseMoveUseCase(inference_engine=_IllegalMoveEngine())
    with pytest.raises(ValueError, match="not legal"):
        use_case.execute(_make_request())


# --- RandomInferenceEngine through ChooseMoveUseCase ---

def test_use_case_works_with_random_engine() -> None:
    use_case = ChooseMoveUseCase(inference_engine=RandomInferenceEngine(rng=random.Random(0)))
    decision = use_case.execute(_make_request())
    assert decision.selected_move in [Move("e2e4"), Move("d2d4")]
    assert decision.request_id == "req-1"


def test_use_case_legality_check_still_enforced_regardless_of_backend() -> None:
    """The use case's legality guard is backend-agnostic.
    A badly behaved RandomInferenceEngine subclass still gets caught.
    """
    class _BadRandomEngine(RandomInferenceEngine):
        def choose_move(self, request: InferenceRequest) -> InferenceDecision:
            return _make_decision(request, Move("a1a8"))  # never in legal_moves

    use_case = ChooseMoveUseCase(inference_engine=_BadRandomEngine())
    with pytest.raises(ValueError, match="not legal"):
        use_case.execute(_make_request())
