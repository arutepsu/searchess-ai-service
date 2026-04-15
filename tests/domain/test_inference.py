import pytest

from searchess_ai.domain.game import Move
from searchess_ai.domain.inference import DecisionType, InferenceDecision
from searchess_ai.domain.model import ModelId, ModelVersion


def _make_decision(confidence: float | None) -> InferenceDecision:
    return InferenceDecision(
        request_id="req-1",
        decision_type=DecisionType.MOVE,
        selected_move=Move("e2e4"),
        model_id=ModelId("stub-model"),
        model_version=ModelVersion("0.1.0"),
        decision_time_millis=1,
        confidence=confidence,
    )


def test_inference_decision_rejects_confidence_above_one() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _make_decision(1.1)


def test_inference_decision_rejects_confidence_below_zero() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _make_decision(-0.1)


def test_inference_decision_accepts_boundary_confidence_values() -> None:
    assert _make_decision(0.0).confidence == 0.0
    assert _make_decision(1.0).confidence == 1.0


def test_inference_decision_accepts_none_confidence() -> None:
    assert _make_decision(None).confidence is None
