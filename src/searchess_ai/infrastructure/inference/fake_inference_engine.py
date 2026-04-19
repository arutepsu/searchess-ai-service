from __future__ import annotations

from dataclasses import dataclass

from searchess_ai.application.port.inference_engine import InferenceEngine
from searchess_ai.domain.inference import (
    DecisionType,
    InferenceDecision,
    InferenceRequest,
)
from searchess_ai.domain.model import ModelId, ModelVersion


@dataclass(slots=True)
class FakeInferenceEngine(InferenceEngine):
    default_model_id: ModelId = ModelId("stub-model")
    default_model_version: ModelVersion = ModelVersion("0.1.0")

    def choose_move(self, request: InferenceRequest) -> InferenceDecision:
        selected_move = request.legal_moves.moves[0]

        return InferenceDecision(
            request_id=request.request_id,
            decision_type=DecisionType.MOVE,
            selected_move=selected_move,
            model_id=request.model_id or self.default_model_id,
            model_version=request.model_version or self.default_model_version,
            decision_time_millis=1,
            policy_profile=request.policy_profile,
            confidence=1.0,
        )