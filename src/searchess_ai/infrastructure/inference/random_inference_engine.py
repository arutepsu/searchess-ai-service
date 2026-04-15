from __future__ import annotations

import random as random_module
from dataclasses import dataclass, field

from searchess_ai.application.port.inference_engine import InferenceEngine
from searchess_ai.domain.inference import DecisionType, InferenceDecision, InferenceRequest
from searchess_ai.domain.model import ModelId, ModelVersion


@dataclass(slots=True)
class RandomInferenceEngine(InferenceEngine):
    """Selects a uniformly random move from the legal move set.

    Pass an explicit ``rng`` for deterministic behaviour in tests:
        RandomInferenceEngine(rng=random.Random(42))
    """

    rng: random_module.Random = field(default_factory=random_module.Random)
    default_model_id: ModelId = ModelId("random-engine")
    default_model_version: ModelVersion = ModelVersion("0.1.0")

    def choose_move(self, request: InferenceRequest) -> InferenceDecision:
        selected_move = self.rng.choice(list(request.legal_moves.moves))
        return InferenceDecision(
            request_id=request.request_id,
            decision_type=DecisionType.MOVE,
            selected_move=selected_move,
            model_id=request.model_id or self.default_model_id,
            model_version=request.model_version or self.default_model_version,
            decision_time_millis=0,
            policy_profile=request.policy_profile,
        )
