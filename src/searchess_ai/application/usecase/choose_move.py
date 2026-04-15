from __future__ import annotations

from dataclasses import dataclass

from searchess_ai.application.port.inference_engine import InferenceEngine
from searchess_ai.domain.inference import InferenceDecision, InferenceRequest


@dataclass(slots=True)
class ChooseMoveUseCase:
    inference_engine: InferenceEngine

    def execute(self, request: InferenceRequest) -> InferenceDecision:
        decision = self.inference_engine.choose_move(request)

        if decision.request_id != request.request_id:
            raise ValueError("Inference decision request_id does not match request.")

        if not request.legal_moves.contains(decision.selected_move):
            raise ValueError("Inference engine returned a move that is not legal.")

        return decision