from __future__ import annotations

from abc import ABC, abstractmethod

from searchess_ai.domain.inference import InferenceDecision, InferenceRequest


class InferenceEngine(ABC):
    """
    Application port for AI inference.

    This defines how the application layer requests a move decision,
    without knowing anything about:
    - OpenSpiel
    - model implementation
    - training framework
    """

    @abstractmethod
    def choose_move(self, request: InferenceRequest) -> InferenceDecision:
        """
        Select a move for the given request.

        Must:
        - return a move from request.legal_moves
        - include model_id and model_version
        - be deterministic or stochastic depending on policy
        """
        pass