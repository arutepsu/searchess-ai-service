from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from searchess_ai.domain.game import LegalMoveSet, Move, Position, SideToMove
from searchess_ai.domain.model import ModelId, ModelVersion, PolicyProfile


class DecisionType(str, Enum):
    MOVE = "move"


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: str
    match_id: str
    position: Position
    side_to_move: SideToMove
    legal_moves: LegalMoveSet
    policy_profile: PolicyProfile | None = None
    model_id: ModelId | None = None
    model_version: ModelVersion | None = None
    remaining_time_millis: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must not be empty.")
        if not self.match_id or not self.match_id.strip():
            raise ValueError("match_id must not be empty.")
        if self.remaining_time_millis is not None and self.remaining_time_millis < 0:
            raise ValueError("remaining_time_millis must be >= 0.")


@dataclass(frozen=True, slots=True)
class InferenceDecision:
    request_id: str
    decision_type: DecisionType
    selected_move: Move
    model_id: ModelId
    model_version: ModelVersion
    decision_time_millis: int
    policy_profile: PolicyProfile | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.request_id.strip():
            raise ValueError("request_id must not be empty.")
        if self.decision_time_millis < 0:
            raise ValueError("decision_time_millis must be >= 0.")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0.")