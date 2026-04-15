from __future__ import annotations

from pydantic import BaseModel, Field


class MoveInferenceRequestDto(BaseModel):
    request_id: str = Field(..., min_length=1)
    match_id: str = Field(..., min_length=1)
    board_state: str = Field(..., min_length=1)
    side_to_move: str = Field(..., pattern="^(white|black)$")
    legal_moves: list[str] = Field(..., min_length=1)
    policy_profile: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    remaining_time_millis: int | None = Field(default=None, ge=0)


class MoveInferenceResponseDto(BaseModel):
    request_id: str
    decision_type: str
    selected_move: str
    model_id: str
    model_version: str
    decision_time_millis: int
    policy_profile: str | None = None
    confidence: float | None = None