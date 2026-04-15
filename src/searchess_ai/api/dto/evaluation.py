from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EvaluationJobRequestDto(BaseModel):
    candidate_model_id: str = Field(..., min_length=1)
    candidate_model_version: str = Field(..., min_length=1)
    baseline_model_id: str = Field(..., min_length=1)
    baseline_model_version: str = Field(..., min_length=1)
    number_of_games: int = Field(..., ge=1)
    notes: str | None = None


class EvaluationJobAcceptedDto(BaseModel):
    evaluation_job_id: str
    candidate_model_id: str
    candidate_model_version: str
    baseline_model_id: str
    baseline_model_version: str
    number_of_games: int
    status: str
    submitted_at: datetime


class ScoreSummaryDto(BaseModel):
    candidate_score: float
    baseline_score: float


class EvaluationResultDto(BaseModel):
    games_played: int
    candidate_wins: int
    baseline_wins: int
    draws: int
    score_summary: ScoreSummaryDto


class EvaluationJobStatusDto(BaseModel):
    evaluation_job_id: str
    candidate_model_id: str
    candidate_model_version: str
    baseline_model_id: str
    baseline_model_version: str
    number_of_games: int
    status: str
    submitted_at: datetime
    updated_at: datetime
    result: EvaluationResultDto | None = None
    notes: str | None = None
    failure_reason: str | None = None
