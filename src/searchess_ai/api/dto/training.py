from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TrainingJobRequestDto(BaseModel):
    training_profile: str = Field(..., pattern="^(supervised|reinforcement|mixed)$")
    base_model_id: str | None = None
    base_model_version: str | None = None
    notes: str | None = None


class TrainingJobAcceptedDto(BaseModel):
    training_job_id: str
    training_profile: str
    status: str
    submitted_at: datetime


class TrainingJobStatusDto(BaseModel):
    training_job_id: str
    training_profile: str
    status: str
    submitted_at: datetime
    updated_at: datetime
    base_model_id: str | None = None
    base_model_version: str | None = None
    produced_model_id: str | None = None
    produced_model_version: str | None = None
    notes: str | None = None
    failure_reason: str | None = None
