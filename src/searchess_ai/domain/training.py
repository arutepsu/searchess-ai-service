from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from searchess_ai.domain.model import ModelId, ModelVersion


class TrainingProfile(str, Enum):
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    MIXED = "mixed"


class TrainingJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TrainingJobId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("TrainingJobId value must not be empty.")


@dataclass(frozen=True, slots=True)
class TrainingJobRequest:
    training_profile: TrainingProfile
    base_model_id: ModelId | None = None
    base_model_version: ModelVersion | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingJobAccepted:
    training_job_id: TrainingJobId
    training_profile: TrainingProfile
    status: TrainingJobStatus
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class TrainingJob:
    training_job_id: TrainingJobId
    training_profile: TrainingProfile
    status: TrainingJobStatus
    submitted_at: datetime
    updated_at: datetime
    base_model_id: ModelId | None = None
    base_model_version: ModelVersion | None = None
    produced_model_id: ModelId | None = None
    produced_model_version: ModelVersion | None = None
    notes: str | None = None
    failure_reason: str | None = None


class TrainingJobNotFoundError(Exception):
    def __init__(self, training_job_id: str) -> None:
        super().__init__(f"Training job not found: {training_job_id}")
        self.training_job_id = training_job_id
