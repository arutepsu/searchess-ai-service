from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from searchess_ai.domain.model import ModelId, ModelVersion


class EvaluationJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class EvaluationJobId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("EvaluationJobId value must not be empty.")


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    candidate_score: float
    baseline_score: float

    def __post_init__(self) -> None:
        if self.candidate_score < 0.0:
            raise ValueError("candidate_score must be >= 0.")
        if self.baseline_score < 0.0:
            raise ValueError("baseline_score must be >= 0.")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    games_played: int
    candidate_wins: int
    baseline_wins: int
    draws: int
    score_summary: ScoreSummary

    def __post_init__(self) -> None:
        if self.games_played < 0:
            raise ValueError("games_played must be >= 0.")
        if self.candidate_wins < 0 or self.baseline_wins < 0 or self.draws < 0:
            raise ValueError("Win/draw counts must be >= 0.")
        if self.candidate_wins + self.baseline_wins + self.draws != self.games_played:
            raise ValueError(
                "candidate_wins + baseline_wins + draws must equal games_played."
            )


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    candidate_model_id: ModelId
    candidate_model_version: ModelVersion
    baseline_model_id: ModelId
    baseline_model_version: ModelVersion
    number_of_games: int
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.number_of_games < 1:
            raise ValueError("number_of_games must be at least 1.")


@dataclass(frozen=True, slots=True)
class EvaluationJobAccepted:
    evaluation_job_id: EvaluationJobId
    candidate_model_id: ModelId
    candidate_model_version: ModelVersion
    baseline_model_id: ModelId
    baseline_model_version: ModelVersion
    number_of_games: int
    status: EvaluationJobStatus
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    evaluation_job_id: EvaluationJobId
    candidate_model_id: ModelId
    candidate_model_version: ModelVersion
    baseline_model_id: ModelId
    baseline_model_version: ModelVersion
    number_of_games: int
    status: EvaluationJobStatus
    submitted_at: datetime
    updated_at: datetime
    result: EvaluationResult | None = None
    notes: str | None = None
    failure_reason: str | None = None


class EvaluationJobNotFoundError(Exception):
    def __init__(self, evaluation_job_id: str) -> None:
        super().__init__(f"Evaluation job not found: {evaluation_job_id}")
        self.evaluation_job_id = evaluation_job_id
