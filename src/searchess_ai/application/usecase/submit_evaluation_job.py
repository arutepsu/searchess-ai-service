from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from searchess_ai.application.port.evaluation_job_repository import EvaluationJobRepository
from searchess_ai.application.port.evaluation_scheduler import EvaluationScheduler
from searchess_ai.domain.evaluation import (
    EvaluationJob,
    EvaluationJobAccepted,
    EvaluationJobId,
    EvaluationJobStatus,
    EvaluationRequest,
)


@dataclass(slots=True)
class SubmitEvaluationJobUseCase:
    repository: EvaluationJobRepository
    scheduler: EvaluationScheduler

    def execute(self, request: EvaluationRequest) -> EvaluationJobAccepted:
        now = datetime.now(UTC)
        job = EvaluationJob(
            evaluation_job_id=EvaluationJobId(str(uuid4())),
            candidate_model_id=request.candidate_model_id,
            candidate_model_version=request.candidate_model_version,
            baseline_model_id=request.baseline_model_id,
            baseline_model_version=request.baseline_model_version,
            number_of_games=request.number_of_games,
            status=EvaluationJobStatus.QUEUED,
            submitted_at=now,
            updated_at=now,
            notes=request.notes,
        )
        self.repository.store(job)
        self.scheduler.submit(job)
        return EvaluationJobAccepted(
            evaluation_job_id=job.evaluation_job_id,
            candidate_model_id=job.candidate_model_id,
            candidate_model_version=job.candidate_model_version,
            baseline_model_id=job.baseline_model_id,
            baseline_model_version=job.baseline_model_version,
            number_of_games=job.number_of_games,
            status=job.status,
            submitted_at=job.submitted_at,
        )
