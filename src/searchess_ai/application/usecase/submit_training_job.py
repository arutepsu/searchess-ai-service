from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from searchess_ai.application.port.training_job_repository import TrainingJobRepository
from searchess_ai.application.port.training_scheduler import TrainingScheduler
from searchess_ai.domain.training import (
    TrainingJob,
    TrainingJobAccepted,
    TrainingJobId,
    TrainingJobRequest,
    TrainingJobStatus,
)


@dataclass(slots=True)
class SubmitTrainingJobUseCase:
    repository: TrainingJobRepository
    scheduler: TrainingScheduler

    def execute(self, request: TrainingJobRequest) -> TrainingJobAccepted:
        now = datetime.now(UTC)
        job = TrainingJob(
            training_job_id=TrainingJobId(str(uuid4())),
            training_profile=request.training_profile,
            status=TrainingJobStatus.QUEUED,
            submitted_at=now,
            updated_at=now,
            base_model_id=request.base_model_id,
            base_model_version=request.base_model_version,
            notes=request.notes,
        )
        self.repository.store(job)
        self.scheduler.submit(job)
        return TrainingJobAccepted(
            training_job_id=job.training_job_id,
            training_profile=job.training_profile,
            status=job.status,
            submitted_at=job.submitted_at,
        )
