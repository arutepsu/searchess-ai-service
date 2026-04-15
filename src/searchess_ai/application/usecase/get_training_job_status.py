from __future__ import annotations

from dataclasses import dataclass

from searchess_ai.application.port.training_job_repository import TrainingJobRepository
from searchess_ai.domain.training import TrainingJob, TrainingJobId, TrainingJobNotFoundError


@dataclass(slots=True)
class GetTrainingJobStatusUseCase:
    repository: TrainingJobRepository

    def execute(self, training_job_id: TrainingJobId) -> TrainingJob:
        job = self.repository.find_by_id(training_job_id)
        if job is None:
            raise TrainingJobNotFoundError(training_job_id.value)
        return job
