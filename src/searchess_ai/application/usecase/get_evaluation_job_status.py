from __future__ import annotations

from dataclasses import dataclass

from searchess_ai.application.port.evaluation_job_repository import EvaluationJobRepository
from searchess_ai.domain.evaluation import (
    EvaluationJob,
    EvaluationJobId,
    EvaluationJobNotFoundError,
)


@dataclass(slots=True)
class GetEvaluationJobStatusUseCase:
    repository: EvaluationJobRepository

    def execute(self, evaluation_job_id: EvaluationJobId) -> EvaluationJob:
        job = self.repository.find_by_id(evaluation_job_id)
        if job is None:
            raise EvaluationJobNotFoundError(evaluation_job_id.value)
        return job
