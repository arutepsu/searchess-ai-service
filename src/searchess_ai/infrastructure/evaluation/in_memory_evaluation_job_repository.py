from __future__ import annotations

from searchess_ai.application.port.evaluation_job_repository import EvaluationJobRepository
from searchess_ai.domain.evaluation import EvaluationJob, EvaluationJobId


class InMemoryEvaluationJobRepository(EvaluationJobRepository):
    def __init__(self) -> None:
        self._store: dict[str, EvaluationJob] = {}

    def store(self, job: EvaluationJob) -> None:
        self._store[job.evaluation_job_id.value] = job

    def find_by_id(self, evaluation_job_id: EvaluationJobId) -> EvaluationJob | None:
        return self._store.get(evaluation_job_id.value)
