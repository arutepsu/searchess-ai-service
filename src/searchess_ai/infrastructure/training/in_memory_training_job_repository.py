from __future__ import annotations

from searchess_ai.application.port.training_job_repository import TrainingJobRepository
from searchess_ai.domain.training import TrainingJob, TrainingJobId


class InMemoryTrainingJobRepository(TrainingJobRepository):
    def __init__(self) -> None:
        self._store: dict[str, TrainingJob] = {}

    def store(self, job: TrainingJob) -> None:
        self._store[job.training_job_id.value] = job

    def find_by_id(self, training_job_id: TrainingJobId) -> TrainingJob | None:
        return self._store.get(training_job_id.value)
