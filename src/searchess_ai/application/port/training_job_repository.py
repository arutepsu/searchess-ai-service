from __future__ import annotations

from abc import ABC, abstractmethod

from searchess_ai.domain.training import TrainingJob, TrainingJobId


class TrainingJobRepository(ABC):
    @abstractmethod
    def store(self, job: TrainingJob) -> None:
        """Persist a training job."""

    @abstractmethod
    def find_by_id(self, training_job_id: TrainingJobId) -> TrainingJob | None:
        """Return a job by id, or None if not found."""
