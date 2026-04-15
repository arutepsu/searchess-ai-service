from __future__ import annotations

from abc import ABC, abstractmethod

from searchess_ai.domain.evaluation import EvaluationJob, EvaluationJobId


class EvaluationJobRepository(ABC):
    @abstractmethod
    def store(self, job: EvaluationJob) -> None:
        """Persist an evaluation job."""

    @abstractmethod
    def find_by_id(self, evaluation_job_id: EvaluationJobId) -> EvaluationJob | None:
        """Return a job by id, or None if not found."""
