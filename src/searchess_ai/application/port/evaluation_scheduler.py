from __future__ import annotations

from abc import ABC, abstractmethod

from searchess_ai.domain.evaluation import EvaluationJob


class EvaluationScheduler(ABC):
    @abstractmethod
    def submit(self, job: EvaluationJob) -> None:
        """Submit an evaluation job for execution."""
