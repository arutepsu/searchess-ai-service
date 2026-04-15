from __future__ import annotations

from abc import ABC, abstractmethod

from searchess_ai.domain.training import TrainingJob


class TrainingScheduler(ABC):
    @abstractmethod
    def submit(self, job: TrainingJob) -> None:
        """Submit a training job for execution."""
