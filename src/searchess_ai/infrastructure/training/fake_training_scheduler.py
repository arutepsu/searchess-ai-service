from __future__ import annotations

from searchess_ai.application.port.training_scheduler import TrainingScheduler
from searchess_ai.domain.training import TrainingJob


class FakeTrainingScheduler(TrainingScheduler):
    def submit(self, job: TrainingJob) -> None:
        pass  # No-op: job remains QUEUED; no real execution in this phase
