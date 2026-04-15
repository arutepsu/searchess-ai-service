from __future__ import annotations

from searchess_ai.application.port.evaluation_scheduler import EvaluationScheduler
from searchess_ai.domain.evaluation import EvaluationJob


class FakeEvaluationScheduler(EvaluationScheduler):
    def submit(self, job: EvaluationJob) -> None:
        pass  # No-op: job remains QUEUED; no real execution in this phase
