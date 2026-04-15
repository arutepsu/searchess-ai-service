import pytest

from searchess_ai.application.usecase.get_training_job_status import GetTrainingJobStatusUseCase
from searchess_ai.application.usecase.submit_training_job import SubmitTrainingJobUseCase
from searchess_ai.domain.training import (
    TrainingJobId,
    TrainingJobNotFoundError,
    TrainingJobRequest,
    TrainingJobStatus,
    TrainingProfile,
)
from searchess_ai.infrastructure.training.fake_training_scheduler import FakeTrainingScheduler
from searchess_ai.infrastructure.training.in_memory_training_job_repository import (
    InMemoryTrainingJobRepository,
)


def _setup() -> tuple[SubmitTrainingJobUseCase, GetTrainingJobStatusUseCase]:
    repo = InMemoryTrainingJobRepository()
    return (
        SubmitTrainingJobUseCase(repository=repo, scheduler=FakeTrainingScheduler()),
        GetTrainingJobStatusUseCase(repository=repo),
    )


def test_get_status_returns_job_when_found() -> None:
    submit, get_status = _setup()
    accepted = submit.execute(TrainingJobRequest(training_profile=TrainingProfile.SUPERVISED))
    job = get_status.execute(accepted.training_job_id)
    assert job.training_job_id == accepted.training_job_id
    assert job.status == TrainingJobStatus.QUEUED


def test_get_status_returns_correct_profile() -> None:
    submit, get_status = _setup()
    accepted = submit.execute(TrainingJobRequest(training_profile=TrainingProfile.REINFORCEMENT))
    job = get_status.execute(accepted.training_job_id)
    assert job.training_profile == TrainingProfile.REINFORCEMENT


def test_get_status_raises_for_unknown_job() -> None:
    _, get_status = _setup()
    with pytest.raises(TrainingJobNotFoundError) as exc_info:
        get_status.execute(TrainingJobId("does-not-exist"))
    assert "does-not-exist" in str(exc_info.value)
