from searchess_ai.application.usecase.submit_training_job import SubmitTrainingJobUseCase
from searchess_ai.domain.training import (
    TrainingJobRequest,
    TrainingJobStatus,
    TrainingProfile,
)
from searchess_ai.infrastructure.training.fake_training_scheduler import FakeTrainingScheduler
from searchess_ai.infrastructure.training.in_memory_training_job_repository import (
    InMemoryTrainingJobRepository,
)


def _use_case() -> SubmitTrainingJobUseCase:
    return SubmitTrainingJobUseCase(
        repository=InMemoryTrainingJobRepository(),
        scheduler=FakeTrainingScheduler(),
    )


def test_submit_returns_accepted_with_queued_status() -> None:
    request = TrainingJobRequest(training_profile=TrainingProfile.SUPERVISED)
    accepted = _use_case().execute(request)
    assert accepted.status == TrainingJobStatus.QUEUED


def test_submit_returns_accepted_with_correct_profile() -> None:
    request = TrainingJobRequest(training_profile=TrainingProfile.REINFORCEMENT)
    accepted = _use_case().execute(request)
    assert accepted.training_profile == TrainingProfile.REINFORCEMENT


def test_submit_assigns_non_empty_job_id() -> None:
    request = TrainingJobRequest(training_profile=TrainingProfile.SUPERVISED)
    accepted = _use_case().execute(request)
    assert accepted.training_job_id.value


def test_submit_stores_job_in_repository() -> None:
    repo = InMemoryTrainingJobRepository()
    use_case = SubmitTrainingJobUseCase(repository=repo, scheduler=FakeTrainingScheduler())
    request = TrainingJobRequest(training_profile=TrainingProfile.MIXED)
    accepted = use_case.execute(request)
    stored = repo.find_by_id(accepted.training_job_id)
    assert stored is not None
    assert stored.training_job_id == accepted.training_job_id


def test_submit_each_job_gets_unique_id() -> None:
    use_case = _use_case()
    request = TrainingJobRequest(training_profile=TrainingProfile.SUPERVISED)
    id_a = use_case.execute(request).training_job_id
    id_b = use_case.execute(request).training_job_id
    assert id_a != id_b
