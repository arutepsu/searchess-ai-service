import pytest

from searchess_ai.application.usecase.get_evaluation_job_status import GetEvaluationJobStatusUseCase
from searchess_ai.application.usecase.submit_evaluation_job import SubmitEvaluationJobUseCase
from searchess_ai.domain.evaluation import (
    EvaluationJobId,
    EvaluationJobNotFoundError,
    EvaluationJobStatus,
    EvaluationRequest,
)
from searchess_ai.domain.model import ModelId, ModelVersion
from searchess_ai.infrastructure.evaluation.fake_evaluation_scheduler import FakeEvaluationScheduler
from searchess_ai.infrastructure.evaluation.in_memory_evaluation_job_repository import (
    InMemoryEvaluationJobRepository,
)


def _setup() -> tuple[SubmitEvaluationJobUseCase, GetEvaluationJobStatusUseCase]:
    repo = InMemoryEvaluationJobRepository()
    return (
        SubmitEvaluationJobUseCase(repository=repo, scheduler=FakeEvaluationScheduler()),
        GetEvaluationJobStatusUseCase(repository=repo),
    )


def _request() -> EvaluationRequest:
    return EvaluationRequest(
        candidate_model_id=ModelId("alpha-v1"),
        candidate_model_version=ModelVersion("1.0.0"),
        baseline_model_id=ModelId("legacy-v0"),
        baseline_model_version=ModelVersion("0.1.0"),
        number_of_games=10,
    )


def test_get_status_returns_job_when_found() -> None:
    submit, get_status = _setup()
    accepted = submit.execute(_request())
    job = get_status.execute(accepted.evaluation_job_id)
    assert job.evaluation_job_id == accepted.evaluation_job_id
    assert job.status == EvaluationJobStatus.QUEUED


def test_get_status_returns_correct_model_ids() -> None:
    submit, get_status = _setup()
    accepted = submit.execute(_request())
    job = get_status.execute(accepted.evaluation_job_id)
    assert job.candidate_model_id.value == "alpha-v1"
    assert job.baseline_model_id.value == "legacy-v0"


def test_get_status_raises_for_unknown_job() -> None:
    _, get_status = _setup()
    with pytest.raises(EvaluationJobNotFoundError) as exc_info:
        get_status.execute(EvaluationJobId("does-not-exist"))
    assert "does-not-exist" in str(exc_info.value)
