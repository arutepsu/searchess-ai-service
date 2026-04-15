from searchess_ai.application.usecase.submit_evaluation_job import SubmitEvaluationJobUseCase
from searchess_ai.domain.evaluation import EvaluationJobStatus, EvaluationRequest
from searchess_ai.domain.model import ModelId, ModelVersion
from searchess_ai.infrastructure.evaluation.fake_evaluation_scheduler import FakeEvaluationScheduler
from searchess_ai.infrastructure.evaluation.in_memory_evaluation_job_repository import (
    InMemoryEvaluationJobRepository,
)


def _request() -> EvaluationRequest:
    return EvaluationRequest(
        candidate_model_id=ModelId("alpha-v1"),
        candidate_model_version=ModelVersion("1.0.0"),
        baseline_model_id=ModelId("legacy-v0"),
        baseline_model_version=ModelVersion("0.1.0"),
        number_of_games=10,
    )


def _use_case() -> SubmitEvaluationJobUseCase:
    return SubmitEvaluationJobUseCase(
        repository=InMemoryEvaluationJobRepository(),
        scheduler=FakeEvaluationScheduler(),
    )


def test_submit_returns_accepted_with_queued_status() -> None:
    accepted = _use_case().execute(_request())
    assert accepted.status == EvaluationJobStatus.QUEUED


def test_submit_returns_accepted_with_correct_models() -> None:
    accepted = _use_case().execute(_request())
    assert accepted.candidate_model_id.value == "alpha-v1"
    assert accepted.baseline_model_id.value == "legacy-v0"


def test_submit_returns_accepted_with_correct_game_count() -> None:
    accepted = _use_case().execute(_request())
    assert accepted.number_of_games == 10


def test_submit_assigns_non_empty_job_id() -> None:
    accepted = _use_case().execute(_request())
    assert accepted.evaluation_job_id.value


def test_submit_stores_job_in_repository() -> None:
    repo = InMemoryEvaluationJobRepository()
    use_case = SubmitEvaluationJobUseCase(repository=repo, scheduler=FakeEvaluationScheduler())
    accepted = use_case.execute(_request())
    stored = repo.find_by_id(accepted.evaluation_job_id)
    assert stored is not None
    assert stored.evaluation_job_id == accepted.evaluation_job_id


def test_submit_each_job_gets_unique_id() -> None:
    use_case = _use_case()
    id_a = use_case.execute(_request()).evaluation_job_id
    id_b = use_case.execute(_request()).evaluation_job_id
    assert id_a != id_b
