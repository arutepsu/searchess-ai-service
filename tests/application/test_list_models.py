from searchess_ai.application.usecase.list_models import ListModelsUseCase
from searchess_ai.domain.model import ModelStatus
from searchess_ai.infrastructure.model.in_memory_model_repository import InMemoryModelRepository


def test_list_models_returns_all_fake_models() -> None:
    use_case = ListModelsUseCase(repository=InMemoryModelRepository())
    summaries = use_case.execute()
    assert len(summaries) == 3


def test_list_models_contains_expected_statuses() -> None:
    use_case = ListModelsUseCase(repository=InMemoryModelRepository())
    statuses = {s.status for s in use_case.execute()}
    assert ModelStatus.ACTIVE in statuses
    assert ModelStatus.CANDIDATE in statuses
    assert ModelStatus.RETIRED in statuses


def test_list_models_returns_summaries_with_required_fields() -> None:
    use_case = ListModelsUseCase(repository=InMemoryModelRepository())
    for summary in use_case.execute():
        assert summary.model_id.value
        assert summary.model_version.value
        assert summary.description
