import pytest

from searchess_ai.application.usecase.get_model_detail import GetModelDetailUseCase
from searchess_ai.domain.model import ModelId, ModelNotFoundError, ModelStatus
from searchess_ai.infrastructure.model.in_memory_model_repository import InMemoryModelRepository


def _use_case() -> GetModelDetailUseCase:
    return GetModelDetailUseCase(repository=InMemoryModelRepository())


def test_get_model_detail_returns_active_model() -> None:
    detail = _use_case().execute(ModelId("alpha-v1"))
    assert detail.model_id.value == "alpha-v1"
    assert detail.status == ModelStatus.ACTIVE
    assert len(detail.supported_profiles) > 0


def test_get_model_detail_returns_candidate_model() -> None:
    detail = _use_case().execute(ModelId("beta-v2"))
    assert detail.status == ModelStatus.CANDIDATE


def test_get_model_detail_raises_for_unknown_model() -> None:
    with pytest.raises(ModelNotFoundError) as exc_info:
        _use_case().execute(ModelId("does-not-exist"))
    assert "does-not-exist" in str(exc_info.value)
