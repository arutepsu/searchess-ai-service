from __future__ import annotations

from fastapi import APIRouter, Depends

from searchess_ai.api.dependencies import get_get_model_detail_use_case, get_list_models_use_case
from searchess_ai.api.dto.model import ModelDetailDto, ModelSummaryDto
from searchess_ai.application.usecase.get_model_detail import GetModelDetailUseCase
from searchess_ai.application.usecase.list_models import ListModelsUseCase
from searchess_ai.domain.model import ModelId

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelSummaryDto])
def list_models(
    use_case: ListModelsUseCase = Depends(get_list_models_use_case),
) -> list[ModelSummaryDto]:
    summaries = use_case.execute()
    return [
        ModelSummaryDto(
            model_id=s.model_id.value,
            model_version=s.model_version.value,
            status=s.status.value,
            description=s.description,
            tags=list(s.tags),
        )
        for s in summaries
    ]


@router.get("/models/{model_id}", response_model=ModelDetailDto)
def get_model(
    model_id: str,
    use_case: GetModelDetailUseCase = Depends(get_get_model_detail_use_case),
) -> ModelDetailDto:
    detail = use_case.execute(ModelId(model_id))
    return ModelDetailDto(
        model_id=detail.model_id.value,
        model_version=detail.model_version.value,
        status=detail.status.value,
        description=detail.description,
        supported_profiles=[p.value for p in detail.supported_profiles],
        tags=list(detail.tags),
        notes=detail.notes,
    )
