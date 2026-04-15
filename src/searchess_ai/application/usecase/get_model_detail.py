from __future__ import annotations

from dataclasses import dataclass

from searchess_ai.application.port.model_repository import ModelRepository
from searchess_ai.domain.model import ModelDetail, ModelId, ModelNotFoundError


@dataclass(slots=True)
class GetModelDetailUseCase:
    repository: ModelRepository

    def execute(self, model_id: ModelId) -> ModelDetail:
        detail = self.repository.find_by_id(model_id)
        if detail is None:
            raise ModelNotFoundError(model_id.value)
        return detail
