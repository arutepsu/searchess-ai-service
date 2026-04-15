from __future__ import annotations

from dataclasses import dataclass

from searchess_ai.application.port.model_repository import ModelRepository
from searchess_ai.domain.model import ModelSummary


@dataclass(slots=True)
class ListModelsUseCase:
    repository: ModelRepository

    def execute(self) -> list[ModelSummary]:
        return self.repository.list_all()
