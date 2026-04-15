from __future__ import annotations

from abc import ABC, abstractmethod

from searchess_ai.domain.model import ModelDetail, ModelId, ModelSummary


class ModelRepository(ABC):
    @abstractmethod
    def list_all(self) -> list[ModelSummary]:
        """Return summaries of all known models."""

    @abstractmethod
    def find_by_id(self, model_id: ModelId) -> ModelDetail | None:
        """Return full detail for one model, or None if not found."""
