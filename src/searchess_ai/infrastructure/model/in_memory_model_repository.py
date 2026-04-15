from __future__ import annotations

from searchess_ai.application.port.model_repository import ModelRepository
from searchess_ai.domain.model import (
    ModelDetail,
    ModelId,
    ModelStatus,
    ModelSummary,
    ModelVersion,
    PolicyProfile,
)

_MODELS: dict[str, ModelDetail] = {
    "alpha-v1": ModelDetail(
        model_id=ModelId("alpha-v1"),
        model_version=ModelVersion("1.0.0"),
        status=ModelStatus.ACTIVE,
        description="First stable model trained on self-play data.",
        supported_profiles=(PolicyProfile.BALANCED, PolicyProfile.SAFE),
        tags=("self-play", "stable"),
        notes="Promoted to active after passing candidate evaluation.",
    ),
    "beta-v2": ModelDetail(
        model_id=ModelId("beta-v2"),
        model_version=ModelVersion("2.0.0-rc1"),
        status=ModelStatus.CANDIDATE,
        description="Second generation model with improved endgame play.",
        supported_profiles=(PolicyProfile.BALANCED, PolicyProfile.FAST, PolicyProfile.EXPERIMENTAL),
        tags=("endgame", "experimental"),
        notes="Under evaluation. Do not use in production matches.",
    ),
    "legacy-v0": ModelDetail(
        model_id=ModelId("legacy-v0"),
        model_version=ModelVersion("0.1.0"),
        status=ModelStatus.RETIRED,
        description="Initial prototype model, no longer in use.",
        supported_profiles=(PolicyProfile.SAFE,),
        tags=("legacy",),
        notes="Retired after alpha-v1 reached active status.",
    ),
}


class InMemoryModelRepository(ModelRepository):
    def list_all(self) -> list[ModelSummary]:
        return [
            ModelSummary(
                model_id=d.model_id,
                model_version=d.model_version,
                status=d.status,
                description=d.description,
                tags=d.tags,
            )
            for d in _MODELS.values()
        ]

    def find_by_id(self, model_id: ModelId) -> ModelDetail | None:
        return _MODELS.get(model_id.value)
