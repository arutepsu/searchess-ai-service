from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PolicyProfile(str, Enum):
    SAFE = "safe"
    FAST = "fast"
    BALANCED = "balanced"
    EXPERIMENTAL = "experimental"


class ModelStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    ACTIVE = "active"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class ModelId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("ModelId value must not be empty.")


@dataclass(frozen=True, slots=True)
class ModelVersion:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("ModelVersion value must not be empty.")


@dataclass(frozen=True, slots=True)
class ModelSummary:
    model_id: ModelId
    model_version: ModelVersion
    status: ModelStatus
    description: str
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ModelDetail:
    model_id: ModelId
    model_version: ModelVersion
    status: ModelStatus
    description: str
    supported_profiles: tuple[PolicyProfile, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str | None = None


class ModelNotFoundError(Exception):
    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model not found: {model_id}")
        self.model_id = model_id