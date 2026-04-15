from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PolicyProfile(str, Enum):
    SAFE = "safe"
    FAST = "fast"
    BALANCED = "balanced"
    EXPERIMENTAL = "experimental"


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