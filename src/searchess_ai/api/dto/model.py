from __future__ import annotations

from pydantic import BaseModel


class ModelSummaryDto(BaseModel):
    model_id: str
    model_version: str
    status: str
    description: str
    tags: list[str]


class ModelDetailDto(BaseModel):
    model_id: str
    model_version: str
    status: str
    description: str
    supported_profiles: list[str]
    tags: list[str]
    notes: str | None = None
