from __future__ import annotations

from typing import Literal

from pydantic import AliasGenerator, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

_CAMEL = ConfigDict(
    alias_generator=AliasGenerator(
        validation_alias=to_camel,
        serialization_alias=to_camel,
    ),
    populate_by_name=True,
)

# ---------------------------------------------------------------------------
# Move DTO — needs explicit aliases because "from" is a Python keyword.
# ---------------------------------------------------------------------------

class MoveDtoIn(BaseModel):
    """Structured move in an incoming request (legalMoves list)."""

    model_config = ConfigDict(populate_by_name=True)

    from_square: str = Field(..., alias="from", pattern=r"^[a-h][1-8]$")
    to: str = Field(..., pattern=r"^[a-h][1-8]$")
    promotion: Literal["queen", "rook", "bishop", "knight"] | None = None


class MoveDtoOut(BaseModel):
    """Structured move in a response body."""

    model_config = ConfigDict(populate_by_name=True)

    from_square: str = Field(..., serialization_alias="from")
    to: str
    promotion: str | None = None


# ---------------------------------------------------------------------------
# Nested request sub-objects (all camelCase on the wire via _CAMEL)
# ---------------------------------------------------------------------------

class EngineSelectionDto(BaseModel):
    model_config = _CAMEL
    engine_id: str | None = None


class ExecutionLimitsDto(BaseModel):
    model_config = _CAMEL
    timeout_millis: int = Field(..., ge=1)


class RequestMetadataDto(BaseModel):
    model_config = _CAMEL
    mode: str | None = None


# ---------------------------------------------------------------------------
# Top-level request / response
# ---------------------------------------------------------------------------

class MoveSuggestionRequestDto(BaseModel):
    model_config = _CAMEL

    request_id: str = Field(..., min_length=1)
    game_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    side_to_move: Literal["white", "black"]
    fen: str = Field(..., min_length=1)
    legal_moves: list[MoveDtoIn] = Field(..., min_length=1)
    engine: EngineSelectionDto | None = None
    limits: ExecutionLimitsDto
    metadata: RequestMetadataDto | None = None


class MoveSuggestionResponseDto(BaseModel):
    model_config = _CAMEL

    request_id: str
    move: MoveDtoOut
    engine_id: str | None = None
    engine_version: str | None = None
    elapsed_millis: int | None = None
    confidence: float | None = None


class ErrorResponseDto(BaseModel):
    model_config = _CAMEL

    request_id: str
    code: str
    message: str
