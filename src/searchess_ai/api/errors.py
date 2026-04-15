from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from searchess_ai.domain.evaluation import EvaluationJobNotFoundError
from searchess_ai.domain.model import ModelNotFoundError
from searchess_ai.domain.training import TrainingJobNotFoundError
from searchess_ai.infrastructure.inference.openspiel_inference_engine import OpenSpielAdapterError


class ErrorResponse(BaseModel):
    error: str
    type: str


async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": str(exc), "type": "not_found"},
    )


async def training_job_not_found_handler(
    request: Request, exc: TrainingJobNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": str(exc), "type": "not_found"},
    )


async def openspiel_adapter_error_handler(
    request: Request, exc: OpenSpielAdapterError
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": "adapter_error"},
    )


async def evaluation_job_not_found_handler(
    request: Request, exc: EvaluationJobNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": str(exc), "type": "not_found"},
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": str(exc), "type": "validation_error"},
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": "internal_error"},
    )
