from __future__ import annotations

import json
from typing import ClassVar

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from searchess_ai.domain.evaluation import EvaluationJobNotFoundError
from searchess_ai.domain.model import ModelNotFoundError
from searchess_ai.domain.training import TrainingJobNotFoundError
from searchess_ai.infrastructure.inference.openspiel_inference_engine import OpenSpielAdapterError


# ---------------------------------------------------------------------------
# Contract exception types (raised inside the inference route)
# ---------------------------------------------------------------------------

class InferenceContractError(Exception):
    status_code: ClassVar[int]
    code: ClassVar[str]


class BadPositionError(InferenceContractError):
    status_code = 422
    code = "BAD_POSITION"


class NoLegalMoveError(InferenceContractError):
    status_code = 422
    code = "NO_LEGAL_MOVE"


class EngineUnavailableError(InferenceContractError):
    status_code = 503
    code = "ENGINE_UNAVAILABLE"


class EngineTimeoutError(InferenceContractError):
    status_code = 504
    code = "ENGINE_TIMEOUT"


class EngineFailureError(InferenceContractError):
    status_code = 500
    code = "ENGINE_FAILURE"


# ---------------------------------------------------------------------------
# Helper: build a contract error JSONResponse
# ---------------------------------------------------------------------------

def inference_error_response(
    request_id: str,
    exc: InferenceContractError,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"requestId": request_id, "code": exc.code, "message": str(exc)},
    )


async def extract_request_id(request: Request) -> str:
    """Best-effort extraction of requestId from the raw request body."""
    try:
        raw = await request.body()
        body = json.loads(raw)
        return str(body.get("requestId", ""))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# FastAPI exception handlers (registered in app.py)
# ---------------------------------------------------------------------------

async def inference_contract_error_handler(
    request: Request, exc: InferenceContractError
) -> JSONResponse:
    request_id = await extract_request_id(request)
    return inference_error_response(request_id, exc)


# --- Handlers preserved for non-inference routes ---

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
