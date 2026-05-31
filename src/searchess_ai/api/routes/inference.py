from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from opentelemetry import context as otel_context
from opentelemetry import trace
from pydantic import ValidationError

from searchess_ai.api.dependencies import get_choose_move_use_case
from searchess_ai.api.dto.inference import (
    MoveSuggestionRequestDto,
    MoveSuggestionResponseDto,
)
from searchess_ai.api.dto.move_mapper import move_dto_to_uci, uci_to_move_dto
from searchess_ai.api.errors import (
    BadPositionError,
    EngineFailureError,
    EngineTimeoutError,
    EngineUnavailableError,
    InferenceContractError,
    inference_error_response,
)
from searchess_ai.api.routes import _test_hook
from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.domain.game import LegalMoveSet, Move, Position, SideToMove
from searchess_ai.domain.inference import InferenceRequest
from searchess_ai.domain.model import ModelId
from searchess_ai.infrastructure.inference._openspiel_mapping import (
    BadPositionAdapterError,
    OpenSpielAdapterError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])

# Timeout enforcement happens here — at the HTTP boundary — not inside the
# engine. Engines are synchronous and unaware of request deadlines; enforcing
# the contract limit here keeps engines simple and the boundary explicit.
#
# Note: Python threads cannot be forcibly killed. A thread that exceeds its
# timeout continues running until the engine returns naturally. The caller
# already received a 504 by that point. max_workers is kept small so that
# leaked threads do not accumulate unboundedly under load.
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="inference",
)


@router.post("/move-suggestions")
async def suggest_move(
    request: Request,
    use_case: ChooseMoveUseCase = Depends(get_choose_move_use_case),
) -> JSONResponse:
    # Parse raw body so we can echo requestId in validation error responses.
    try:
        raw = await request.body()
        body = json.loads(raw)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"requestId": "", "code": "BAD_REQUEST", "message": "request body is not valid JSON"},
        )

    request_id = str(body.get("requestId", ""))

    try:
        dto = MoveSuggestionRequestDto.model_validate(body)
    except ValidationError as exc:
        first_msg = exc.errors(include_url=False)[0]["msg"]
        return JSONResponse(
            status_code=400,
            content={"requestId": request_id, "code": "BAD_REQUEST", "message": first_msg},
        )

    # Integration test hook — short-circuits normal inference when testMode is set.
    # Unknown values fall through to normal behavior unchanged.
    if dto.metadata and dto.metadata.test_mode == "illegal_move":
        return _test_hook.illegal_move_response(dto)
    if dto.metadata and dto.metadata.test_mode == "malformed_response":
        return _test_hook.malformed_response(dto)

    timeout_seconds = dto.limits.timeout_millis / 1000.0
    started_at = time.monotonic()
    # Capture the current OTel context (carries the FastAPI request span created
    # by FastAPIInstrumentor) so the manual python_ai.inference span created
    # inside the thread pool becomes a proper child span.
    ctx = otel_context.get_current()
    future = _executor.submit(_execute, dto, use_case, ctx)

    try:
        response_dto = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        logger.warning(
            "inference timeout",
            extra={
                "event": "inference_timeout",
                "request_id": dto.request_id,
                "game_id": dto.game_id,
                "session_id": dto.session_id,
                "timeout_millis": dto.limits.timeout_millis,
                "elapsed_millis": elapsed_ms,
            },
        )
        return inference_error_response(
            dto.request_id,
            EngineTimeoutError(f"engine did not respond within {dto.limits.timeout_millis}ms"),
        )
    except InferenceContractError as exc:
        return inference_error_response(dto.request_id, exc)
    except Exception as exc:
        return inference_error_response(
            dto.request_id,
            EngineFailureError(f"unexpected engine error: {exc}"),
        )

    return JSONResponse(content=response_dto.model_dump(by_alias=True))


def _execute(
    dto: MoveSuggestionRequestDto,
    use_case: ChooseMoveUseCase,
    parent_context: object | None = None,
) -> MoveSuggestionResponseDto:
    # Attach the parent OTel context so the manual span is a child of the
    # FastAPI request span created in the async handler.
    otel_token = otel_context.attach(parent_context) if parent_context is not None else None
    try:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("python_ai.inference") as span:
            span.set_attribute("ai.backend", os.getenv("INFERENCE_BACKEND", "fake"))
            span.set_attribute("ai.legal_move_count", len(dto.legal_moves))
            span.set_attribute("ai.request_id", dto.request_id)
            if dto.engine and dto.engine.engine_id:
                span.set_attribute("ai.engine_id", dto.engine.engine_id)

            # Adapt public contract → internal domain objects.
            legal_moves = LegalMoveSet(tuple(Move(move_dto_to_uci(m)) for m in dto.legal_moves))

            inference_request = InferenceRequest(
                request_id=dto.request_id,
                # gameId is the closest public concept to internal match_id
                match_id=dto.game_id,
                position=Position(dto.fen),
                side_to_move=SideToMove(dto.side_to_move),
                legal_moves=legal_moves,
                model_id=ModelId(dto.engine.engine_id) if dto.engine and dto.engine.engine_id else None,
                # limits.timeoutMillis → remaining_time_millis (adapter step; internal semantics differ)
                remaining_time_millis=dto.limits.timeout_millis,
            )

            try:
                decision = use_case.execute(inference_request)
            except BadPositionAdapterError as exc:
                raise BadPositionError(str(exc)) from exc
            except OpenSpielAdapterError as exc:
                msg = str(exc)
                if "not installed" in msg.lower():
                    raise EngineUnavailableError(msg) from exc
                raise EngineFailureError(msg) from exc
            except ValueError as exc:
                raise EngineFailureError(str(exc)) from exc

            # Record the actual engine id chosen by the backend.
            span.set_attribute("ai.engine_id", decision.model_id.value)

            # Adapt internal decision → public response contract.
            return MoveSuggestionResponseDto(
                request_id=decision.request_id,
                move=uci_to_move_dto(decision.selected_move.value),
                engine_id=decision.model_id.value,
                engine_version=decision.model_version.value,
                elapsed_millis=decision.decision_time_millis,
                confidence=decision.confidence,
            )
    finally:
        if otel_token is not None:
            otel_context.detach(otel_token)
