from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from searchess_ai.api.dependencies import get_choose_move_use_case
from searchess_ai.api.dto.inference import (
    MoveDtoOut,
    MoveSuggestionRequestDto,
    MoveSuggestionResponseDto,
)
from searchess_ai.api.errors import (
    EngineFailureError,
    InferenceContractError,
    inference_error_response,
)
from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.domain.game import LegalMoveSet, Move, Position, SideToMove
from searchess_ai.domain.inference import InferenceRequest
from searchess_ai.domain.model import ModelId

router = APIRouter(tags=["inference"])

_PROMO_TO_UCI: dict[str, str] = {
    "queen": "q",
    "rook": "r",
    "bishop": "b",
    "knight": "n",
}
_UCI_TO_PROMO: dict[str, str] = {v: k for k, v in _PROMO_TO_UCI.items()}


def _to_uci(dto: "MoveSuggestionRequestDto.__annotations__['legal_moves'].__args__[0]") -> str:  # type: ignore[valid-type]
    suffix = _PROMO_TO_UCI.get(dto.promotion, "") if dto.promotion else ""
    return f"{dto.from_square}{dto.to}{suffix}"


def _from_uci(value: str) -> MoveDtoOut:
    from_sq = value[0:2]
    to_sq = value[2:4]
    promo_char = value[4] if len(value) == 5 else None
    return MoveDtoOut(
        from_square=from_sq,
        to=to_sq,
        promotion=_UCI_TO_PROMO.get(promo_char) if promo_char else None,
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

    try:
        response_dto = _execute(dto, use_case)
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
) -> MoveSuggestionResponseDto:
    legal_moves = LegalMoveSet(tuple(Move(_to_uci(m)) for m in dto.legal_moves))

    inference_request = InferenceRequest(
        request_id=dto.request_id,
        match_id=dto.game_id,
        position=Position(dto.fen),
        side_to_move=SideToMove(dto.side_to_move),
        legal_moves=legal_moves,
        model_id=ModelId(dto.engine.engine_id) if dto.engine and dto.engine.engine_id else None,
    )

    try:
        decision = use_case.execute(inference_request)
    except ValueError as exc:
        raise EngineFailureError(str(exc)) from exc

    return MoveSuggestionResponseDto(
        request_id=decision.request_id,
        move=_from_uci(decision.selected_move.value),
        engine_id=decision.model_id.value,
        engine_version=decision.model_version.value,
        elapsed_millis=decision.decision_time_millis,
        confidence=decision.confidence,
    )
