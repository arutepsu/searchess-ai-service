"""
Integration test hook for POST /v1/move-suggestions.

Activated by metadata.testMode in the request.  See docs/test-hook.md.

THIS MODULE MUST NOT be imported or invoked on any normal inference path.
It is called only when testMode is explicitly set to a known value.
"""
from __future__ import annotations

from fastapi.responses import JSONResponse

from searchess_ai.api.dto.inference import MoveSuggestionRequestDto

_FILES = "abcdefgh"
_RANKS = "12345678"


def _pick_illegal_move(dto: MoveSuggestionRequestDto) -> dict[str, str]:
    """Return a syntactically plausible move that is NOT in legalMoves.

    Iterates all 4 096 from/to square combinations in a fixed order and
    returns the first one absent from the request's legal move set.
    In practice this always resolves on the first or second candidate
    (a1a1 or a1a2) since those are never legal.

    The fallback {"from": "a1", "to": "h8"} is unreachable: there are at
    most ~219 legal moves in any chess position, far fewer than 4 096.
    """
    legal: set[str] = {f"{m.from_square}{m.to}" for m in dto.legal_moves}
    for f in _FILES:
        for r in _RANKS:
            for tf in _FILES:
                for tr in _RANKS:
                    if f"{f}{r}{tf}{tr}" not in legal:
                        return {"from": f"{f}{r}", "to": f"{tf}{tr}"}
    return {"from": "a1", "to": "h8"}  # unreachable


def illegal_move_response(dto: MoveSuggestionRequestDto) -> JSONResponse:
    """Return a structurally valid 200 whose move is not in legalMoves.

    Lets the Scala Game Service reach its own legality-check path and
    confirm it correctly rejects a move that was not offered by the engine.
    """
    return JSONResponse(
        status_code=200,
        content={
            "requestId": dto.request_id,
            "move": _pick_illegal_move(dto),
            "engineId": "test-hook",
            "engineVersion": "0.0.0",
            "elapsedMillis": 0,
            "confidence": None,
        },
    )


def malformed_response(dto: MoveSuggestionRequestDto) -> JSONResponse:
    """Return a deliberately malformed 200 — the required 'move' field is absent.

    Forces the Scala client/decoder into its malformed-provider-response path.
    The body is otherwise valid JSON so the failure is a schema problem, not
    a parse problem.
    """
    return JSONResponse(
        status_code=200,
        content={
            "requestId": dto.request_id,
            # 'move' intentionally omitted — required field in shared contract
            "engineId": "test-hook",
            "engineVersion": "0.0.0",
        },
    )
