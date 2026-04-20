"""Pure mapping between public MoveDto objects and internal UCI move strings.

Public contract: camelCase MoveDto fields (from/to/promotion).
Internal contract: UCI strings (e.g. "e2e4", "e7e8q").

These functions are the sole place where promotion names ↔ UCI suffix
conversion is defined for the HTTP boundary.
"""
from __future__ import annotations

from searchess_ai.api.dto.inference import MoveDtoIn, MoveDtoOut

_PROMO_TO_UCI: dict[str, str] = {
    "queen": "q",
    "rook": "r",
    "bishop": "b",
    "knight": "n",
}
_UCI_TO_PROMO: dict[str, str] = {v: k for k, v in _PROMO_TO_UCI.items()}


def move_dto_to_uci(dto: MoveDtoIn) -> str:
    """Convert a public MoveDto from a request into a UCI string.

    Examples:
        {from:"e2", to:"e4"}                  -> "e2e4"
        {from:"e7", to:"e8", promotion:"queen"} -> "e7e8q"
    """
    suffix = _PROMO_TO_UCI[dto.promotion] if dto.promotion else ""
    return f"{dto.from_square}{dto.to}{suffix}"


def uci_to_move_dto(uci: str) -> MoveDtoOut:
    """Convert an internal UCI string into a public MoveDto for a response.

    Examples:
        "e2e4"   -> {from:"e2", to:"e4"}
        "e7e8q"  -> {from:"e7", to:"e8", promotion:"queen"}

    Raises:
        ValueError: if the UCI string length is not 4 or 5, or if the
            promotion character is unrecognised.
    """
    if len(uci) not in (4, 5):
        raise ValueError(f"Invalid UCI move string (expected 4 or 5 chars): {uci!r}")
    from_sq = uci[0:2]
    to_sq = uci[2:4]
    promo_char = uci[4] if len(uci) == 5 else None
    if promo_char is not None and promo_char not in _UCI_TO_PROMO:
        raise ValueError(
            f"Unknown promotion character {promo_char!r} in UCI move: {uci!r}. "
            f"Expected one of: {list(_UCI_TO_PROMO)}"
        )
    return MoveDtoOut(
        from_square=from_sq,
        to=to_sq,
        promotion=_UCI_TO_PROMO[promo_char] if promo_char else None,
    )
