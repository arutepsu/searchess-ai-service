"""Game-level filtering for the dataset pipeline.

All filtering decisions are captured in FilterResult so that the pipeline can
produce an accurate rejection breakdown in dataset_metadata.json.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import chess.pgn
except ImportError as exc:
    raise ImportError(
        "python-chess is required. Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.data_pipeline.config import FilterConfig

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}

# Standard chess variant identifiers that indicate non-standard games.
_VARIANT_HEADERS = {"Variant", "GameType"}
_STANDARD_VARIANT_VALUES = {"", "standard", "Standard", "chess", "Chess"}


@dataclass
class FilterResult:
    accepted: bool
    reason: str = "ok"


def filter_game(game: chess.pgn.Game, config: FilterConfig) -> FilterResult:
    """Decide whether to accept a parsed game.

    Returns FilterResult with accepted=True or a named rejection reason.
    The reason string is used to populate rejection_reasons in metadata.
    """
    headers = game.headers

    # Reject chess variants
    if config.skip_variants:
        for variant_key in _VARIANT_HEADERS:
            variant_val = headers.get(variant_key, "")
            if variant_val and variant_val not in _STANDARD_VARIANT_VALUES:
                return FilterResult(False, "variant")
        # Also reject if FEN header implies a non-standard start (Fischer random etc.)
        if headers.get("FEN") and headers.get("SetUp") == "1":
            return FilterResult(False, "non_standard_start")

    # Reject games without a result when required
    if config.require_result:
        result = headers.get("Result", "*")
        if result not in VALID_RESULTS:
            return FilterResult(False, "no_result")

    # Reject ultra-short games
    ply_count = _estimate_ply_count(game)
    if ply_count < config.min_ply_count:
        return FilterResult(False, "too_short")

    # Reject games below rating threshold
    if config.min_rating is not None:
        white_elo = _parse_elo(headers.get("WhiteElo"))
        black_elo = _parse_elo(headers.get("BlackElo"))
        if white_elo is not None and white_elo < config.min_rating:
            return FilterResult(False, "rating_too_low")
        if black_elo is not None and black_elo < config.min_rating:
            return FilterResult(False, "rating_too_low")

    return FilterResult(True)


def _estimate_ply_count(game: chess.pgn.Game) -> int:
    """Count mainline half-moves without replaying the board."""
    count = 0
    node = game
    while node.variations:
        node = node.variations[0]
        count += 1
    return count


def _parse_elo(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (ValueError, TypeError):
        return None
