"""Unit tests for the pure MoveDto ↔ UCI move string mapper."""
import pytest

from searchess_ai.api.dto.inference import MoveDtoIn, MoveDtoOut
from searchess_ai.api.dto.move_mapper import move_dto_to_uci, uci_to_move_dto


# ---------------------------------------------------------------------------
# move_dto_to_uci — MoveDto → UCI string
# ---------------------------------------------------------------------------

def test_quiet_move_to_uci() -> None:
    dto = MoveDtoIn(**{"from": "e2", "to": "e4"})
    assert move_dto_to_uci(dto) == "e2e4"


def test_another_quiet_move_to_uci() -> None:
    dto = MoveDtoIn(**{"from": "g1", "to": "f3"})
    assert move_dto_to_uci(dto) == "g1f3"


def test_promotion_queen_to_uci() -> None:
    dto = MoveDtoIn(**{"from": "e7", "to": "e8", "promotion": "queen"})
    assert move_dto_to_uci(dto) == "e7e8q"


def test_promotion_rook_to_uci() -> None:
    dto = MoveDtoIn(**{"from": "a7", "to": "a8", "promotion": "rook"})
    assert move_dto_to_uci(dto) == "a7a8r"


def test_promotion_bishop_to_uci() -> None:
    dto = MoveDtoIn(**{"from": "b7", "to": "b8", "promotion": "bishop"})
    assert move_dto_to_uci(dto) == "b7b8b"


def test_promotion_knight_to_uci() -> None:
    dto = MoveDtoIn(**{"from": "c7", "to": "c8", "promotion": "knight"})
    assert move_dto_to_uci(dto) == "c7c8n"


def test_no_promotion_suffix_when_promotion_is_none() -> None:
    dto = MoveDtoIn(**{"from": "d2", "to": "d4"})
    assert move_dto_to_uci(dto) == "d2d4"
    assert len(move_dto_to_uci(dto)) == 4


# ---------------------------------------------------------------------------
# uci_to_move_dto — UCI string → MoveDto
# ---------------------------------------------------------------------------

def test_quiet_uci_to_dto() -> None:
    dto = uci_to_move_dto("e2e4")
    assert dto.from_square == "e2"
    assert dto.to == "e4"
    assert dto.promotion is None


def test_another_quiet_uci_to_dto() -> None:
    dto = uci_to_move_dto("g1f3")
    assert dto.from_square == "g1"
    assert dto.to == "f3"
    assert dto.promotion is None


def test_promotion_queen_uci_to_dto() -> None:
    dto = uci_to_move_dto("e7e8q")
    assert dto.from_square == "e7"
    assert dto.to == "e8"
    assert dto.promotion == "queen"


def test_promotion_rook_uci_to_dto() -> None:
    dto = uci_to_move_dto("a7a8r")
    assert dto.promotion == "rook"


def test_promotion_bishop_uci_to_dto() -> None:
    dto = uci_to_move_dto("b7b8b")
    assert dto.promotion == "bishop"


def test_promotion_knight_uci_to_dto() -> None:
    dto = uci_to_move_dto("c7c8n")
    assert dto.promotion == "knight"


def test_uci_to_dto_serializes_from_as_json_key() -> None:
    dto = uci_to_move_dto("e2e4")
    serialized = dto.model_dump(by_alias=True)
    assert "from" in serialized
    assert "from_square" not in serialized


# ---------------------------------------------------------------------------
# Error cases for uci_to_move_dto
# ---------------------------------------------------------------------------

def test_too_short_uci_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid UCI"):
        uci_to_move_dto("e2e")


def test_too_long_uci_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid UCI"):
        uci_to_move_dto("e2e4qq")


def test_unknown_promotion_char_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown promotion character"):
        uci_to_move_dto("e7e8x")


# ---------------------------------------------------------------------------
# Round-trip: move_dto_to_uci → uci_to_move_dto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("from_sq,to_sq,promo", [
    ("e2", "e4", None),
    ("g1", "f3", None),
    ("e7", "e8", "queen"),
    ("a7", "a8", "rook"),
    ("b7", "b8", "bishop"),
    ("c7", "c8", "knight"),
])
def test_round_trip(from_sq: str, to_sq: str, promo: str | None) -> None:
    dto_in = MoveDtoIn(**{"from": from_sq, "to": to_sq, "promotion": promo})
    uci = move_dto_to_uci(dto_in)
    dto_out = uci_to_move_dto(uci)
    assert dto_out.from_square == from_sq
    assert dto_out.to == to_sq
    assert dto_out.promotion == promo
