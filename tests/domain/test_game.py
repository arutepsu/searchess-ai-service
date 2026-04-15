import pytest

from searchess_ai.domain.game import LegalMoveSet, Move


def test_move_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        Move("")


def test_move_rejects_whitespace_only() -> None:
    with pytest.raises(ValueError):
        Move("   ")


def test_legal_move_set_rejects_empty_tuple() -> None:
    with pytest.raises(ValueError):
        LegalMoveSet(())


def test_legal_move_set_contains_returns_true_for_member() -> None:
    move = Move("e2e4")
    assert LegalMoveSet((move,)).contains(move)


def test_legal_move_set_contains_returns_false_for_non_member() -> None:
    assert not LegalMoveSet((Move("e2e4"),)).contains(Move("d2d4"))
