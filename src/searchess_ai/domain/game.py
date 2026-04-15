from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SideToMove(str, Enum):
    WHITE = "white"
    BLACK = "black"


@dataclass(frozen=True, slots=True)
class Position:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("Position value must not be empty.")


@dataclass(frozen=True, slots=True)
class Move:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("Move value must not be empty.")


@dataclass(frozen=True, slots=True)
class LegalMoveSet:
    moves: tuple[Move, ...]

    def __post_init__(self) -> None:
        if not self.moves:
            raise ValueError("LegalMoveSet must contain at least one move.")

    def contains(self, move: Move) -> bool:
        return move in self.moves