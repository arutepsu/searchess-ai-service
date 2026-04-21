"""Position-level training sample extraction from a single game.

Each accepted move in a game becomes one TrainingSample. The canonical
representations used here (FEN, UCI) must match what the inference runtime
expects — they flow through unchanged into the prepared dataset.

EXTRACTION_VERSION must be bumped whenever the extraction logic changes in a
way that affects the training samples produced from the same PGN + filter config.
Dataset metadata records this version so that consumers know whether a dataset
needs to be regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump this when extraction logic changes (e.g. new columns, different FEN handling).
EXTRACTION_VERSION: str = "1.0"

try:
    import chess
    import chess.pgn
except ImportError as exc:
    raise ImportError(
        "python-chess is required. Install training extras: uv sync --extra training"
    ) from exc


@dataclass
class TrainingSample:
    sample_id: str
    source_game_id: str
    position_fen: str
    side_to_move: str          # "white" or "black"
    played_move_uci: str
    legal_moves_uci: list[str] | None
    ply_index: int
    game_result: str           # "1-0", "0-1", "1/2-1/2", or "*"
    white_rating: int | None
    black_rating: int | None
    opening: str | None
    time_control: str | None
    termination: str | None


def extract_samples(
    game: chess.pgn.Game,
    game_id: str,
    include_legal_moves: bool = True,
) -> list[TrainingSample]:
    """Return one TrainingSample per mainline ply in game.

    Positions are captured *before* the move is applied (standard convention).
    The game is not replayed twice — the board state is advanced incrementally.
    """
    headers = game.headers
    result = headers.get("Result", "*")
    white_rating = _parse_elo(headers.get("WhiteElo"))
    black_rating = _parse_elo(headers.get("BlackElo"))
    opening = headers.get("Opening") or headers.get("ECO") or None
    time_control = headers.get("TimeControl") or None
    termination = headers.get("Termination") or None

    board = game.board()
    samples: list[TrainingSample] = []
    ply = 0

    for node in game.mainline():
        move = node.move
        fen = board.fen()
        side = "white" if board.turn == chess.WHITE else "black"

        legal_moves: list[str] | None = None
        if include_legal_moves:
            legal_moves = [m.uci() for m in board.legal_moves]

        samples.append(
            TrainingSample(
                sample_id=f"{game_id}_{ply}",
                source_game_id=game_id,
                position_fen=fen,
                side_to_move=side,
                played_move_uci=move.uci(),
                legal_moves_uci=legal_moves,
                ply_index=ply,
                game_result=result,
                white_rating=white_rating,
                black_rating=black_rating,
                opening=opening,
                time_control=time_control,
                termination=termination,
            )
        )
        board.push(move)
        ply += 1

    return samples


def _parse_elo(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (ValueError, TypeError):
        return None
