"""Small-scale streamed full-game dataset builder.

This path is intentionally separate from puzzle ingestion. It reads a local PGN
file lazily, extracts position -> played move samples from standard chess games,
and stops as soon as configured bounds are reached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import chess
    import chess.pgn
except ImportError as exc:
    raise ImportError(
        "python-chess is required. Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.data_pipeline.config import FilterConfig
from searchess_ai.data_pipeline.extractor import EXTRACTION_VERSION, TrainingSample
from searchess_ai.data_pipeline.filter import filter_game
from searchess_ai.data_pipeline.ingestion import iter_pgn_games, pgn_sha256
from searchess_ai.data_pipeline.writer import build_metadata, write_dataset

FULL_GAME_EXTRACTION_VERSION = f"{EXTRACTION_VERSION}+full-game-small-v1"


@dataclass(frozen=True, slots=True)
class FullGameDatasetConfig:
    """Configuration for one bounded local PGN-to-dataset build."""

    source_pgn: Path
    output_dir: Path
    dataset_name: str = "lichess_full_game_small"
    max_games: int | None = 2_000
    max_samples: int = 20_000
    min_game_ply_count: int = 20
    min_sample_ply_index: int = 8
    min_rating: int | None = None
    include_legal_moves: bool = True
    random_seed: int = 42

    def to_dict(self) -> dict:
        return {
            "source_pgn": str(self.source_pgn),
            "output_dir": str(self.output_dir),
            "dataset_name": self.dataset_name,
            "max_games": self.max_games,
            "max_samples": self.max_samples,
            "min_game_ply_count": self.min_game_ply_count,
            "min_sample_ply_index": self.min_sample_ply_index,
            "min_rating": self.min_rating,
            "include_legal_moves": self.include_legal_moves,
            "random_seed": self.random_seed,
            "standard_chess_only": True,
        }


def run_full_game_pipeline(config: FullGameDatasetConfig) -> dict:
    """Stream local PGN games into a bounded full-game supervised dataset."""

    if config.max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")
    if config.max_games is not None and config.max_games <= 0:
        raise ValueError("max_games must be greater than zero when provided")
    if config.min_sample_ply_index < 0:
        raise ValueError("min_sample_ply_index must be >= 0")

    filter_config = FilterConfig(
        min_ply_count=config.min_game_ply_count,
        require_result=True,
        skip_variants=True,
        min_rating=config.min_rating,
        max_games=config.max_games,
    )

    source_sha256 = pgn_sha256(config.source_pgn)
    samples: list[TrainingSample] = []
    total_seen = 0
    total_accepted = 0
    malformed_games = 0
    rejection_counts: dict[str, int] = defaultdict(int)

    print(f"[full-game-pipeline] Reading PGN: {config.source_pgn}")

    for game in iter_pgn_games(config.source_pgn):
        total_seen += 1

        result = filter_game(game, filter_config)
        if not result.accepted:
            rejection_counts[result.reason] += 1
            continue

        game_id = f"full_game_{total_accepted:08d}"
        try:
            game_samples = extract_full_game_samples(
                game,
                game_id=game_id,
                include_legal_moves=config.include_legal_moves,
                min_sample_ply_index=config.min_sample_ply_index,
                max_remaining_samples=config.max_samples - len(samples),
            )
        except Exception:
            malformed_games += 1
            rejection_counts["malformed"] += 1
            continue

        if game_samples:
            samples.extend(game_samples)
            total_accepted += 1

        if total_accepted and total_accepted % 100 == 0:
            print(
                "[full-game-pipeline] "
                f"Accepted {total_accepted} games, {len(samples)} samples so far"
            )

        if len(samples) >= config.max_samples:
            print(
                f"[full-game-pipeline] Reached max_samples={config.max_samples}, stopping."
            )
            break
        if config.max_games is not None and total_accepted >= config.max_games:
            print(f"[full-game-pipeline] Reached max_games={config.max_games}, stopping.")
            break

    metadata = build_metadata(
        source_pgn=str(config.source_pgn),
        source_pgn_sha256=source_sha256,
        dataset_name=config.dataset_name,
        pipeline_config=config.to_dict(),
        total_games_seen=total_seen,
        games_accepted=total_accepted,
        rejection_reasons=dict(rejection_counts),
        total_samples=len(samples),
        extraction_version=FULL_GAME_EXTRACTION_VERSION,
        filter_config_hash=_config_hash(config),
    )
    metadata["malformed_games"] = malformed_games
    metadata["notes"] = (
        "Small streamed full-game dataset. Stops at max_samples/max_games; "
        "not intended for full monthly archive processing."
    )

    parquet_path = write_dataset(config.output_dir, samples, metadata)
    print(
        f"[full-game-pipeline] Done. Games: {total_seen} seen, "
        f"{total_accepted} accepted. Samples: {len(samples)}"
    )
    if rejection_counts:
        print(f"[full-game-pipeline] Rejection breakdown: {dict(rejection_counts)}")
    print(f"[full-game-pipeline] Dataset written to: {parquet_path.parent}")

    return metadata


def extract_full_game_samples(
    game: chess.pgn.Game,
    *,
    game_id: str,
    include_legal_moves: bool,
    min_sample_ply_index: int,
    max_remaining_samples: int,
) -> list[TrainingSample]:
    """Extract bounded position samples from one already-filtered game."""

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
        if len(samples) >= max_remaining_samples:
            break

        move = node.move
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move {move.uci()} at ply {ply}")

        if ply >= min_sample_ply_index:
            legal_moves = None
            if include_legal_moves:
                legal_moves = [legal_move.uci() for legal_move in board.legal_moves]

            samples.append(
                TrainingSample(
                    sample_id=f"{game_id}_{ply}",
                    source_game_id=game_id,
                    position_fen=board.fen(),
                    side_to_move="white" if board.turn == chess.WHITE else "black",
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a small bounded full-game dataset from a local PGN file."
    )
    parser.add_argument("--pgn", required=True, help="Path to the local PGN file")
    parser.add_argument("--output-dir", required=True, help="Directory to write dataset files")
    parser.add_argument("--name", default="lichess_full_game_small", help="Dataset name")
    parser.add_argument("--max-games", type=int, default=2_000)
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--min-game-ply", type=int, default=20)
    parser.add_argument("--min-sample-ply", type=int, default=8)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument(
        "--no-legal-moves",
        action="store_true",
        help="Skip legal move computation; not recommended for move-scoring training.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    config = FullGameDatasetConfig(
        source_pgn=Path(args.pgn),
        output_dir=Path(args.output_dir),
        dataset_name=args.name,
        max_games=args.max_games,
        max_samples=args.max_samples,
        min_game_ply_count=args.min_game_ply,
        min_sample_ply_index=args.min_sample_ply,
        min_rating=args.min_rating,
        include_legal_moves=not args.no_legal_moves,
        random_seed=args.seed,
    )
    run_full_game_pipeline(config)


def _config_hash(config: FullGameDatasetConfig) -> str:
    serialized = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _parse_elo(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
