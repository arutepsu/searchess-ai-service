"""Lichess puzzle CSV adapter.

Completely separate from the PGN-based game ingestion path. This module maps
rows from lichess_db_puzzle.csv into TrainingSample objects for the existing
move-scoring pipeline.

Lichess puzzle CSV column layout
---------------------------------
  PuzzleId      — unique puzzle identifier
  FEN           — board position *before* the opponent's tactical setup move
  Moves         — space-separated UCI moves:
                    moves[0] = opponent's forced setup move
                    moves[1] = first solution move  ← training target
                    moves[2+] = continuation        ← ignored this milestone
  Rating        — puzzle Elo difficulty rating
  RatingDeviation
  Popularity
  NbPlays
  Themes        — space-separated tags (fork, pin, mateIn2, …)
  GameUrl
  OpeningTags

Training mapping
----------------
  position  = FEN with moves[0] applied  (puzzle position, solver to move)
  target    = moves[1]                   (what the solver must find)
  legal_moves = all legal UCI moves from the puzzle position (computed fresh)

Each puzzle produces exactly one TrainingSample.  The puzzle ID becomes the
source_game_id so the training split treats each puzzle independently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import chess
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "python-chess and pandas are required. Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.data_pipeline.extractor import TrainingSample

PUZZLE_ADAPTER_VERSION = "1.0"
PUZZLE_DATASET_TYPE = "lichess_puzzles"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_sample_puzzles(
    csv_path: Path,
    *,
    sample_size: int = 20_000,
    random_seed: int = 42,
) -> tuple[list[TrainingSample], dict[str, Any]]:
    """Read the Lichess puzzle CSV, sample deterministically, and convert rows.

    Args:
        csv_path:    Path to lichess_db_puzzle.csv.
        sample_size: Number of rows to draw.  The full CSV is read first so
                     the total row count can be recorded in metadata.
        random_seed: Fixed seed for reproducible sampling.

    Returns:
        samples: One TrainingSample per valid puzzle row.
        stats:   Ingestion counts for metadata.
    """
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    rows_to_process = df
    if sample_size < total_rows:
        rows_to_process = df.sample(n=sample_size, random_state=random_seed)

    samples: list[TrainingSample] = []
    rejection_reasons: dict[str, int] = {
        "too_few_moves": 0,
        "invalid_fen": 0,
        "illegal_setup_move": 0,
        "illegal_target_move": 0,
    }

    for _, row in rows_to_process.iterrows():
        sample = _convert_row(row, rejection_reasons)
        if sample is not None:
            samples.append(sample)

    stats: dict[str, Any] = {
        "total_rows_in_source": total_rows,
        "rows_sampled": len(rows_to_process),
        "rows_accepted": len(samples),
        "rows_rejected": len(rows_to_process) - len(samples),
        "rejection_reasons": rejection_reasons,
    }
    return samples, stats


def csv_sha256(csv_path: Path) -> str:
    """SHA-256 of the puzzle CSV (dataset identity)."""
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_puzzle_metadata(
    *,
    source_csv: str,
    source_csv_sha256: str,
    dataset_name: str,
    sample_size: int,
    random_seed: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Build the dataset_metadata.json dict for a puzzle dataset.

    Extends the standard format so the training pipeline reads it without
    modification.  Puzzle-specific fields live alongside the common ones.
    """
    sampling_config = {
        "sample_size": sample_size,
        "random_seed": random_seed,
        "puzzle_adapter_version": PUZZLE_ADAPTER_VERSION,
    }
    # Short hash of the sampling config — used as dataset identity alongside
    # the source file hash (mirrors filter_config_hash in PGN pipeline).
    sampling_config_hash = hashlib.sha256(
        json.dumps(sampling_config, sort_keys=True).encode()
    ).hexdigest()[:16]

    return {
        # --- standard fields (expected by training pipeline) ---
        "dataset_id": uuid.uuid4().hex,
        "dataset_name": dataset_name,
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        # extraction_version and filter_config_hash are the PGN pipeline's
        # identity keys; we map the puzzle equivalents here so the training
        # pipeline's dataset_ref is populated meaningfully.
        "extraction_version": PUZZLE_ADAPTER_VERSION,
        "filter_config_hash": sampling_config_hash,

        # --- puzzle-specific fields ---
        "dataset_type": PUZZLE_DATASET_TYPE,
        "source_csv": source_csv,
        "source_csv_sha256": source_csv_sha256,
        "puzzle_adapter_version": PUZZLE_ADAPTER_VERSION,
        "pipeline_config": {
            "source_csv": source_csv,
            "sample_size": sample_size,
            "random_seed": random_seed,
        },

        # --- ingestion counts ---
        "total_rows_in_source": stats["total_rows_in_source"],
        "rows_sampled": stats["rows_sampled"],
        "rows_accepted": stats["rows_accepted"],
        "rows_rejected": stats["rows_rejected"],
        "rejection_reasons": stats["rejection_reasons"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_row(
    row: "pd.Series",
    rejection_reasons: dict[str, int],
) -> TrainingSample | None:
    """Convert one CSV row to a TrainingSample, or None if invalid."""
    puzzle_id = str(row.get("PuzzleId", "")).strip()
    fen = str(row.get("FEN", "")).strip()
    moves_str = str(row.get("Moves", "")).strip()

    moves = moves_str.split()
    if len(moves) < 2:
        rejection_reasons["too_few_moves"] += 1
        return None

    try:
        board = chess.Board(fen)
    except (ValueError, Exception):
        rejection_reasons["invalid_fen"] += 1
        return None

    # Apply the opponent's forced setup move to reach the puzzle position.
    try:
        setup_move = chess.Move.from_uci(moves[0])
        if setup_move not in board.legal_moves:
            rejection_reasons["illegal_setup_move"] += 1
            return None
        board.push(setup_move)
    except (ValueError, AssertionError, Exception):
        rejection_reasons["illegal_setup_move"] += 1
        return None

    puzzle_fen = board.fen()
    side_to_move = "white" if board.turn == chess.WHITE else "black"

    # Validate the first solution move against the puzzle position.
    target_uci = moves[1]
    try:
        target_move = chess.Move.from_uci(target_uci)
        if target_move not in board.legal_moves:
            rejection_reasons["illegal_target_move"] += 1
            return None
    except (ValueError, AssertionError, Exception):
        rejection_reasons["illegal_target_move"] += 1
        return None

    legal_moves = [m.uci() for m in board.legal_moves]

    # Optional metadata: puzzle rating → white_rating, themes → opening.
    rating_raw = row.get("Rating")
    try:
        puzzle_rating: int | None = int(rating_raw) if pd.notna(rating_raw) else None
    except (ValueError, TypeError):
        puzzle_rating = None

    themes_raw = row.get("Themes")
    themes: str | None = str(themes_raw).strip() if pd.notna(themes_raw) else None

    return TrainingSample(
        sample_id=f"puzzle_{puzzle_id}",
        source_game_id=f"puzzle_{puzzle_id}",
        position_fen=puzzle_fen,
        side_to_move=side_to_move,
        played_move_uci=target_uci,
        legal_moves_uci=legal_moves,
        ply_index=0,
        game_result="*",
        white_rating=puzzle_rating,
        black_rating=None,
        opening=themes,
        time_control=None,
        termination=None,
    )
