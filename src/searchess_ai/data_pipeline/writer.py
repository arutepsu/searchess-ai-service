"""Dataset artifact writer.

Writes the prepared dataset as two files:
  <output_dir>/prepared_samples.parquet   — position-level training rows
  <output_dir>/dataset_metadata.json      — provenance, config, counts

The parquet schema is stable across pipeline runs with the same schema_version.
The metadata JSON is the ground truth for reproducibility.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas and pyarrow are required. Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.data_pipeline.extractor import TrainingSample

DATASET_SCHEMA_VERSION = "1.0"
PARQUET_FILENAME = "prepared_samples.parquet"
METADATA_FILENAME = "dataset_metadata.json"


def write_dataset(
    output_dir: Path,
    samples: list[TrainingSample],
    metadata: dict[str, Any],
) -> Path:
    """Serialise samples to parquet and metadata to JSON.

    Returns the path to the written parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / PARQUET_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    _write_parquet(parquet_path, samples)
    _write_metadata(metadata_path, metadata, parquet_path, len(samples))

    return parquet_path


def build_metadata(
    *,
    source_pgn: str,
    source_pgn_sha256: str,
    dataset_name: str,
    pipeline_config: dict[str, Any],
    total_games_seen: int,
    games_accepted: int,
    rejection_reasons: dict[str, int],
    total_samples: int,
    extraction_version: str,
    filter_config_hash: str,
) -> dict[str, Any]:
    """Construct the metadata dict that gets persisted alongside the parquet.

    extraction_version + filter_config_hash together with source_pgn_sha256
    form the complete dataset identity: same inputs → same dataset content.
    """
    return {
        "dataset_id": uuid.uuid4().hex,
        "dataset_name": dataset_name,
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_pgn": source_pgn,
        "source_pgn_sha256": source_pgn_sha256,
        "extraction_version": extraction_version,
        "filter_config_hash": filter_config_hash,
        "pipeline_config": pipeline_config,
        "total_games_seen": total_games_seen,
        "games_accepted": games_accepted,
        "games_rejected": total_games_seen - games_accepted,
        "rejection_reasons": rejection_reasons,
        "total_samples": total_samples,
    }


def _write_parquet(path: Path, samples: list[TrainingSample]) -> None:
    import json as json_mod

    rows = []
    for s in samples:
        rows.append(
            {
                "sample_id": s.sample_id,
                "source_game_id": s.source_game_id,
                "position_fen": s.position_fen,
                "side_to_move": s.side_to_move,
                "played_move_uci": s.played_move_uci,
                # Store as JSON string to keep parquet schema simple.
                "legal_moves_uci": json_mod.dumps(s.legal_moves_uci)
                if s.legal_moves_uci is not None
                else None,
                "ply_index": s.ply_index,
                "game_result": s.game_result,
                "white_rating": s.white_rating,
                "black_rating": s.black_rating,
                "opening": s.opening,
                "time_control": s.time_control,
                "termination": s.termination,
            }
        )

    df = pd.DataFrame(rows)
    df = df.astype(
        {
            "sample_id": "string",
            "source_game_id": "string",
            "position_fen": "string",
            "side_to_move": "string",
            "played_move_uci": "string",
            "ply_index": "int32",
            "game_result": "string",
        }
    )
    df.to_parquet(path, index=False)


def _write_metadata(
    path: Path,
    metadata: dict[str, Any],
    parquet_path: Path,
    num_samples: int,
) -> None:
    metadata["parquet_file"] = parquet_path.name
    metadata["total_samples"] = num_samples
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
