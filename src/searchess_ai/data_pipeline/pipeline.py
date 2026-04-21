"""Dataset pipeline orchestrator.

Entry point for the dataset-build mode. Coordinates:
  ingestion → filtering → sample extraction → dataset write

Run as:
  uv run prepare-dataset --pgn <file.pgn> --output-dir <dir> [options]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from searchess_ai.data_pipeline.config import FilterConfig, PipelineConfig
from searchess_ai.data_pipeline.extractor import EXTRACTION_VERSION, TrainingSample, extract_samples
from searchess_ai.data_pipeline.filter import filter_game
from searchess_ai.data_pipeline.ingestion import iter_pgn_games, pgn_sha256
from searchess_ai.data_pipeline.writer import build_metadata, write_dataset


def run_pipeline(config: PipelineConfig) -> dict:
    """Execute the full dataset-preparation pipeline.

    Returns the metadata dict that was persisted alongside the parquet.
    """
    pgn_path = config.source_pgn
    source_sha256 = pgn_sha256(pgn_path)

    all_samples: list[TrainingSample] = []
    total_seen = 0
    total_accepted = 0
    rejection_counts: dict[str, int] = defaultdict(int)

    print(f"[pipeline] Reading PGN: {pgn_path}")

    for game in iter_pgn_games(pgn_path):
        total_seen += 1

        result = filter_game(game, config.filter_config)
        if not result.accepted:
            rejection_counts[result.reason] += 1
            continue

        game_id = f"game_{total_accepted:08d}"
        samples = extract_samples(
            game,
            game_id=game_id,
            include_legal_moves=config.include_legal_moves,
        )
        all_samples.extend(samples)
        total_accepted += 1

        if total_accepted % 1000 == 0:
            print(f"[pipeline] Accepted {total_accepted} games, {len(all_samples)} samples so far")

        if (
            config.filter_config.max_games is not None
            and total_accepted >= config.filter_config.max_games
        ):
            print(f"[pipeline] Reached max_games={config.filter_config.max_games}, stopping.")
            break

    print(
        f"[pipeline] Done. Games: {total_seen} seen, {total_accepted} accepted, "
        f"{total_seen - total_accepted} rejected. Samples: {len(all_samples)}"
    )
    if rejection_counts:
        print(f"[pipeline] Rejection breakdown: {dict(rejection_counts)}")

    metadata = build_metadata(
        source_pgn=str(pgn_path),
        source_pgn_sha256=source_sha256,
        dataset_name=config.dataset_name,
        pipeline_config=config.to_dict(),
        total_games_seen=total_seen,
        games_accepted=total_accepted,
        rejection_reasons=dict(rejection_counts),
        total_samples=len(all_samples),
        extraction_version=EXTRACTION_VERSION,
        filter_config_hash=_filter_config_hash(config.filter_config),
    )

    parquet_path = write_dataset(config.output_dir, all_samples, metadata)
    print(f"[pipeline] Dataset written to: {parquet_path.parent}")

    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a chess training dataset from a PGN file."
    )
    parser.add_argument("--pgn", required=True, help="Path to the input PGN file")
    parser.add_argument("--output-dir", required=True, help="Directory to write dataset files")
    parser.add_argument("--name", default="dataset", help="Dataset name (for metadata)")
    parser.add_argument(
        "--min-ply", type=int, default=10, help="Minimum half-moves to accept a game (default: 10)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Hard cap on accepted games (default: no limit)",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=None,
        help="Minimum player ELO to accept a game (default: no filter)",
    )
    parser.add_argument(
        "--no-legal-moves",
        action="store_true",
        help="Skip legal move computation (faster but disables masked training)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args(argv)

    config = PipelineConfig(
        source_pgn=Path(args.pgn),
        output_dir=Path(args.output_dir),
        dataset_name=args.name,
        filter_config=FilterConfig(
            min_ply_count=args.min_ply,
            max_games=args.max_games,
            min_rating=args.min_rating,
        ),
        include_legal_moves=not args.no_legal_moves,
        random_seed=args.seed,
    )

    run_pipeline(config)


def _filter_config_hash(filter_config) -> str:
    """Short deterministic hash of the filter config for dataset identity."""
    serialized = json.dumps(filter_config.to_dict(), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


if __name__ == "__main__":
    main()
