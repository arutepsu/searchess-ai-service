"""Puzzle dataset pipeline — ingestion and dataset preparation.

Entry point for building a sampled Lichess puzzle dataset for training.

Run as:
  uv run prepare-puzzle-dataset \\
      --csv train_data/lichess_db_puzzle.csv \\
      --output-dir datasets/puzzles_20k \\
      [--name lichess_puzzles_20k] \\
      [--sample-size 20000] \\
      [--seed 42]

Output (same format as the PGN pipeline — trainable with existing train-model CLI):
  <output-dir>/prepared_samples.parquet
  <output-dir>/dataset_metadata.json

Design notes
------------
- Sampling is deterministic: same --csv + --sample-size + --seed always
  produces the same dataset (barring source file changes).
- Legal moves are computed fresh from the puzzle position via python-chess.
  This is correct and necessary: the CSV's Moves field is the solution only.
- Puzzle metadata (rating, themes) is preserved in the parquet columns
  white_rating and opening respectively.
- This pipeline never touches the PGN ingestion path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from searchess_ai.data_pipeline.puzzle_ingestion import (
    PUZZLE_ADAPTER_VERSION,
    build_puzzle_metadata,
    csv_sha256,
    load_and_sample_puzzles,
)
from searchess_ai.data_pipeline.writer import write_dataset

_DEFAULT_SAMPLE_SIZE = 20_000
_DEFAULT_SEED = 42


def run_puzzle_pipeline(
    csv_path: Path,
    output_dir: Path,
    dataset_name: str,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    random_seed: int = _DEFAULT_SEED,
) -> dict:
    """Execute the puzzle dataset-preparation pipeline.

    Returns the metadata dict written alongside the parquet file.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Puzzle CSV not found: {csv_path}")

    print(f"[puzzle-pipeline] Source CSV : {csv_path}")
    print(f"[puzzle-pipeline] Sample size: {sample_size:,}  seed={random_seed}")

    print("[puzzle-pipeline] Computing source file hash …")
    source_hash = csv_sha256(csv_path)
    print(f"[puzzle-pipeline] SHA-256    : {source_hash[:16]}…")

    print("[puzzle-pipeline] Loading and sampling CSV …")
    samples, stats = load_and_sample_puzzles(
        csv_path,
        sample_size=sample_size,
        random_seed=random_seed,
    )

    print(
        f"[puzzle-pipeline] Rows: {stats['total_rows_in_source']:,} total, "
        f"{stats['rows_sampled']:,} sampled, "
        f"{stats['rows_accepted']:,} accepted, "
        f"{stats['rows_rejected']:,} rejected"
    )
    if stats["rejection_reasons"]:
        for reason, count in stats["rejection_reasons"].items():
            if count:
                print(f"[puzzle-pipeline]   rejected ({reason}): {count:,}")

    if not samples:
        print("[puzzle-pipeline] ERROR: no valid samples produced. Aborting.", file=sys.stderr)
        sys.exit(1)

    metadata = build_puzzle_metadata(
        source_csv=str(csv_path),
        source_csv_sha256=source_hash,
        dataset_name=dataset_name,
        sample_size=sample_size,
        random_seed=random_seed,
        stats=stats,
    )

    parquet_path = write_dataset(output_dir, samples, metadata)
    print(f"[puzzle-pipeline] Dataset written to: {parquet_path.parent}")
    print(f"[puzzle-pipeline] Total samples     : {len(samples):,}")
    print(f"[puzzle-pipeline] Adapter version   : {PUZZLE_ADAPTER_VERSION}")

    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a chess training dataset from the Lichess puzzle CSV. "
            "Output is a parquet + metadata pair compatible with train-model."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to lichess_db_puzzle.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write prepared_samples.parquet and dataset_metadata.json",
    )
    parser.add_argument(
        "--name",
        default="lichess_puzzles",
        help="Dataset name recorded in metadata (default: lichess_puzzles)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help=f"Number of puzzle rows to sample (default: {_DEFAULT_SAMPLE_SIZE:,})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Random seed for reproducible sampling (default: {_DEFAULT_SEED})",
    )

    args = parser.parse_args(argv)

    run_puzzle_pipeline(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
        dataset_name=args.name,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
