"""Configuration dataclasses for the dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterConfig:
    """Controls which games are accepted into the training dataset.

    These rules are persisted in dataset_metadata.json so that future
    consumers know exactly what filtering was applied.
    """

    # Skip games shorter than this many half-moves (plies).
    # Ultra-short games are usually forfeitures, errors, or garbage.
    min_ply_count: int = 10

    # Skip games that do not carry a decisive or drawn result.
    require_result: bool = True

    # Skip games with a Variant header (chess960, crazyhouse, etc.).
    skip_variants: bool = True

    # Skip games where either player has an ELO below this threshold.
    # None means no rating filter is applied.
    min_rating: int | None = None

    # Hard cap on accepted games. Useful for quick local smoke tests.
    # None means no cap.
    max_games: int | None = None

    def to_dict(self) -> dict:
        return {
            "min_ply_count": self.min_ply_count,
            "require_result": self.require_result,
            "skip_variants": self.skip_variants,
            "min_rating": self.min_rating,
            "max_games": self.max_games,
        }


@dataclass
class PipelineConfig:
    """Top-level configuration for a single dataset-preparation run."""

    source_pgn: Path
    output_dir: Path
    dataset_name: str

    filter_config: FilterConfig = field(default_factory=FilterConfig)

    # Whether to compute and store legal moves for every position.
    # Adds significant per-sample work but enables masked cross-entropy training.
    include_legal_moves: bool = True

    # Seed for any randomness (currently unused but reserved for future sampling).
    random_seed: int = 42

    def to_dict(self) -> dict:
        return {
            "source_pgn": str(self.source_pgn),
            "output_dir": str(self.output_dir),
            "dataset_name": self.dataset_name,
            "filter_config": self.filter_config.to_dict(),
            "include_legal_moves": self.include_legal_moves,
            "random_seed": self.random_seed,
        }
