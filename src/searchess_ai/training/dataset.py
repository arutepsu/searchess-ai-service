"""PyTorch Dataset for the move-scoring architecture.

Each sample is a (position, legal_moves, played_move_index) triple:

  pos_features   — float32 (FEATURE_SIZE,)            board encoding
  move_features  — float32 (N, MOVE_FEATURE_SIZE)     all legal moves encoded
  played_idx     — int64 scalar                        index of played move in legal list

N varies per sample (different positions have different legal move counts).
The custom collate function pads within the batch so DataLoader can batch
variable-length samples together.

Splits are performed at game level (not sample level) to prevent positions
from the same game appearing in both training and test sets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

try:
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:
    raise ImportError(
        "torch, numpy, and pandas are required. "
        "Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.data_pipeline.writer import METADATA_FILENAME, PARQUET_FILENAME
from searchess_ai.model_runtime.encoder import encode_fen
from searchess_ai.model_runtime.move_encoder import MOVE_FEATURE_SIZE, encode_move

Split = Literal["train", "val", "test"]


class ChessDataset(Dataset):
    """One sample = (pos_features, move_features, played_idx).

    Requires legal_moves_uci to be present and valid for every sample.
    Raises ValueError at construction time if a required column is missing.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if "legal_moves_uci" not in df.columns:
            raise ValueError(
                "Dataset is missing the 'legal_moves_uci' column. "
                "Re-run the data pipeline to produce a dataset with legal moves."
            )
        self._fens: list[str] = df["position_fen"].tolist()
        self._played: list[str] = df["played_move_uci"].tolist()
        self._legal_raw: list[str] = df["legal_moves_uci"].tolist()

    def __len__(self) -> int:
        return len(self._fens)

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        pos_features = torch.from_numpy(encode_fen(self._fens[idx]))

        legal_ucis = _parse_legal_moves(self._legal_raw[idx])
        played_uci = self._played[idx]

        # Guarantee the played move is in the legal list.
        if played_uci not in legal_ucis:
            legal_ucis = [played_uci] + legal_ucis

        played_idx = legal_ucis.index(played_uci)

        move_features = torch.from_numpy(
            np.stack([encode_move(uci) for uci in legal_ucis])
        )  # (N, MOVE_FEATURE_SIZE)

        return pos_features, move_features, torch.tensor(played_idx, dtype=torch.long)


def move_scoring_collate(
    batch: list[tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Pad variable-length move lists within the batch.

    Returns:
      pos_batch    — (B, FEATURE_SIZE)
      move_batch   — (B, N_max, MOVE_FEATURE_SIZE)  zero-padded
      move_mask    — (B, N_max)  True for real moves, False for padding
      played_batch — (B,)
    """
    pos_batch = torch.stack([item[0] for item in batch])
    played_batch = torch.stack([item[2] for item in batch])

    max_n = max(item[1].shape[0] for item in batch)
    B = len(batch)
    move_batch = torch.zeros(B, max_n, MOVE_FEATURE_SIZE, dtype=torch.float32)
    move_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, (_, moves, _) in enumerate(batch):
        n = moves.shape[0]
        move_batch[i, :n] = moves
        move_mask[i, :n] = True

    return pos_batch, move_batch, move_mask, played_batch


def load_splits(
    dataset_dir: Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple["ChessDataset", "ChessDataset", "ChessDataset"]:
    """Load parquet, split by game_id, return (train, val, test) datasets."""
    parquet_path = dataset_dir / PARQUET_FILENAME
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    game_ids = sorted(df["source_game_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(game_ids)

    n = len(game_ids)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_ids = set(game_ids[:n_train])
    val_ids = set(game_ids[n_train : n_train + n_val])
    test_ids = set(game_ids[n_train + n_val :])

    train_df = df[df["source_game_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["source_game_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["source_game_id"].isin(test_ids)].reset_index(drop=True)

    return ChessDataset(train_df), ChessDataset(val_df), ChessDataset(test_df)


def load_dataset_metadata(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def _parse_legal_moves(raw: str | None) -> list[str]:
    """Parse a JSON-encoded legal move list from the parquet column."""
    if raw is None or (isinstance(raw, float)):
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
