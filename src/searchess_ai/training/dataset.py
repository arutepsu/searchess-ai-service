"""PyTorch Dataset for the prepared chess parquet dataset.

Splits are performed at game level (not sample level) to prevent positions
from the same game appearing in both training and test sets. This avoids
trivial continuation-memorisation and gives honest evaluation numbers.
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
from searchess_ai.model_runtime.encoder import MOVE_VOCAB_SIZE, encode_fen, encode_uci_move

Split = Literal["train", "val", "test"]


class ChessDataset(Dataset):
    """One sample = (features, label, legal_mask).

    features:    float32 tensor (FEATURE_SIZE,)
    label:       int64 scalar — vocabulary index of the played move
    legal_mask:  float32 tensor (MOVE_VOCAB_SIZE,) — 1.0 where legal, 0.0 elsewhere
                 All-ones when legal_moves_uci is absent for a sample.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._fens: list[str] = df["position_fen"].tolist()
        self._moves: list[str] = df["played_move_uci"].tolist()
        legal_col = df["legal_moves_uci"].tolist() if "legal_moves_uci" in df.columns else []
        self._legal: list[str | None] = legal_col if legal_col else [None] * len(df)

    def __len__(self) -> int:
        return len(self._fens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(encode_fen(self._fens[idx]))
        label = torch.tensor(encode_uci_move(self._moves[idx]), dtype=torch.long)
        mask = _build_mask(self._legal[idx])
        return features, label, mask


def load_splits(
    dataset_dir: Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[ChessDataset, ChessDataset, ChessDataset]:
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
    import json

    metadata_path = dataset_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def _build_mask(legal_json: str | None) -> torch.Tensor:
    """Build a legal-move binary mask tensor of shape (MOVE_VOCAB_SIZE,)."""
    if legal_json is None or (isinstance(legal_json, float)):
        # pandas NaN arrives as float; treat as unknown → all-legal mask
        return torch.ones(MOVE_VOCAB_SIZE, dtype=torch.float32)
    try:
        moves: list[str] = json.loads(legal_json)
    except (json.JSONDecodeError, TypeError):
        return torch.ones(MOVE_VOCAB_SIZE, dtype=torch.float32)

    mask = torch.zeros(MOVE_VOCAB_SIZE, dtype=torch.float32)
    for uci in moves:
        try:
            mask[encode_uci_move(uci)] = 1.0
        except ValueError:
            pass
    # If mask is all-zeros (shouldn't happen in a valid position), fall back to all-legal.
    if mask.sum() == 0:
        mask.fill_(1.0)
    return mask
