"""Training run configuration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TrainingConfig:
    """All hyperparameters and paths for a single training run.

    Persisted verbatim in the artifact's training_run.json so that runs are
    fully reproducible given the same dataset and config.
    """

    dataset_dir: Path
    output_dir: Path

    # Stable run identifier — used to name the artifact directory.
    run_id: str = field(
        default_factory=lambda: (
            f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
    )

    # Architecture
    hidden_size: int = 512
    num_hidden_layers: int = 2
    dropout: float = 0.2

    # Optimiser
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 10

    # Split fractions (game-level, not sample-level, to avoid data leakage).
    # test fraction = 1 - train_split - val_split
    train_split: float = 0.8
    val_split: float = 0.1

    random_seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset_dir": str(self.dataset_dir),
            "output_dir": str(self.output_dir),
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "random_seed": self.random_seed,
            "device": self.device,
        }
