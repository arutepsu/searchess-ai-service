"""Training pipeline orchestrator.

Coordinates: dataset loading → split → train → evaluate → save artifact.

Run as:
  uv run train-model --dataset-dir <dir> --output-dir <dir> [options]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError(
        "torch and numpy are required. Install training extras: uv sync --extra training"
    ) from exc

from searchess_ai.artifacts.schema import ArtifactManifest, save_artifact
from searchess_ai.evaluation.report import compute_report
from searchess_ai.model_runtime.encoder import encoder_config
from searchess_ai.training.config import TrainingConfig
from searchess_ai.training.dataset import load_dataset_metadata, load_splits
from searchess_ai.training.model import PolicyNetwork
from searchess_ai.training.trainer import evaluate_epoch, train_epoch


def run_training(config: TrainingConfig) -> Path:
    """Execute the full training pipeline.

    Returns the path to the saved artifact directory.
    """
    _seed_everything(config.random_seed)
    device = torch.device(config.device)

    print(f"[training] Run ID: {config.run_id}")
    print(f"[training] Device: {device}")

    # --- Dataset ---
    dataset_meta = load_dataset_metadata(config.dataset_dir)
    dataset_version = dataset_meta.get("dataset_id", "unknown")
    print(f"[training] Dataset version: {dataset_version}")

    train_ds, val_ds, test_ds = load_splits(
        config.dataset_dir,
        train_split=config.train_split,
        val_split=config.val_split,
        seed=config.random_seed,
    )
    print(
        f"[training] Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # --- Model ---
    model = PolicyNetwork(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        dropout=config.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[training] Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_state = None
    history: list[dict] = []

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, device)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[training] Epoch {epoch}/{config.num_epochs} — "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train top1: {train_metrics['top1']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f}, "
            f"val top1: {val_metrics['top1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint before final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final evaluation on test set ---
    test_metrics = evaluate_epoch(model, test_loader, device)
    print(
        f"[training] Test — loss: {test_metrics['loss']:.4f}, "
        f"top1: {test_metrics['top1']:.4f}, "
        f"top5: {test_metrics['top5']:.4f}"
    )

    # --- Artifact identity (known before weights are finalised) ---
    model_version = f"v{config.run_id}"
    enc_cfg = encoder_config()

    # --- Evaluation report ---
    report = compute_report(
        model=model,
        test_loader=test_loader,
        device=device,
        dataset_sizes={
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        dataset_id=dataset_version,
        model_version=model_version,
        encoder_version=enc_cfg["version"],
    )

    # --- Artifact ---
    artifact_dir = config.output_dir / config.run_id

    manifest = ArtifactManifest.new(
        model_version=model_version,
        dataset_version=dataset_version,
        training_run_id=config.run_id,
        model_config=model.config(),
        encoder_config=enc_cfg,
        dataset_ref={
            "dataset_id": dataset_version,
            "dataset_dir": str(config.dataset_dir),
            "dataset_name": dataset_meta.get("dataset_name", "unknown"),
            "source_pgn_sha256": dataset_meta.get("source_pgn_sha256", "unknown"),
            "extraction_version": dataset_meta.get("extraction_version", "unknown"),
            "filter_config_hash": dataset_meta.get("filter_config_hash", "unknown"),
            "total_samples": dataset_meta.get("total_samples", -1),
        },
        training_run=config.to_dict(),
        evaluation_summary=report,
    )

    save_artifact(artifact_dir, model.state_dict(), manifest)
    print(f"[training] Artifact saved to: {artifact_dir}")
    return artifact_dir


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train baseline supervised chess policy model.")
    parser.add_argument("--dataset-dir", required=True, help="Prepared dataset directory")
    parser.add_argument("--output-dir", required=True, help="Directory to write artifact")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    config = TrainingConfig(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        random_seed=args.seed,
    )

    run_training(config)


if __name__ == "__main__":
    main()
