"""Artifact versioning, save, and load contracts.

An artifact is a self-contained directory that packages:
  - model weights
  - model architecture config
  - encoder config (must match training-time encoder)
  - dataset provenance reference
  - training run metadata
  - offline evaluation summary
  - a top-level manifest tying everything together

Directory layout:
  <artifact_dir>/
    manifest.json
    model.pt
    model_config.json
    encoder_config.json
    dataset_ref.json
    training_run.json
    evaluation.json

Completeness rule:
  load_manifest() raises ValueError if any required field is absent.
  Partial artifacts are not accepted — all fields must be present.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = "1.0"

# All fields that must be present in a valid manifest.
# Adding a field here causes load_manifest to reject old artifacts missing it.
REQUIRED_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {
        "artifact_id",
        "model_version",
        "schema_version",
        "created_at",
        "dataset_version",
        "training_run_id",
        "model_config",
        "encoder_config",
        "dataset_ref",
        "training_run",
        "evaluation_summary",
    }
)


@dataclass
class ArtifactManifest:
    artifact_id: str
    model_version: str
    schema_version: str
    created_at: str
    dataset_version: str
    training_run_id: str
    model_config: dict[str, Any]
    encoder_config: dict[str, Any]
    dataset_ref: dict[str, Any]
    training_run: dict[str, Any]
    evaluation_summary: dict[str, Any]

    @staticmethod
    def new(
        *,
        model_version: str,
        dataset_version: str,
        training_run_id: str,
        model_config: dict[str, Any],
        encoder_config: dict[str, Any],
        dataset_ref: dict[str, Any],
        training_run: dict[str, Any],
        evaluation_summary: dict[str, Any],
    ) -> "ArtifactManifest":
        return ArtifactManifest(
            artifact_id=uuid.uuid4().hex,
            model_version=model_version,
            schema_version=ARTIFACT_SCHEMA_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_version=dataset_version,
            training_run_id=training_run_id,
            model_config=model_config,
            encoder_config=encoder_config,
            dataset_ref=dataset_ref,
            training_run=training_run,
            evaluation_summary=evaluation_summary,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArtifactManifest":
        missing = REQUIRED_MANIFEST_FIELDS - d.keys()
        if missing:
            raise ValueError(
                f"Incomplete artifact manifest — missing required fields: {sorted(missing)}. "
                "This artifact may be corrupt, partially written, or produced by an "
                "incompatible version of the pipeline. Do not attempt to load it."
            )
        # Pass only known fields to avoid unexpected-keyword errors on schema upgrades.
        known = {k: d[k] for k in REQUIRED_MANIFEST_FIELDS}
        return ArtifactManifest(**known)


def save_artifact(
    artifact_dir: Path,
    model_state_dict: Any,
    manifest: ArtifactManifest,
) -> None:
    """Write all artifact files to artifact_dir (created if absent)."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for saving artifacts. "
            "Install training extras: uv sync --extra training"
        ) from exc

    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model_state_dict, artifact_dir / "model.pt")

    _write_json(artifact_dir / "model_config.json", manifest.model_config)
    _write_json(artifact_dir / "encoder_config.json", manifest.encoder_config)
    _write_json(artifact_dir / "dataset_ref.json", manifest.dataset_ref)
    _write_json(artifact_dir / "training_run.json", manifest.training_run)
    _write_json(artifact_dir / "evaluation.json", manifest.evaluation_summary)
    # Write manifest last so its presence signals a complete artifact.
    _write_json(artifact_dir / "manifest.json", manifest.to_dict())


def load_manifest(artifact_dir: Path) -> ArtifactManifest:
    """Load and validate the artifact manifest from artifact_dir.

    Raises FileNotFoundError if manifest.json is absent.
    Raises ValueError if required fields are missing (partial artifact).
    """
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in artifact directory: {artifact_dir}. "
            "Either the path is wrong or the artifact was not saved completely."
        )
    with open(manifest_path, encoding="utf-8") as f:
        raw = json.load(f)
    return ArtifactManifest.from_dict(raw)


def load_model_state(artifact_dir: Path, map_location: str = "cpu") -> Any:
    """Load PyTorch state dict from artifact_dir."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for loading model artifacts. "
            "Install training extras: uv sync --extra training"
        ) from exc

    model_path = artifact_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model.pt in artifact directory: {artifact_dir}. "
            "The artifact may be incomplete."
        )
    return torch.load(model_path, map_location=map_location, weights_only=True)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
