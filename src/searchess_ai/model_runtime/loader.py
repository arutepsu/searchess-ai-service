"""Artifact loading for the move-scoring inference runtime.

Validates artifact completeness and both encoder compatibilities before
constructing the model. Fails fast and explicitly — partial, mismatched, or
old fixed-vocabulary artifacts are rejected immediately with actionable errors.

Compatibility rules
-------------------
1. All required manifest fields must be present (no partial artifacts).
2. Position encoder version must exactly match ENCODER_VERSION.
3. Position encoder fingerprint must match (catches drift even if version not bumped).
4. Position encoder feature_size must match FEATURE_SIZE.
5. Move encoder version must exactly match MOVE_ENCODER_VERSION.
6. Move encoder fingerprint must match.
7. Move encoder move_feature_size must match MOVE_FEATURE_SIZE.
8. Model config must declare architecture="MoveScoringNetwork" with all required keys.
9. Artifacts using the old architecture ("PolicyNetwork" or missing move_encoder_config)
   are rejected at manifest load time — move_encoder_config is a required manifest field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from searchess_ai.artifacts.schema import load_manifest, load_model_state
from searchess_ai.model_runtime.encoder import (
    ENCODER_VERSION,
    FEATURE_SIZE,
    compute_encoder_fingerprint,
)
from searchess_ai.model_runtime.move_encoder import (
    MOVE_ENCODER_VERSION,
    MOVE_FEATURE_SIZE,
    compute_move_encoder_fingerprint,
)

_REQUIRED_MODEL_CONFIG_KEYS = {
    "architecture",
    "position_feature_size",
    "move_feature_size",
    "hidden_size",
    "num_hidden_layers",
    "dropout",
}

_REQUIRED_ENCODER_CONFIG_KEYS = {"version", "feature_size", "fingerprint"}
_REQUIRED_MOVE_ENCODER_CONFIG_KEYS = {"version", "move_feature_size", "fingerprint"}


@dataclass
class LoadedModel:
    """Container for a fully loaded, validated model ready for inference."""

    model: object        # MoveScoringNetwork — typed as object to avoid circular import
    model_version: str
    artifact_id: str
    encoder_version: str
    move_encoder_version: str


def load_artifact(artifact_dir: Path, device: str = "cpu") -> LoadedModel:
    """Load and validate a trained model artifact from artifact_dir.

    Raises:
      FileNotFoundError  — artifact directory or model.pt missing
      ValueError         — encoder mismatch, model config incomplete, or manifest incomplete
      ImportError        — torch not installed
    """
    try:
        from searchess_ai.training.model import MoveScoringNetwork
    except ImportError as exc:
        raise ImportError(
            "torch is required for artifact loading. "
            "Install training extras: uv sync --extra training"
        ) from exc

    # load_manifest raises ValueError if move_encoder_config is absent,
    # which rejects all old fixed-vocabulary artifacts at this point.
    manifest = load_manifest(artifact_dir)

    _validate_encoder_config(manifest.encoder_config)
    _validate_move_encoder_config(manifest.move_encoder_config)
    _validate_model_config(manifest.model_config)

    model = MoveScoringNetwork.from_config(manifest.model_config)
    state_dict = load_model_state(artifact_dir, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedModel(
        model=model,
        model_version=manifest.model_version,
        artifact_id=manifest.artifact_id,
        encoder_version=manifest.encoder_config["version"],
        move_encoder_version=manifest.move_encoder_config["version"],
    )


def _validate_encoder_config(artifact_enc: dict) -> None:
    """Fail fast if the artifact's position encoder differs from the runtime encoder."""
    missing = _REQUIRED_ENCODER_CONFIG_KEYS - artifact_enc.keys()
    if missing:
        raise ValueError(
            f"Artifact encoder_config is missing required keys: {sorted(missing)}. "
            "The artifact may be corrupt or was produced by an older pipeline version."
        )

    artifact_version = artifact_enc["version"]
    if artifact_version != ENCODER_VERSION:
        raise ValueError(
            f"Encoder version mismatch: artifact={artifact_version!r}, "
            f"runtime={ENCODER_VERSION!r}. "
            "Re-train the model with the current encoder or update the runtime."
        )

    if artifact_enc["feature_size"] != FEATURE_SIZE:
        raise ValueError(
            f"Encoder feature_size mismatch: artifact={artifact_enc['feature_size']}, "
            f"runtime={FEATURE_SIZE}"
        )

    artifact_fingerprint = artifact_enc["fingerprint"]
    current_fingerprint = compute_encoder_fingerprint()
    if artifact_fingerprint != current_fingerprint:
        raise ValueError(
            f"Encoder behavioral fingerprint mismatch.\n"
            f"  Artifact fingerprint : {artifact_fingerprint!r}\n"
            f"  Runtime fingerprint  : {current_fingerprint!r}\n"
            "encode_fen() produced different output than when this artifact was trained.\n"
            "Either re-train with the current encoder or revert the encoder change.\n"
            "If the encoder change is intentional, bump ENCODER_VERSION and re-train."
        )


def _validate_move_encoder_config(artifact_move_enc: dict) -> None:
    """Fail fast if the artifact's move encoder differs from the runtime move encoder."""
    missing = _REQUIRED_MOVE_ENCODER_CONFIG_KEYS - artifact_move_enc.keys()
    if missing:
        raise ValueError(
            f"Artifact move_encoder_config is missing required keys: {sorted(missing)}. "
            "The artifact may be corrupt or was produced by an older pipeline version."
        )

    artifact_version = artifact_move_enc["version"]
    if artifact_version != MOVE_ENCODER_VERSION:
        raise ValueError(
            f"Move encoder version mismatch: artifact={artifact_version!r}, "
            f"runtime={MOVE_ENCODER_VERSION!r}. "
            "Re-train the model with the current move encoder or update the runtime."
        )

    if artifact_move_enc["move_feature_size"] != MOVE_FEATURE_SIZE:
        raise ValueError(
            f"Move encoder move_feature_size mismatch: "
            f"artifact={artifact_move_enc['move_feature_size']}, runtime={MOVE_FEATURE_SIZE}"
        )

    artifact_fingerprint = artifact_move_enc["fingerprint"]
    current_fingerprint = compute_move_encoder_fingerprint()
    if artifact_fingerprint != current_fingerprint:
        raise ValueError(
            f"Move encoder behavioral fingerprint mismatch.\n"
            f"  Artifact fingerprint : {artifact_fingerprint!r}\n"
            f"  Runtime fingerprint  : {current_fingerprint!r}\n"
            "encode_move() produced different output than when this artifact was trained.\n"
            "Either re-train with the current move encoder or revert the change.\n"
            "If the change is intentional, bump MOVE_ENCODER_VERSION and re-train."
        )


def _validate_model_config(model_config: dict) -> None:
    """Ensure the model config describes a MoveScoringNetwork with all required keys."""
    missing = _REQUIRED_MODEL_CONFIG_KEYS - model_config.keys()
    if missing:
        raise ValueError(
            f"Artifact model_config is missing required keys: {sorted(missing)}. "
            "The artifact may be incomplete or produced by an incompatible training version."
        )

    architecture = model_config.get("architecture", "")
    if architecture != "MoveScoringNetwork":
        raise ValueError(
            f"Artifact model_config declares architecture={architecture!r}. "
            "Expected 'MoveScoringNetwork'. "
            "Old fixed-vocabulary artifacts (e.g. PolicyNetwork) are not supported. "
            "Re-train with the current pipeline to produce a compatible artifact."
        )
