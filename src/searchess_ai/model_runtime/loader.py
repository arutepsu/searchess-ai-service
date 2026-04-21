"""Artifact loading for inference runtime.

Validates artifact completeness and encoder compatibility before constructing
the model. Fails fast and explicitly — partial or mismatched artifacts are
rejected immediately with actionable error messages.

Compatibility rules:
  1. All required manifest fields must be present (no partial artifacts).
  2. Encoder version must exactly match the runtime's ENCODER_VERSION.
  3. Encoder feature_size and move_vocab_size must match.
  4. Model config must contain all required architecture keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from searchess_ai.artifacts.schema import load_manifest, load_model_state
from searchess_ai.model_runtime.encoder import (
    ENCODER_VERSION,
    FEATURE_SIZE,
    MOVE_VOCAB_SIZE,
    compute_encoder_fingerprint,
)

# Model config keys that must be present to reconstruct the network.
_REQUIRED_MODEL_CONFIG_KEYS = {
    "architecture",
    "feature_size",
    "hidden_size",
    "move_vocab_size",
    "num_hidden_layers",
    "dropout",
}

# Encoder config keys that must be present for compatibility checks.
# "fingerprint" catches behavioral drift even when the version string is not bumped.
_REQUIRED_ENCODER_CONFIG_KEYS = {"version", "feature_size", "move_vocab_size", "fingerprint"}


@dataclass
class LoadedModel:
    """Container for a fully loaded, validated model ready for inference."""

    model: object        # PolicyNetwork — typed as object to avoid circular import
    model_version: str
    artifact_id: str
    encoder_version: str


def load_artifact(artifact_dir: Path, device: str = "cpu") -> LoadedModel:
    """Load and validate a trained model artifact from artifact_dir.

    Raises:
      FileNotFoundError  — artifact directory or model.pt missing
      ValueError         — encoder mismatch, model config incomplete, or manifest incomplete
      ImportError        — torch not installed
    """
    try:
        from searchess_ai.training.model import PolicyNetwork
    except ImportError as exc:
        raise ImportError(
            "torch is required for artifact loading. "
            "Install training extras: uv sync --extra training"
        ) from exc

    manifest = load_manifest(artifact_dir)  # already validates manifest completeness

    _validate_encoder_config(manifest.encoder_config)
    _validate_model_config(manifest.model_config)

    model = PolicyNetwork.from_config(manifest.model_config)
    state_dict = load_model_state(artifact_dir, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedModel(
        model=model,
        model_version=manifest.model_version,
        artifact_id=manifest.artifact_id,
        encoder_version=manifest.encoder_config["version"],
    )


def _validate_encoder_config(artifact_enc: dict) -> None:
    """Fail fast if the artifact's encoder differs from the runtime encoder.

    Three checks, each independently reported:
      - presence of required encoder config keys
      - exact encoder version string match
      - numeric parameter match (feature_size, move_vocab_size)

    Rationale: version mismatch means different feature layout or vocabulary,
    which would silently produce wrong predictions without this guard.
    """
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
            "Re-train the model with the current encoder or update the runtime. "
            "Do not bypass this check — a version mismatch means different feature "
            "encoding and will produce wrong move predictions."
        )

    mismatches: list[str] = []
    if artifact_enc["feature_size"] != FEATURE_SIZE:
        mismatches.append(
            f"feature_size: artifact={artifact_enc['feature_size']}, runtime={FEATURE_SIZE}"
        )
    if artifact_enc["move_vocab_size"] != MOVE_VOCAB_SIZE:
        mismatches.append(
            f"move_vocab_size: artifact={artifact_enc['move_vocab_size']}, "
            f"runtime={MOVE_VOCAB_SIZE}"
        )
    if mismatches:
        raise ValueError(
            "Encoder numeric parameter mismatch — training and inference use different "
            "feature representations. Mismatches: " + "; ".join(mismatches)
        )

    # Behavioral fingerprint check — catches encode_fen() drift even when the
    # version string is not bumped.  The fingerprint is a SHA-256 of the sorted
    # non-zero indices on a fixed reference FEN, computed independently at
    # training time and at load time.
    artifact_fingerprint = artifact_enc["fingerprint"]
    current_fingerprint = compute_encoder_fingerprint()
    if artifact_fingerprint != current_fingerprint:
        raise ValueError(
            f"Encoder behavioral fingerprint mismatch.\n"
            f"  Artifact fingerprint : {artifact_fingerprint!r}\n"
            f"  Runtime fingerprint  : {current_fingerprint!r}\n"
            "encode_fen() produced different output than when this artifact was trained.\n"
            "This means the feature vectors the model saw during training differ from\n"
            "what the runtime would produce — predictions would be wrong.\n"
            "Either re-train with the current encoder or revert the encoder change.\n"
            "If the encoder change is intentional, bump ENCODER_VERSION and re-train."
        )


def _validate_model_config(model_config: dict) -> None:
    """Ensure the model config contains all keys needed to reconstruct the network."""
    missing = _REQUIRED_MODEL_CONFIG_KEYS - model_config.keys()
    if missing:
        raise ValueError(
            f"Artifact model_config is missing required keys: {sorted(missing)}. "
            "The artifact may be incomplete or produced by an incompatible training version."
        )
