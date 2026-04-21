"""Serving state — tracks which model artifact is actively handling inference.

Responsibilities:
  - loads and holds the active model artifact
  - supports atomic reload (old model stays active until new one is ready)
  - manages a validated-before-promotion candidate slot
  - exposes model metadata for observability and response enrichment

Candidate promotion contract:
  - A candidate is always validated (full load + all compatibility checks) before
    it is accepted into the candidate slot.
  - An invalid candidate cannot be promoted.  The candidate slot stores both the
    validation outcome and the pre-loaded runtime so promotion is an atomic swap.
  - promote_candidate() fails explicitly if no valid candidate exists.

This is intentionally NOT a full model registry.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Identity snapshot of a loaded model artifact."""

    model_version: str
    artifact_id: str
    encoder_version: str
    artifact_dir: str
    loaded_at: str

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "artifact_id": self.artifact_id,
            "encoder_version": self.encoder_version,
            "artifact_dir": self.artifact_dir,
            "loaded_at": self.loaded_at,
        }


@dataclass(frozen=True, slots=True)
class CandidateStatus:
    """Result of validating a candidate artifact.

    is_valid=True  → the candidate passed all compatibility checks and is safe
                     to promote.  metadata is populated.
    is_valid=False → validation failed.  validation_error contains the reason.
                     The candidate MUST NOT be promoted.
    """

    candidate_dir: Path
    is_valid: bool
    validated_at: str
    validation_error: str | None
    metadata: ModelMetadata | None

    def to_dict(self) -> dict:
        return {
            "candidate_dir": str(self.candidate_dir),
            "is_valid": self.is_valid,
            "validated_at": self.validated_at,
            "validation_error": self.validation_error,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


class ServingState:
    """Manages the lifecycle of the active model for inference.

    Thread-safety contract:
      - reads of _runtime and _metadata are safe without a lock
        (Python GIL makes single object-reference assignment atomic)
      - all writes (reload, promote_candidate) acquire _lock so no caller
        sees a half-swapped state
    """

    def __init__(self, artifact_dir: Path, device: str = "cpu") -> None:
        self._lock = threading.Lock()
        self._artifact_dir = artifact_dir
        self._device = device
        self._runtime = None
        self._metadata: ModelMetadata | None = None
        self._load_error: str | None = None

        # Candidate slot — holds a pre-loaded, validated model waiting for promotion.
        self._candidate_status: CandidateStatus | None = None
        self._candidate_runtime = None  # pre-loaded, ready for zero-cost swap

        try:
            self._load(artifact_dir)
        except Exception as exc:
            self._load_error = str(exc)

    # ------------------------------------------------------------------
    # Current model
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def runtime(self):
        """The active SupervisedModelRuntime, or None if load failed."""
        return self._runtime

    @property
    def metadata(self) -> ModelMetadata | None:
        """Identity of the active model artifact."""
        return self._metadata

    @property
    def load_error(self) -> str | None:
        """Load-time error message, or None if model is loaded successfully."""
        return self._load_error

    def reload(self, new_artifact_dir: Path | None = None) -> ModelMetadata:
        """Reload the active model atomically.

        On failure the existing model stays active; the error is re-raised.
        """
        target = new_artifact_dir or self._artifact_dir
        new_runtime, new_metadata = _load_runtime_and_metadata(target, self._device)
        with self._lock:
            self._runtime = new_runtime
            self._metadata = new_metadata
            self._load_error = None
            self._artifact_dir = target
        return new_metadata

    # ------------------------------------------------------------------
    # Candidate validation and promotion
    # ------------------------------------------------------------------

    @property
    def candidate_status(self) -> CandidateStatus | None:
        """Validation status of the registered candidate, or None."""
        return self._candidate_status

    def register_and_validate_candidate(self, candidate_dir: Path) -> CandidateStatus:
        """Fully validate a candidate artifact and register it for promotion.

        Performs the same load-time validation as the active model:
          - manifest completeness check
          - encoder version + fingerprint check
          - model config check
          - weights load

        The pre-loaded model is retained in memory so promote_candidate()
        is an atomic in-memory swap (no disk I/O at promotion time).

        Returns the CandidateStatus regardless of outcome — caller decides
        whether to proceed based on status.is_valid.
        """
        validated_at = datetime.now(timezone.utc).isoformat()
        try:
            new_runtime, new_metadata = _load_runtime_and_metadata(
                candidate_dir, self._device
            )
            status = CandidateStatus(
                candidate_dir=candidate_dir,
                is_valid=True,
                validated_at=validated_at,
                validation_error=None,
                metadata=new_metadata,
            )
            with self._lock:
                self._candidate_status = status
                self._candidate_runtime = new_runtime
        except Exception as exc:
            status = CandidateStatus(
                candidate_dir=candidate_dir,
                is_valid=False,
                validated_at=validated_at,
                validation_error=str(exc),
                metadata=None,
            )
            with self._lock:
                self._candidate_status = status
                self._candidate_runtime = None
        return status

    def promote_candidate(self) -> ModelMetadata:
        """Atomically promote the validated candidate to active.

        Requirements (all enforced, any violation raises RuntimeError):
          - A candidate must be registered via register_and_validate_candidate().
          - The candidate must have passed validation (is_valid=True).
          - The pre-loaded candidate runtime must still be present.

        After promotion:
          - The candidate becomes the active model.
          - The candidate slot is cleared.
          - The old active model is released.
        """
        with self._lock:
            if self._candidate_status is None:
                raise RuntimeError(
                    "No candidate registered. "
                    "Call register_and_validate_candidate(path) first."
                )
            if not self._candidate_status.is_valid:
                raise RuntimeError(
                    f"Cannot promote candidate at {self._candidate_status.candidate_dir}: "
                    f"validation failed with error: {self._candidate_status.validation_error!r}. "
                    "Fix the artifact or use a different candidate."
                )
            if self._candidate_runtime is None:
                # Shouldn't happen if register_and_validate_candidate was used, but guard anyway.
                raise RuntimeError(
                    "Candidate passed validation but pre-loaded runtime is missing. "
                    "Re-validate the candidate before promoting."
                )

            # Atomic swap — old model and candidate are replaced in one step.
            self._runtime = self._candidate_runtime
            self._metadata = self._candidate_status.metadata
            self._artifact_dir = self._candidate_status.candidate_dir
            self._load_error = None
            promoted_metadata = self._candidate_status.metadata

            # Clear candidate slot so it cannot be promoted a second time.
            self._candidate_status = None
            self._candidate_runtime = None

        return promoted_metadata

    def status_report(self) -> dict:
        """Serialisable snapshot of the full serving state.

        Suitable for management endpoints, periodic logs, or health checks.
        """
        return {
            "current_model": self._metadata.to_dict() if self._metadata else None,
            "load_error": self._load_error,
            "artifact_dir": str(self._artifact_dir),
            "candidate": self._candidate_status.to_dict()
            if self._candidate_status
            else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, artifact_dir: Path) -> None:
        new_runtime, new_metadata = _load_runtime_and_metadata(artifact_dir, self._device)
        with self._lock:
            self._runtime = new_runtime
            self._metadata = new_metadata
            self._load_error = None
            self._artifact_dir = artifact_dir


def _load_runtime_and_metadata(artifact_dir: Path, device: str):
    """Load artifact and return (runtime, metadata). Raises on any failure."""
    from searchess_ai.model_runtime.loader import load_artifact
    from searchess_ai.model_runtime.runtime import SupervisedModelRuntime

    loaded = load_artifact(artifact_dir, device=device)
    runtime = SupervisedModelRuntime(loaded)
    metadata = ModelMetadata(
        model_version=loaded.model_version,
        artifact_id=loaded.artifact_id,
        encoder_version=loaded.encoder_version,
        artifact_dir=str(artifact_dir),
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )
    return runtime, metadata
