"""Inference engine backed by a trained supervised policy model.

Decision transparency contract:
  - Every call to choose_move() produces an InferenceDecision with:
      * confidence set for MODEL decisions (None for fallbacks)
      * a structured log line that always carries:
          request_id, model_version, decision_mode, fallback_reason, latency_ms
  - Fallback decisions are never silent — they are logged at WARNING level.
  - Load failures are captured at startup and surfaced on first inference call.

Serving state contract:
  - ServingState owns artifact loading and holds model metadata.
  - reload() swaps the model atomically; the old model stays active until
    the new one is fully loaded and validated.

Counters:
  - _counters.snapshot() returns a dict of running decision-mode rates.
  - Useful for health checks or periodic log dumps.

Backend selection: INFERENCE_BACKEND=supervised + MODEL_ARTIFACT_DIR=<path>.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from searchess_ai.application.port.inference_engine import InferenceEngine
from searchess_ai.domain.game import Move
from searchess_ai.domain.inference import DecisionType, InferenceDecision, InferenceRequest
from searchess_ai.domain.model import ModelId, ModelVersion
from searchess_ai.model_runtime.decision import DecisionMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-process decision counters with per-reason breakdown
# ---------------------------------------------------------------------------


class _DecisionCounters:
    """Thread-safe running tallies of decision modes and fallback reasons.

    Designed to be cheap to update and snapshot.  Resets on process restart —
    use structured log aggregation for persistent trend analysis.

    Periodic snapshot logging fires every LOG_INTERVAL requests so that
    aggregate degradation is visible in logs without per-request scanning.
    """

    LOG_INTERVAL: int = 500  # emit a counters snapshot every N requests

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total: int = 0
        self.model_decision_count: int = 0
        self.fallback_count: int = 0
        self.error_recovered_count: int = 0
        # Per-FallbackReason breakdown — keys match FallbackReason.value strings.
        self.model_failure_count: int = 0
        self.invalid_output_count: int = 0
        self.runtime_exception_count: int = 0
        self.no_legal_moves_count: int = 0

    def record(self, mode: DecisionMode, fallback_reason=None) -> None:
        """Record one decision. Emits a periodic structured log snapshot."""
        should_log = False
        with self._lock:
            self.total += 1
            if mode == DecisionMode.MODEL:
                self.model_decision_count += 1
            elif mode == DecisionMode.FALLBACK:
                self.fallback_count += 1
            else:
                self.error_recovered_count += 1

            if fallback_reason is not None:
                reason_val = (
                    fallback_reason.value
                    if hasattr(fallback_reason, "value")
                    else str(fallback_reason)
                )
                if reason_val == "model_failure":
                    self.model_failure_count += 1
                elif reason_val == "invalid_output":
                    self.invalid_output_count += 1
                elif reason_val == "runtime_exception":
                    self.runtime_exception_count += 1
                elif reason_val == "no_legal_moves":
                    self.no_legal_moves_count += 1

            should_log = (self.total % self.LOG_INTERVAL == 0)

        if should_log:
            logger.info(json.dumps({"event": "counters_snapshot", **self.snapshot()}))

    def snapshot(self) -> dict:
        """Return a complete, serialisable counters snapshot with derived rates."""
        with self._lock:
            denom = self.total or 1
            all_fallbacks = self.fallback_count + self.error_recovered_count
            return {
                "total_requests": self.total,
                "model_decision_count": self.model_decision_count,
                "fallback_count": self.fallback_count,
                "error_recovered_count": self.error_recovered_count,
                "model_failure_count": self.model_failure_count,
                "invalid_output_count": self.invalid_output_count,
                "runtime_exception_count": self.runtime_exception_count,
                "no_legal_moves_count": self.no_legal_moves_count,
                "model_decision_rate": round(self.model_decision_count / denom, 4),
                "fallback_rate": round(all_fallbacks / denom, 4),
                "invalid_output_rate": round(self.invalid_output_count / denom, 4),
                "model_failure_rate": round(self.model_failure_count / denom, 4),
            }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


_DEFAULT_MODEL_ID = ModelId("supervised-policy")


class SupervisedInferenceEngine(InferenceEngine):
    """Move selection via a trained supervised policy network.

    Constructed by the dependency injection layer when INFERENCE_BACKEND=supervised.
    """

    def __init__(self, artifact_dir: Path, device: str = "cpu") -> None:
        self.artifact_dir = artifact_dir
        self.device = device
        self._counters = _DecisionCounters()
        self._state = None
        self._load_error: str | None = None

        try:
            from searchess_ai.model_runtime.serving_state import ServingState

            self._state = ServingState(artifact_dir, device=device)
            if not self._state.is_loaded:
                self._load_error = self._state.load_error
                logger.error(
                    json.dumps(
                        {
                            "event": "supervised_engine_load_failed",
                            "artifact_dir": str(artifact_dir),
                            "error": self._load_error,
                        }
                    )
                )
            else:
                meta = self._state.metadata
                logger.info(
                    json.dumps(
                        {
                            "event": "supervised_engine_loaded",
                            "artifact_dir": str(artifact_dir),
                            "model_version": meta.model_version if meta else "unknown",
                            "artifact_id": meta.artifact_id if meta else "unknown",
                            "encoder_version": meta.encoder_version if meta else "unknown",
                            "move_encoder_version": meta.move_encoder_version if meta else "unknown",
                        }
                    )
                )
        except ImportError as exc:
            self._load_error = f"Missing dependency: {exc}"
            logger.error(
                json.dumps({"event": "supervised_engine_import_error", "error": str(exc)})
            )
        except Exception as exc:
            self._load_error = f"Unexpected load error: {exc}"
            logger.error(
                json.dumps(
                    {
                        "event": "supervised_engine_load_failed",
                        "artifact_dir": str(artifact_dir),
                        "error": self._load_error,
                    }
                )
            )

    # ------------------------------------------------------------------
    # InferenceEngine implementation
    # ------------------------------------------------------------------

    def choose_move(self, request: InferenceRequest) -> InferenceDecision:
        start_ns = time.monotonic_ns()

        if self._state is None or not self._state.is_loaded:
            raise RuntimeError(
                f"SupervisedInferenceEngine is unavailable: {self._load_error}. "
                "Check MODEL_ARTIFACT_DIR and ensure training extras are installed."
            )

        meta = self._state.metadata
        model_version = ModelVersion(meta.model_version if meta else "unknown")
        legal_ucis = [m.value for m in request.legal_moves.moves]
        fen = request.position.value

        decision = self._state.runtime.select_move(fen, legal_ucis)

        elapsed_ms = max(0, (time.monotonic_ns() - start_ns) // 1_000_000)

        self._counters.record(decision.decision_mode, decision.fallback_reason)

        _log_decision(
            request_id=request.request_id,
            model_version=model_version.value,
            decision=decision,
            latency_ms=elapsed_ms,
        )

        return InferenceDecision(
            request_id=request.request_id,
            decision_type=DecisionType.MOVE,
            selected_move=Move(decision.selected_uci),
            model_id=request.model_id or _DEFAULT_MODEL_ID,
            model_version=request.model_version or model_version,
            decision_time_millis=elapsed_ms,
            policy_profile=request.policy_profile,
            # confidence is None for fallback decisions so callers can distinguish them.
            confidence=decision.confidence,
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def counters_snapshot(self) -> dict:
        """Return current decision-mode counters with per-reason breakdown and rates."""
        return self._counters.snapshot()

    def current_model_metadata(self) -> dict | None:
        """Return a snapshot of the active model's identity, or None if not loaded."""
        if self._state and self._state.metadata:
            return self._state.metadata.to_dict()
        return None

    def serving_status(self) -> dict:
        """Full status snapshot: current model, candidate, counters, and load error.

        Safe to call at any time. Suitable for health checks or management logs.
        """
        return {
            "current_model": self.current_model_metadata(),
            "load_error": self._load_error,
            "candidate": (
                self._state.candidate_status.to_dict()
                if self._state and self._state.candidate_status
                else None
            ),
            "counters": self._counters.snapshot(),
        }

    def reload(self, new_artifact_dir: Path | None = None) -> dict:
        """Reload the active model artifact atomically.

        Returns the new model metadata dict.
        Raises RuntimeError if loading fails (existing model stays active).
        """
        if self._state is None:
            raise RuntimeError(
                "Cannot reload: engine was never successfully initialised."
            )
        try:
            new_meta = self._state.reload(new_artifact_dir)
            logger.info(
                json.dumps(
                    {
                        "event": "supervised_engine_reloaded",
                        "model_version": new_meta.model_version,
                        "artifact_id": new_meta.artifact_id,
                        "artifact_dir": new_meta.artifact_dir,
                    }
                )
            )
            return new_meta.to_dict()
        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "supervised_engine_reload_failed",
                        "error": str(exc),
                        "note": "existing model remains active",
                    }
                )
            )
            raise RuntimeError(f"Reload failed, existing model still active: {exc}") from exc

    def register_and_validate_candidate(self, candidate_dir: Path) -> dict:
        """Validate a candidate artifact without activating it.

        Returns the CandidateStatus dict (includes is_valid and any error).
        A valid candidate can then be promoted via promote_candidate().
        """
        if self._state is None:
            raise RuntimeError("Engine not initialised — cannot register candidate.")
        status = self._state.register_and_validate_candidate(candidate_dir)
        level = logger.info if status.is_valid else logger.warning
        level(
            json.dumps(
                {
                    "event": "candidate_validation",
                    "candidate_dir": str(candidate_dir),
                    "is_valid": status.is_valid,
                    "validation_error": status.validation_error,
                    "metadata": status.metadata.to_dict() if status.metadata else None,
                }
            )
        )
        return status.to_dict()

    def promote_candidate(self) -> dict:
        """Atomically promote the validated candidate to active.

        Raises RuntimeError if no valid candidate exists.
        Returns the newly active model metadata dict.
        """
        if self._state is None:
            raise RuntimeError("Engine not initialised — cannot promote candidate.")
        try:
            new_meta = self._state.promote_candidate()
            logger.info(
                json.dumps(
                    {
                        "event": "candidate_promoted",
                        "model_version": new_meta.model_version,
                        "artifact_id": new_meta.artifact_id,
                        "artifact_dir": new_meta.artifact_dir,
                    }
                )
            )
            return new_meta.to_dict()
        except RuntimeError as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "candidate_promotion_failed",
                        "error": str(exc),
                        "note": "existing model remains active",
                    }
                )
            )
            raise


# ---------------------------------------------------------------------------
# Internal logging helper
# ---------------------------------------------------------------------------


def _log_decision(
    *,
    request_id: str,
    model_version: str,
    decision,
    latency_ms: int,
) -> None:
    payload = {
        "event": "inference_decision",
        "request_id": request_id,
        "model_version": model_version,
        "decision_mode": decision.decision_mode.value,
        "fallback_reason": decision.fallback_reason.value if decision.fallback_reason else None,
        "confidence": round(decision.confidence, 4) if decision.confidence is not None else None,
        "latency_ms": latency_ms,
    }
    if decision.is_fallback():
        # Fallbacks at WARNING so they stand out in logs.
        logger.warning(json.dumps(payload))
    else:
        logger.info(json.dumps(payload))
