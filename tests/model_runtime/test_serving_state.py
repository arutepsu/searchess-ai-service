"""Tests for ServingState candidate validation and promotion.

Uses unittest.mock.patch to isolate from artifact I/O and torch.
Verifies the candidate validation/promotion contracts without requiring
a real artifact directory or GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from searchess_ai.model_runtime.serving_state import (
    CandidateStatus,
    ModelMetadata,
    ServingState,
)

_MODULE = "searchess_ai.model_runtime.serving_state"

_META_V1 = ModelMetadata(
    model_version="v1",
    artifact_id="art-v1",
    encoder_version="1.0",
    move_encoder_version="1.0",
    artifact_dir="/fake/v1",
    loaded_at="2026-01-01T00:00:00+00:00",
)

_META_V2 = ModelMetadata(
    model_version="v2",
    artifact_id="art-v2",
    encoder_version="1.0",
    move_encoder_version="1.0",
    artifact_dir="/fake/v2",
    loaded_at="2026-01-02T00:00:00+00:00",
)


def _make_state(meta: ModelMetadata = _META_V1) -> ServingState:
    """Create a ServingState that loaded successfully using a mocked runtime."""
    runtime = MagicMock()
    with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(runtime, meta)):
        return ServingState(Path(meta.artifact_dir))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestServingStateInit:
    def test_is_loaded_after_successful_init(self):
        state = _make_state()
        assert state.is_loaded is True
        assert state.load_error is None

    def test_metadata_reflects_loaded_artifact(self):
        state = _make_state()
        assert state.metadata.model_version == "v1"
        assert state.metadata.artifact_id == "art-v1"

    def test_no_candidate_at_startup(self):
        state = _make_state()
        assert state.candidate_status is None

    def test_load_error_captured_when_init_fails(self):
        with patch(f"{_MODULE}._load_runtime_and_metadata", side_effect=ValueError("bad artifact")):
            state = ServingState(Path("/bad/path"))
        assert state.is_loaded is False
        assert state.load_error is not None
        assert "bad artifact" in state.load_error

    def test_not_loaded_when_init_fails(self):
        with patch(f"{_MODULE}._load_runtime_and_metadata", side_effect=FileNotFoundError("missing")):
            state = ServingState(Path("/missing"))
        assert state.is_loaded is False
        assert state.runtime is None


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------


class TestCandidateValidation:
    def test_valid_candidate_is_accepted(self):
        state = _make_state()
        candidate_rt = MagicMock()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(candidate_rt, _META_V2)):
            status = state.register_and_validate_candidate(Path("/fake/v2"))

        assert status.is_valid is True
        assert status.validation_error is None
        assert status.metadata.model_version == "v2"

    def test_valid_candidate_stored_in_slot(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(MagicMock(), _META_V2)):
            status = state.register_and_validate_candidate(Path("/fake/v2"))

        assert state.candidate_status is status

    def test_invalid_candidate_records_failure(self):
        state = _make_state()
        with patch(
            f"{_MODULE}._load_runtime_and_metadata",
            side_effect=ValueError("encoder fingerprint mismatch"),
        ):
            status = state.register_and_validate_candidate(Path("/bad/candidate"))

        assert status.is_valid is False
        assert "fingerprint mismatch" in status.validation_error
        assert status.metadata is None

    def test_invalid_candidate_stored_in_slot(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", side_effect=ValueError("bad")):
            status = state.register_and_validate_candidate(Path("/bad"))

        assert state.candidate_status is status
        assert not state.candidate_status.is_valid

    def test_registering_new_candidate_replaces_old_one(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(MagicMock(), _META_V2)):
            state.register_and_validate_candidate(Path("/fake/v2"))
        # Register a second (invalid) candidate on top
        with patch(f"{_MODULE}._load_runtime_and_metadata", side_effect=ValueError("worse artifact")):
            status2 = state.register_and_validate_candidate(Path("/bad"))

        assert state.candidate_status is status2
        assert not state.candidate_status.is_valid

    def test_candidate_status_has_validated_at_timestamp(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(MagicMock(), _META_V2)):
            status = state.register_and_validate_candidate(Path("/fake/v2"))

        assert status.validated_at is not None
        assert "T" in status.validated_at  # ISO timestamp


# ---------------------------------------------------------------------------
# Promotion safety
# ---------------------------------------------------------------------------


class TestCandidatePromotion:
    def test_promote_without_candidate_raises(self):
        state = _make_state()
        with pytest.raises(RuntimeError, match="No candidate registered"):
            state.promote_candidate()

    def test_promote_invalid_candidate_raises(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", side_effect=ValueError("bad")):
            state.register_and_validate_candidate(Path("/bad"))

        with pytest.raises(RuntimeError, match="validation failed"):
            state.promote_candidate()

    def test_promote_invalid_candidate_does_not_change_active_model(self):
        state = _make_state()
        original_meta = state.metadata
        with patch(f"{_MODULE}._load_runtime_and_metadata", side_effect=ValueError("bad")):
            state.register_and_validate_candidate(Path("/bad"))

        try:
            state.promote_candidate()
        except RuntimeError:
            pass

        assert state.metadata is original_meta

    def test_promotion_swaps_active_model(self):
        state = _make_state()
        candidate_rt = MagicMock()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(candidate_rt, _META_V2)):
            state.register_and_validate_candidate(Path("/fake/v2"))

        promoted = state.promote_candidate()

        assert promoted.model_version == "v2"
        assert state.metadata.model_version == "v2"
        assert state.runtime is candidate_rt

    def test_promotion_clears_candidate_slot(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(MagicMock(), _META_V2)):
            state.register_and_validate_candidate(Path("/fake/v2"))

        state.promote_candidate()

        assert state.candidate_status is None

    def test_double_promotion_raises_after_first_succeeds(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(MagicMock(), _META_V2)):
            state.register_and_validate_candidate(Path("/fake/v2"))

        state.promote_candidate()  # first promotion succeeds

        with pytest.raises(RuntimeError, match="No candidate registered"):
            state.promote_candidate()  # second must fail — slot was cleared


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------


class TestStatusReport:
    def test_report_has_current_model(self):
        state = _make_state()
        report = state.status_report()
        assert report["current_model"]["model_version"] == "v1"
        assert report["load_error"] is None

    def test_report_candidate_is_none_before_registration(self):
        state = _make_state()
        assert state.status_report()["candidate"] is None

    def test_report_shows_valid_candidate_metadata(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(MagicMock(), _META_V2)):
            state.register_and_validate_candidate(Path("/fake/v2"))

        report = state.status_report()
        assert report["candidate"]["is_valid"] is True
        assert report["candidate"]["metadata"]["model_version"] == "v2"
        assert report["candidate"]["validation_error"] is None

    def test_report_shows_invalid_candidate_error(self):
        state = _make_state()
        with patch(
            f"{_MODULE}._load_runtime_and_metadata",
            side_effect=ValueError("encoder version mismatch"),
        ):
            state.register_and_validate_candidate(Path("/bad"))

        report = state.status_report()
        assert report["candidate"]["is_valid"] is False
        assert "encoder version mismatch" in report["candidate"]["validation_error"]
        assert report["candidate"]["metadata"] is None

    def test_report_after_promotion_shows_new_model_no_candidate(self):
        state = _make_state()
        with patch(f"{_MODULE}._load_runtime_and_metadata", return_value=(MagicMock(), _META_V2)):
            state.register_and_validate_candidate(Path("/fake/v2"))

        state.promote_candidate()
        report = state.status_report()

        assert report["current_model"]["model_version"] == "v2"
        assert report["candidate"] is None
