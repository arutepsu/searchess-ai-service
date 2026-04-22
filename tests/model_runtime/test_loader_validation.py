"""Tests for artifact loader validation logic.

Covers position encoder config, move encoder config, and model config checks.
The fingerprint mismatch path is the key defense against silent encoder drift.

No torch or real artifact files are required: the validation functions operate
on plain dicts and are tested in isolation.
"""

from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed (run: uv sync --extra training)")
np = pytest.importorskip("numpy", reason="numpy not installed (run: uv sync --extra training)")

from searchess_ai.model_runtime.encoder import (
    ENCODER_VERSION,
    FEATURE_SIZE,
    compute_encoder_fingerprint,
)
from searchess_ai.model_runtime.loader import (
    _validate_encoder_config,
    _validate_model_config,
    _validate_move_encoder_config,
)
from searchess_ai.model_runtime.move_encoder import (
    MOVE_ENCODER_VERSION,
    MOVE_FEATURE_SIZE,
    compute_move_encoder_fingerprint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_encoder_config() -> dict:
    return {
        "version": ENCODER_VERSION,
        "feature_size": FEATURE_SIZE,
        "fingerprint": compute_encoder_fingerprint(),
    }


def _valid_move_encoder_config() -> dict:
    return {
        "version": MOVE_ENCODER_VERSION,
        "move_feature_size": MOVE_FEATURE_SIZE,
        "fingerprint": compute_move_encoder_fingerprint(),
    }


def _valid_model_config() -> dict:
    return {
        "architecture": "MoveScoringNetwork",
        "position_feature_size": FEATURE_SIZE,
        "move_feature_size": MOVE_FEATURE_SIZE,
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "dropout": 0.1,
    }


# ---------------------------------------------------------------------------
# Position encoder config
# ---------------------------------------------------------------------------


class TestEncoderConfigHappyPath:
    def test_valid_config_passes(self):
        _validate_encoder_config(_valid_encoder_config())

    def test_extra_keys_ignored(self):
        config = _valid_encoder_config()
        config["piece_plane_order"] = ["P", "p"]
        _validate_encoder_config(config)


class TestEncoderConfigMissingKeys:
    @pytest.mark.parametrize("key", ["version", "feature_size", "fingerprint"])
    def test_missing_key_raises(self, key: str):
        config = _valid_encoder_config()
        del config[key]
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_encoder_config(config)


class TestEncoderVersionMismatch:
    def test_wrong_version_raises(self):
        config = _valid_encoder_config()
        config["version"] = "0.0"
        with pytest.raises(ValueError, match="Encoder version mismatch"):
            _validate_encoder_config(config)

    def test_version_mismatch_message_names_both_versions(self):
        config = _valid_encoder_config()
        config["version"] = "0.0"
        with pytest.raises(ValueError) as exc_info:
            _validate_encoder_config(config)
        msg = str(exc_info.value)
        assert "0.0" in msg and ENCODER_VERSION in msg

    def test_version_mismatch_raised_before_fingerprint_check(self):
        config = _valid_encoder_config()
        config["version"] = "999.0"
        config["fingerprint"] = "bad" * 21
        with pytest.raises(ValueError, match="Encoder version mismatch"):
            _validate_encoder_config(config)


class TestEncoderFingerprintMismatch:
    def test_wrong_fingerprint_raises(self):
        config = _valid_encoder_config()
        config["fingerprint"] = "a" * 64
        with pytest.raises(ValueError, match="[Ff]ingerprint"):
            _validate_encoder_config(config)

    def test_fingerprint_shows_stale_value_in_message(self):
        config = _valid_encoder_config()
        stale = "c" * 64
        config["fingerprint"] = stale
        with pytest.raises(ValueError) as exc_info:
            _validate_encoder_config(config)
        assert stale in str(exc_info.value)

    def test_fingerprint_catches_drift_even_with_correct_version(self):
        config = _valid_encoder_config()
        assert config["version"] == ENCODER_VERSION
        config["fingerprint"] = "0" * 64
        with pytest.raises(ValueError, match="[Ff]ingerprint"):
            _validate_encoder_config(config)


# ---------------------------------------------------------------------------
# Move encoder config
# ---------------------------------------------------------------------------


class TestMoveEncoderConfigHappyPath:
    def test_valid_config_passes(self):
        _validate_move_encoder_config(_valid_move_encoder_config())

    def test_extra_keys_ignored(self):
        config = _valid_move_encoder_config()
        config["promotion_classes"] = ["none", "queen"]
        _validate_move_encoder_config(config)


class TestMoveEncoderConfigMissingKeys:
    @pytest.mark.parametrize("key", ["version", "move_feature_size", "fingerprint"])
    def test_missing_key_raises(self, key: str):
        config = _valid_move_encoder_config()
        del config[key]
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_move_encoder_config(config)


class TestMoveEncoderVersionMismatch:
    def test_wrong_version_raises(self):
        config = _valid_move_encoder_config()
        config["version"] = "0.0"
        with pytest.raises(ValueError, match="Move encoder version mismatch"):
            _validate_move_encoder_config(config)


class TestMoveEncoderFeatureSizeMismatch:
    def test_wrong_feature_size_raises(self):
        config = _valid_move_encoder_config()
        config["move_feature_size"] = 64
        with pytest.raises(ValueError, match="move_feature_size"):
            _validate_move_encoder_config(config)


class TestMoveEncoderFingerprintMismatch:
    def test_wrong_fingerprint_raises(self):
        config = _valid_move_encoder_config()
        config["fingerprint"] = "b" * 64
        with pytest.raises(ValueError, match="[Ff]ingerprint"):
            _validate_move_encoder_config(config)

    def test_fingerprint_catches_drift_even_with_correct_version(self):
        config = _valid_move_encoder_config()
        assert config["version"] == MOVE_ENCODER_VERSION
        config["fingerprint"] = "0" * 64
        with pytest.raises(ValueError, match="[Ff]ingerprint"):
            _validate_move_encoder_config(config)


# ---------------------------------------------------------------------------
# Model config validation
# ---------------------------------------------------------------------------


class TestModelConfigValidation:
    def test_valid_config_passes(self):
        _validate_model_config(_valid_model_config())

    @pytest.mark.parametrize(
        "key",
        [
            "architecture",
            "position_feature_size",
            "move_feature_size",
            "hidden_size",
            "num_hidden_layers",
            "dropout",
        ],
    )
    def test_missing_key_raises(self, key: str):
        config = _valid_model_config()
        del config[key]
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_model_config(config)

    def test_old_policy_network_architecture_rejected(self):
        """Old fixed-vocabulary artifacts must be rejected explicitly."""
        config = _valid_model_config()
        config["architecture"] = "PolicyNetwork"
        with pytest.raises(ValueError, match="MoveScoringNetwork"):
            _validate_model_config(config)

    def test_unknown_architecture_rejected(self):
        config = _valid_model_config()
        config["architecture"] = "SomeOtherNet"
        with pytest.raises(ValueError, match="MoveScoringNetwork"):
            _validate_model_config(config)

    def test_extra_keys_ignored(self):
        config = _valid_model_config()
        config["extra"] = "ignored"
        _validate_model_config(config)
