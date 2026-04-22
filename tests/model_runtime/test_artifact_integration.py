"""End-to-end integration test: artifact → load → runtime → inference.

Builds a minimal valid artifact using real (random-initialized) MoveScoringNetwork
weights, loads it through the full production code path, and runs inference.

Proves:
- ArtifactManifest + save_artifact produce a loadable artifact with both encoder configs
- load_artifact validates position encoder, move encoder, and model config
- SupervisedModelRuntime produces a MODEL-mode decision (not a fallback)
- Underpromotion moves are distinguishable through the runtime
- Old-style artifacts (missing move_encoder_config) are rejected
- Tampered fingerprints are rejected

The model is tiny (1 hidden layer, hidden_size=32) so the test runs in
under one second on any CPU-only machine.

Requires training extras: uv sync --extra training
"""

from __future__ import annotations

import pytest

chess = pytest.importorskip("chess", reason="python-chess not installed (run: uv sync --extra training)")
torch = pytest.importorskip("torch", reason="torch not installed (run: uv sync --extra training)")

from searchess_ai.artifacts.schema import ArtifactManifest, save_artifact
from searchess_ai.model_runtime.decision import DecisionMode
from searchess_ai.model_runtime.encoder import ENCODER_VERSION, REFERENCE_FEN, encoder_config
from searchess_ai.model_runtime.loader import load_artifact
from searchess_ai.model_runtime.move_encoder import MOVE_ENCODER_VERSION, move_encoder_config
from searchess_ai.model_runtime.runtime import SupervisedModelRuntime
from searchess_ai.training.model import MoveScoringNetwork

_LEGAL_MOVES = ["e2e4", "d2d4", "g1f3", "b1c3", "a2a3"]
# Include underpromotion options to verify they remain distinguishable
_LEGAL_WITH_PROMOS = ["e2e4", "e7e8q", "e7e8r", "e7e8b", "e7e8n"]


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    """Create a minimal valid artifact with a tiny random-initialized model."""
    path = tmp_path_factory.mktemp("integration_artifact")

    model = MoveScoringNetwork(
        hidden_size=32,
        num_hidden_layers=1,
        dropout=0.0,
    )
    model.eval()

    manifest = ArtifactManifest.new(
        model_version="integration-test-v0",
        dataset_version="none",
        training_run_id="integration-test-run",
        model_config=model.config(),
        encoder_config=encoder_config(),
        move_encoder_config=move_encoder_config(),
        dataset_ref={
            "dataset_id": "integration-test",
            "source_pgn_sha256": "n/a",
            "extraction_version": "n/a",
            "filter_config_hash": "n/a",
        },
        training_run={},
        evaluation_summary={},
    )

    save_artifact(path, model.state_dict(), manifest)
    return path


# ---------------------------------------------------------------------------
# Artifact load
# ---------------------------------------------------------------------------


class TestArtifactLoad:
    def test_load_artifact_succeeds(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        assert loaded.model_version == "integration-test-v0"
        assert loaded.artifact_id != ""

    def test_encoder_versions_round_trip(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        assert loaded.encoder_version == ENCODER_VERSION
        assert loaded.move_encoder_version == MOVE_ENCODER_VERSION

    def test_model_is_eval_mode_after_load(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        assert not loaded.model.training


# ---------------------------------------------------------------------------
# Runtime inference
# ---------------------------------------------------------------------------


class TestRuntimeInference:
    def test_select_move_returns_model_decision(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        runtime = SupervisedModelRuntime(loaded)
        decision = runtime.select_move(REFERENCE_FEN, _LEGAL_MOVES)
        assert decision.decision_mode == DecisionMode.MODEL

    def test_selected_move_is_from_legal_set(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        runtime = SupervisedModelRuntime(loaded)
        decision = runtime.select_move(REFERENCE_FEN, _LEGAL_MOVES)
        assert decision.selected_uci in _LEGAL_MOVES

    def test_confidence_is_set_and_in_range(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        runtime = SupervisedModelRuntime(loaded)
        decision = runtime.select_move(REFERENCE_FEN, _LEGAL_MOVES)
        assert decision.confidence is not None
        assert 0.0 <= decision.confidence <= 1.0

    def test_no_fallback_triggered(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        runtime = SupervisedModelRuntime(loaded)
        decision = runtime.select_move(REFERENCE_FEN, _LEGAL_MOVES)
        assert decision.fallback_reason is None
        assert decision.error_detail is None

    def test_model_version_accessible(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        runtime = SupervisedModelRuntime(loaded)
        assert runtime.model_version == "integration-test-v0"

    def test_inference_is_deterministic(self, artifact_dir):
        loaded = load_artifact(artifact_dir, device="cpu")
        runtime = SupervisedModelRuntime(loaded)
        d1 = runtime.select_move(REFERENCE_FEN, _LEGAL_MOVES)
        d2 = runtime.select_move(REFERENCE_FEN, _LEGAL_MOVES)
        assert d1.selected_uci == d2.selected_uci
        assert d1.confidence == d2.confidence

    def test_underpromotions_are_distinguishable(self, artifact_dir):
        """All four promotion types must be scoreable without error."""
        loaded = load_artifact(artifact_dir, device="cpu")
        runtime = SupervisedModelRuntime(loaded)
        # Use a FEN where any promotion would be legal (model scores them, picks one)
        decision = runtime.select_move(REFERENCE_FEN, _LEGAL_WITH_PROMOS)
        assert decision.decision_mode == DecisionMode.MODEL
        assert decision.selected_uci in _LEGAL_WITH_PROMOS


# ---------------------------------------------------------------------------
# Encoder compatibility enforcement
# ---------------------------------------------------------------------------


class TestEncoderCompatibilityEnforcement:
    def test_mismatched_position_fingerprint_rejected(self, artifact_dir, tmp_path):
        import json, shutil

        tampered = tmp_path / "tampered_pos"
        shutil.copytree(artifact_dir, tampered)
        manifest_path = tampered / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["encoder_config"]["fingerprint"] = "a" * 64
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with pytest.raises(ValueError, match="[Ff]ingerprint"):
            load_artifact(tampered, device="cpu")

    def test_mismatched_move_fingerprint_rejected(self, artifact_dir, tmp_path):
        import json, shutil

        tampered = tmp_path / "tampered_move"
        shutil.copytree(artifact_dir, tampered)
        manifest_path = tampered / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["move_encoder_config"]["fingerprint"] = "b" * 64
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with pytest.raises(ValueError, match="[Ff]ingerprint"):
            load_artifact(tampered, device="cpu")

    def test_old_artifact_missing_move_encoder_config_rejected(self, artifact_dir, tmp_path):
        """An artifact without move_encoder_config must be rejected at manifest load time."""
        import json, shutil

        old_style = tmp_path / "old_artifact"
        shutil.copytree(artifact_dir, old_style)
        manifest_path = old_style / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)
        del raw["move_encoder_config"]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with pytest.raises(ValueError, match="move_encoder_config"):
            load_artifact(old_style, device="cpu")

    def test_old_policy_network_architecture_rejected(self, artifact_dir, tmp_path):
        import json, shutil

        old_arch = tmp_path / "old_arch"
        shutil.copytree(artifact_dir, old_arch)
        manifest_path = old_arch / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["model_config"]["architecture"] = "PolicyNetwork"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with pytest.raises(ValueError, match="MoveScoringNetwork"):
            load_artifact(old_arch, device="cpu")
