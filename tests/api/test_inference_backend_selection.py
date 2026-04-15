"""Tests proving backend replaceability at the composition root.

The inference endpoint and use case are unchanged.
Only the wiring selects which InferenceEngine is active.
"""
import random

import pytest
from fastapi.testclient import TestClient

from searchess_ai.api.app import create_app
from searchess_ai.api.dependencies import _resolve_inference_engine, get_choose_move_use_case
from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.infrastructure.inference.fake_inference_engine import FakeInferenceEngine
from searchess_ai.infrastructure.inference.random_inference_engine import RandomInferenceEngine

_VALID_PAYLOAD = {
    "request_id": "req-1",
    "match_id": "match-1",
    "board_state": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "side_to_move": "white",
    "legal_moves": ["e2e4", "d2d4"],
}


# --- Composition root unit tests ---

def test_resolve_returns_fake_engine_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    assert isinstance(_resolve_inference_engine(), FakeInferenceEngine)


def test_resolve_returns_fake_engine_when_explicitly_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "fake")
    assert isinstance(_resolve_inference_engine(), FakeInferenceEngine)


def test_resolve_returns_random_engine_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "random")
    assert isinstance(_resolve_inference_engine(), RandomInferenceEngine)


def test_resolve_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "RANDOM")
    assert isinstance(_resolve_inference_engine(), RandomInferenceEngine)


# --- API integration: endpoint unchanged when backend changes ---

def test_inference_endpoint_works_with_random_backend() -> None:
    """Backend swapped via dependency_overrides — route and use case unchanged."""
    app = create_app()
    app.dependency_overrides[get_choose_move_use_case] = lambda: ChooseMoveUseCase(
        inference_engine=RandomInferenceEngine(rng=random.Random(0))
    )
    with TestClient(app) as c:
        response = c.post("/api/v1/inference/move", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["selected_move"] in ["e2e4", "d2d4"]
    assert body["request_id"] == "req-1"
    assert body["decision_type"] == "move"
    assert body["model_id"] == "random-engine"


def test_inference_response_shape_identical_across_backends() -> None:
    """Both backends return the same DTO shape — the contract is stable."""
    fake_app = create_app()
    fake_app.dependency_overrides[get_choose_move_use_case] = lambda: ChooseMoveUseCase(
        inference_engine=FakeInferenceEngine()
    )
    random_app = create_app()
    random_app.dependency_overrides[get_choose_move_use_case] = lambda: ChooseMoveUseCase(
        inference_engine=RandomInferenceEngine(rng=random.Random(0))
    )

    with TestClient(fake_app) as fc, TestClient(random_app) as rc:
        fake_body = fc.post("/api/v1/inference/move", json=_VALID_PAYLOAD).json()
        random_body = rc.post("/api/v1/inference/move", json=_VALID_PAYLOAD).json()

    assert set(fake_body.keys()) == set(random_body.keys())


def test_inference_endpoint_works_with_env_var_random_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend selected via env var — composition root reads it at call time."""
    monkeypatch.setenv("INFERENCE_BACKEND", "random")
    app = create_app()
    with TestClient(app) as c:
        response = c.post("/api/v1/inference/move", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["selected_move"] in ["e2e4", "d2d4"]
