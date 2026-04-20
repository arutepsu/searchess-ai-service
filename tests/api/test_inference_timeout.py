"""Tests for ENGINE_TIMEOUT (504) enforcement via limits.timeoutMillis."""
import time

import pytest
from fastapi.testclient import TestClient

from searchess_ai.api.app import create_app
from searchess_ai.api.dependencies import get_choose_move_use_case
from searchess_ai.application.port.inference_engine import InferenceEngine
from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.domain.inference import InferenceDecision, InferenceRequest
from searchess_ai.infrastructure.inference.fake_inference_engine import FakeInferenceEngine

_URL = "/v1/move-suggestions"
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_REQUEST_ID = "timeout-test-00000000-0000-0000"
_GAME_ID = "game-timeout-test"
_SESSION_ID = "session-timeout-test"


def _payload(timeout_millis: int) -> dict:
    return {
        "requestId": _REQUEST_ID,
        "gameId": _GAME_ID,
        "sessionId": _SESSION_ID,
        "sideToMove": "white",
        "fen": _START_FEN,
        "legalMoves": [{"from": "e2", "to": "e4"}],
        "limits": {"timeoutMillis": timeout_millis},
    }


class SlowEngine(InferenceEngine):
    """Engine that sleeps for sleep_ms before delegating to a real engine."""

    def __init__(self, sleep_ms: int, delegate: InferenceEngine) -> None:
        self._sleep_ms = sleep_ms
        self._delegate = delegate

    def choose_move(self, request: InferenceRequest) -> InferenceDecision:
        time.sleep(self._sleep_ms / 1000.0)
        return self._delegate.choose_move(request)


def _client_with_engine(engine: InferenceEngine) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_choose_move_use_case] = lambda: ChooseMoveUseCase(
        inference_engine=engine
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# Timeout triggered
# ---------------------------------------------------------------------------

def test_engine_timeout_returns_504() -> None:
    # Engine sleeps 300ms; limit is 10ms → must time out.
    with _client_with_engine(SlowEngine(300, FakeInferenceEngine())) as c:
        resp = c.post(_URL, json=_payload(10))

    assert resp.status_code == 504


def test_engine_timeout_code_is_engine_timeout() -> None:
    with _client_with_engine(SlowEngine(300, FakeInferenceEngine())) as c:
        resp = c.post(_URL, json=_payload(10))

    assert resp.json()["code"] == "ENGINE_TIMEOUT"


def test_engine_timeout_echoes_request_id() -> None:
    with _client_with_engine(SlowEngine(300, FakeInferenceEngine())) as c:
        resp = c.post(_URL, json=_payload(10))

    assert resp.json()["requestId"] == _REQUEST_ID


def test_engine_timeout_message_contains_millis() -> None:
    with _client_with_engine(SlowEngine(300, FakeInferenceEngine())) as c:
        resp = c.post(_URL, json=_payload(10))

    assert "10ms" in resp.json()["message"]


def test_engine_timeout_response_shape_matches_contract() -> None:
    """Timeout error body must match the shared error contract."""
    with _client_with_engine(SlowEngine(300, FakeInferenceEngine())) as c:
        resp = c.post(_URL, json=_payload(10))

    body = resp.json()
    assert set(body.keys()) >= {"requestId", "code", "message"}


# ---------------------------------------------------------------------------
# No timeout — fast engine returns well within limit
# ---------------------------------------------------------------------------

def test_fast_engine_not_timed_out() -> None:
    with _client_with_engine(FakeInferenceEngine()) as c:
        resp = c.post(_URL, json=_payload(5000))

    assert resp.status_code == 200


def test_fast_engine_returns_valid_move() -> None:
    with _client_with_engine(FakeInferenceEngine()) as c:
        resp = c.post(_URL, json=_payload(5000))

    body = resp.json()
    assert body["requestId"] == _REQUEST_ID
    assert "move" in body


# ---------------------------------------------------------------------------
# Boundary — engine finishes just under the limit
# ---------------------------------------------------------------------------

def test_engine_just_under_timeout_succeeds() -> None:
    # Engine sleeps 20ms; limit is 300ms → should complete in time.
    with _client_with_engine(SlowEngine(20, FakeInferenceEngine())) as c:
        resp = c.post(_URL, json=_payload(300))

    assert resp.status_code == 200
    assert resp.json()["requestId"] == _REQUEST_ID


# ---------------------------------------------------------------------------
# Timeout does not swallow other error codes
# ---------------------------------------------------------------------------

def test_bad_position_not_overridden_by_timeout() -> None:
    """BAD_POSITION must still return 422 when timeout is generous."""
    from searchess_ai.infrastructure.inference._openspiel_mapping import BadPositionAdapterError

    class _BadPosEngine(InferenceEngine):
        def choose_move(self, request: InferenceRequest) -> InferenceDecision:  # type: ignore[override]
            raise BadPositionAdapterError("invalid FEN")

    with _client_with_engine(_BadPosEngine()) as c:
        resp = c.post(_URL, json=_payload(5000))

    assert resp.status_code == 422
    assert resp.json()["code"] == "BAD_POSITION"


def test_engine_unavailable_not_overridden_by_timeout() -> None:
    """ENGINE_UNAVAILABLE must still return 503 when timeout is generous."""
    from searchess_ai.infrastructure.inference._openspiel_mapping import OpenSpielAdapterError

    class _UnavailableEngine(InferenceEngine):
        def choose_move(self, request: InferenceRequest) -> InferenceDecision:  # type: ignore[override]
            raise OpenSpielAdapterError("pyspiel is not installed.")

    with _client_with_engine(_UnavailableEngine()) as c:
        resp = c.post(_URL, json=_payload(5000))

    assert resp.status_code == 503
    assert resp.json()["code"] == "ENGINE_UNAVAILABLE"


def test_engine_failure_not_overridden_by_timeout() -> None:
    """ENGINE_FAILURE must still return 500 when timeout is generous."""
    class _CrashEngine(InferenceEngine):
        def choose_move(self, request: InferenceRequest) -> InferenceDecision:  # type: ignore[override]
            raise RuntimeError("boom")

    with _client_with_engine(_CrashEngine()) as c:
        resp = c.post(_URL, json=_payload(5000))

    assert resp.status_code == 500
    assert resp.json()["code"] == "ENGINE_FAILURE"
