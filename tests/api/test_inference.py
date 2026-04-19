"""Contract compliance tests for POST /v1/move-suggestions."""
import random

import pytest
from fastapi.testclient import TestClient

from searchess_ai.api.app import create_app
from searchess_ai.api.dependencies import get_choose_move_use_case
from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.infrastructure.inference.fake_inference_engine import FakeInferenceEngine
from searchess_ai.infrastructure.inference.random_inference_engine import RandomInferenceEngine

_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_REQUEST_ID = "550e8400-e29b-41d4-a716-446655440000"
_GAME_ID = "123e4567-e89b-12d3-a456-426614174000"
_SESSION_ID = "789abcde-f012-3456-b789-abcdef012345"

_VALID_PAYLOAD: dict = {
    "requestId": _REQUEST_ID,
    "gameId": _GAME_ID,
    "sessionId": _SESSION_ID,
    "sideToMove": "white",
    "fen": _START_FEN,
    "legalMoves": [
        {"from": "e2", "to": "e4"},
        {"from": "d2", "to": "d4"},
    ],
    "limits": {"timeoutMillis": 3000},
}

_URL = "/v1/move-suggestions"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_suggest_move_returns_200(client: TestClient) -> None:
    assert client.post(_URL, json=_VALID_PAYLOAD).status_code == 200


def test_suggest_move_echoes_request_id(client: TestClient) -> None:
    body = client.post(_URL, json=_VALID_PAYLOAD).json()
    assert body["requestId"] == _REQUEST_ID


def test_suggest_move_returns_move_from_legal_set(client: TestClient) -> None:
    body = client.post(_URL, json=_VALID_PAYLOAD).json()
    move = body["move"]
    legal = {f"{m['from']}{m['to']}" for m in _VALID_PAYLOAD["legalMoves"]}
    assert f"{move['from']}{move['to']}" in legal


def test_suggest_move_response_includes_engine_metadata(client: TestClient) -> None:
    body = client.post(_URL, json=_VALID_PAYLOAD).json()
    assert "engineId" in body
    assert "engineVersion" in body
    assert "elapsedMillis" in body


def test_suggest_move_response_keys_are_camel_case(client: TestClient) -> None:
    body = client.post(_URL, json=_VALID_PAYLOAD).json()
    assert "requestId" in body
    assert "request_id" not in body
    assert "engineId" in body
    assert "engine_id" not in body


def test_suggest_move_move_object_uses_from_to_keys(client: TestClient) -> None:
    body = client.post(_URL, json=_VALID_PAYLOAD).json()
    move = body["move"]
    assert "from" in move
    assert "to" in move
    assert "from_square" not in move


def test_suggest_move_with_engine_selection(client: TestClient) -> None:
    payload = {**_VALID_PAYLOAD, "engine": {"engineId": "random-legal"}}
    assert client.post(_URL, json=payload).status_code == 200


def test_suggest_move_with_metadata(client: TestClient) -> None:
    payload = {**_VALID_PAYLOAD, "metadata": {"mode": "HumanVsAI"}}
    assert client.post(_URL, json=payload).status_code == 200


def test_suggest_move_with_promotion_in_legal_moves(client: TestClient) -> None:
    payload = {
        **_VALID_PAYLOAD,
        "legalMoves": [{"from": "e7", "to": "e8", "promotion": "queen"}],
    }
    body = client.post(_URL, json=payload).json()
    assert body["move"]["from"] == "e7"
    assert body["move"]["to"] == "e8"
    assert body["move"]["promotion"] == "queen"


def test_suggest_move_with_multiple_promotion_choices(client: TestClient) -> None:
    payload = {
        **_VALID_PAYLOAD,
        "legalMoves": [
            {"from": "e7", "to": "e8", "promotion": "queen"},
            {"from": "e7", "to": "e8", "promotion": "rook"},
            {"from": "e7", "to": "e8", "promotion": "bishop"},
            {"from": "e7", "to": "e8", "promotion": "knight"},
        ],
    }
    body = client.post(_URL, json=payload).json()
    assert body["move"]["promotion"] in ("queen", "rook", "bishop", "knight")


# ---------------------------------------------------------------------------
# Validation failures → 400 BAD_REQUEST with contract error shape
# ---------------------------------------------------------------------------

def _assert_bad_request(response, expected_request_id: str = _REQUEST_ID) -> None:
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "BAD_REQUEST"
    assert body["requestId"] == expected_request_id
    assert "message" in body


def test_missing_request_id_returns_400(client: TestClient) -> None:
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "requestId"}
    _assert_bad_request(client.post(_URL, json=payload), expected_request_id="")


def test_missing_fen_returns_400(client: TestClient) -> None:
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "fen"}
    _assert_bad_request(client.post(_URL, json=payload))


def test_missing_game_id_returns_400(client: TestClient) -> None:
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "gameId"}
    _assert_bad_request(client.post(_URL, json=payload))


def test_missing_session_id_returns_400(client: TestClient) -> None:
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "sessionId"}
    _assert_bad_request(client.post(_URL, json=payload))


def test_missing_limits_returns_400(client: TestClient) -> None:
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "limits"}
    _assert_bad_request(client.post(_URL, json=payload))


def test_invalid_side_to_move_returns_400(client: TestClient) -> None:
    payload = {**_VALID_PAYLOAD, "sideToMove": "invalid"}
    _assert_bad_request(client.post(_URL, json=payload))


def test_empty_legal_moves_returns_400(client: TestClient) -> None:
    payload = {**_VALID_PAYLOAD, "legalMoves": []}
    _assert_bad_request(client.post(_URL, json=payload))


def test_invalid_json_returns_400_with_empty_request_id(client: TestClient) -> None:
    response = client.post(_URL, content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "BAD_REQUEST"
    assert body["requestId"] == ""


def test_invalid_move_square_format_returns_400(client: TestClient) -> None:
    payload = {**_VALID_PAYLOAD, "legalMoves": [{"from": "z9", "to": "e4"}]}
    _assert_bad_request(client.post(_URL, json=payload))


def test_invalid_promotion_value_returns_400(client: TestClient) -> None:
    payload = {
        **_VALID_PAYLOAD,
        "legalMoves": [{"from": "e7", "to": "e8", "promotion": "king"}],
    }
    _assert_bad_request(client.post(_URL, json=payload))


def test_zero_timeout_millis_returns_400(client: TestClient) -> None:
    payload = {**_VALID_PAYLOAD, "limits": {"timeoutMillis": 0}}
    _assert_bad_request(client.post(_URL, json=payload))


# ---------------------------------------------------------------------------
# Backend replaceability — contract shape is identical across engines
# ---------------------------------------------------------------------------

def test_random_backend_returns_contract_shape() -> None:
    app = create_app()
    app.dependency_overrides[get_choose_move_use_case] = lambda: ChooseMoveUseCase(
        inference_engine=RandomInferenceEngine(rng=random.Random(0))
    )
    with TestClient(app) as c:
        body = c.post(_URL, json=_VALID_PAYLOAD).json()
    assert "requestId" in body
    assert "move" in body
    assert "from" in body["move"]


def test_response_shape_identical_across_backends() -> None:
    fake_app = create_app()
    fake_app.dependency_overrides[get_choose_move_use_case] = lambda: ChooseMoveUseCase(
        inference_engine=FakeInferenceEngine()
    )
    random_app = create_app()
    random_app.dependency_overrides[get_choose_move_use_case] = lambda: ChooseMoveUseCase(
        inference_engine=RandomInferenceEngine(rng=random.Random(0))
    )
    with TestClient(fake_app) as fc, TestClient(random_app) as rc:
        fake_body = fc.post(_URL, json=_VALID_PAYLOAD).json()
        random_body = rc.post(_URL, json=_VALID_PAYLOAD).json()
    assert set(fake_body.keys()) == set(random_body.keys())
