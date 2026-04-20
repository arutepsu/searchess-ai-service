"""API boundary tests for metadata.testMode integration hook."""
import pytest
from fastapi.testclient import TestClient

from searchess_ai.api.app import create_app

_URL = "/v1/move-suggestions"
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_REQUEST_ID = "hook-test-request-id"

_BASE = {
    "requestId": _REQUEST_ID,
    "gameId": "hook-test-game",
    "sessionId": "hook-test-session",
    "sideToMove": "white",
    "fen": _START_FEN,
    "legalMoves": [
        {"from": "e2", "to": "e4"},
        {"from": "d2", "to": "d4"},
        {"from": "g1", "to": "f3"},
    ],
    "limits": {"timeoutMillis": 5000},
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Default behavior — metadata absent or testMode absent/unknown
# ---------------------------------------------------------------------------

def test_no_metadata_uses_normal_inference(client: TestClient) -> None:
    resp = client.post(_URL, json=_BASE)
    assert resp.status_code == 200
    body = resp.json()
    legal = {f"{m['from']}{m['to']}" for m in _BASE["legalMoves"]}
    assert f"{body['move']['from']}{body['move']['to']}" in legal


def test_metadata_without_test_mode_uses_normal_inference(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"mode": "HumanVsAI"}}
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    legal = {f"{m['from']}{m['to']}" for m in _BASE["legalMoves"]}
    assert f"{body['move']['from']}{body['move']['to']}" in legal


def test_unknown_test_mode_falls_through_to_normal(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"testMode": "totally_unknown_value"}}
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    legal = {f"{m['from']}{m['to']}" for m in _BASE["legalMoves"]}
    assert f"{body['move']['from']}{body['move']['to']}" in legal


# ---------------------------------------------------------------------------
# illegal_move mode
# ---------------------------------------------------------------------------

def test_illegal_move_returns_200(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"testMode": "illegal_move"}}
    assert client.post(_URL, json=payload).status_code == 200


def test_illegal_move_echoes_request_id(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"testMode": "illegal_move"}}
    assert client.post(_URL, json=payload).json()["requestId"] == _REQUEST_ID


def test_illegal_move_is_not_in_legal_moves(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"testMode": "illegal_move"}}
    body = client.post(_URL, json=payload).json()
    legal = {f"{m['from']}{m['to']}" for m in _BASE["legalMoves"]}
    returned = f"{body['move']['from']}{body['move']['to']}"
    assert returned not in legal


def test_illegal_move_response_has_valid_move_shape(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"testMode": "illegal_move"}}
    move = client.post(_URL, json=payload).json()["move"]
    assert "from" in move
    assert "to" in move
    assert len(move["from"]) == 2
    assert len(move["to"]) == 2


def test_illegal_move_not_in_legal_set_with_single_move(client: TestClient) -> None:
    """Even when only one legal move is provided the hook returns something else."""
    payload = {
        **_BASE,
        "legalMoves": [{"from": "e2", "to": "e4"}],
        "metadata": {"testMode": "illegal_move"},
    }
    body = client.post(_URL, json=payload).json()
    returned = f"{body['move']['from']}{body['move']['to']}"
    assert returned != "e2e4"


# ---------------------------------------------------------------------------
# malformed_response mode
# ---------------------------------------------------------------------------

def test_malformed_response_returns_200(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"testMode": "malformed_response"}}
    assert client.post(_URL, json=payload).status_code == 200


def test_malformed_response_echoes_request_id(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"testMode": "malformed_response"}}
    assert client.post(_URL, json=payload).json()["requestId"] == _REQUEST_ID


def test_malformed_response_is_missing_move_field(client: TestClient) -> None:
    payload = {**_BASE, "metadata": {"testMode": "malformed_response"}}
    body = client.post(_URL, json=payload).json()
    assert "move" not in body


def test_malformed_response_is_otherwise_valid_json(client: TestClient) -> None:
    """Body must be parseable JSON so the Scala client reaches its schema-check path."""
    payload = {**_BASE, "metadata": {"testMode": "malformed_response"}}
    resp = client.post(_URL, json=payload)
    body = resp.json()  # would raise if not valid JSON
    assert isinstance(body, dict)
