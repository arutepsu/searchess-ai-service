from fastapi.testclient import TestClient

_VALID_PAYLOAD = {
    "request_id": "req-1",
    "match_id": "match-1",
    "board_state": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "side_to_move": "white",
    "legal_moves": ["e2e4", "d2d4"],
}


def test_inference_move_returns_200_with_valid_response(client: TestClient) -> None:
    response = client.post("/api/v1/inference/move", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["selected_move"] in ["e2e4", "d2d4"]
    assert body["decision_type"] == "move"
    assert "model_id" in body
    assert "model_version" in body
    assert "decision_time_millis" in body


def test_inference_move_returns_422_for_invalid_side_to_move(client: TestClient) -> None:
    response = client.post(
        "/api/v1/inference/move",
        json={**_VALID_PAYLOAD, "side_to_move": "invalid"},
    )
    assert response.status_code == 422


def test_inference_move_returns_422_for_missing_required_field(client: TestClient) -> None:
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "request_id"}
    response = client.post("/api/v1/inference/move", json=payload)
    assert response.status_code == 422


def test_inference_move_returns_422_for_empty_legal_moves(client: TestClient) -> None:
    response = client.post(
        "/api/v1/inference/move",
        json={**_VALID_PAYLOAD, "legal_moves": []},
    )
    assert response.status_code == 422
