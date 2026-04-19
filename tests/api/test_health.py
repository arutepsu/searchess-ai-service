from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_health_returns_expected_json(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "searchess-ai-service"
    assert "version" in body
