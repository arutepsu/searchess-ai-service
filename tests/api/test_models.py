from fastapi.testclient import TestClient


def test_list_models_returns_200(client: TestClient) -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200


def test_list_models_returns_list_of_summaries(client: TestClient) -> None:
    body = client.get("/api/v1/models").json()
    assert isinstance(body, list)
    assert len(body) == 3
    for item in body:
        assert "model_id" in item
        assert "model_version" in item
        assert "status" in item
        assert "description" in item
        assert "tags" in item


def test_list_models_contains_known_model(client: TestClient) -> None:
    body = client.get("/api/v1/models").json()
    ids = {item["model_id"] for item in body}
    assert "alpha-v1" in ids


def test_get_model_returns_200_for_known_model(client: TestClient) -> None:
    response = client.get("/api/v1/models/alpha-v1")
    assert response.status_code == 200


def test_get_model_returns_expected_detail(client: TestClient) -> None:
    body = client.get("/api/v1/models/alpha-v1").json()
    assert body["model_id"] == "alpha-v1"
    assert body["status"] == "active"
    assert isinstance(body["supported_profiles"], list)
    assert len(body["supported_profiles"]) > 0
    assert "tags" in body


def test_get_model_returns_404_for_unknown_model(client: TestClient) -> None:
    response = client.get("/api/v1/models/does-not-exist")
    assert response.status_code == 404


def test_get_model_404_has_structured_error_body(client: TestClient) -> None:
    body = client.get("/api/v1/models/does-not-exist").json()
    assert body["type"] == "not_found"
    assert "does-not-exist" in body["error"]
