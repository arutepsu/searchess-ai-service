from fastapi.testclient import TestClient

_VALID_PAYLOAD = {"training_profile": "supervised"}


def test_submit_training_job_returns_202(client: TestClient) -> None:
    response = client.post("/api/v1/training/jobs", json=_VALID_PAYLOAD)
    assert response.status_code == 202


def test_submit_training_job_returns_expected_shape(client: TestClient) -> None:
    body = client.post("/api/v1/training/jobs", json=_VALID_PAYLOAD).json()
    assert "training_job_id" in body
    assert body["training_profile"] == "supervised"
    assert body["status"] == "queued"
    assert "submitted_at" in body


def test_submit_training_job_returns_non_empty_job_id(client: TestClient) -> None:
    body = client.post("/api/v1/training/jobs", json=_VALID_PAYLOAD).json()
    assert body["training_job_id"]


def test_get_training_job_returns_200_for_known_job(client: TestClient) -> None:
    job_id = client.post("/api/v1/training/jobs", json=_VALID_PAYLOAD).json()["training_job_id"]
    response = client.get(f"/api/v1/training/jobs/{job_id}")
    assert response.status_code == 200


def test_get_training_job_returns_expected_status(client: TestClient) -> None:
    job_id = client.post("/api/v1/training/jobs", json=_VALID_PAYLOAD).json()["training_job_id"]
    body = client.get(f"/api/v1/training/jobs/{job_id}").json()
    assert body["training_job_id"] == job_id
    assert body["status"] == "queued"
    assert body["training_profile"] == "supervised"
    assert "submitted_at" in body
    assert "updated_at" in body


def test_get_training_job_returns_404_for_unknown_job(client: TestClient) -> None:
    response = client.get("/api/v1/training/jobs/does-not-exist")
    assert response.status_code == 404


def test_get_training_job_404_has_structured_error_body(client: TestClient) -> None:
    body = client.get("/api/v1/training/jobs/does-not-exist").json()
    assert body["type"] == "not_found"
    assert "does-not-exist" in body["error"]


def test_submit_training_job_returns_422_for_invalid_profile(client: TestClient) -> None:
    response = client.post("/api/v1/training/jobs", json={"training_profile": "invalid"})
    assert response.status_code == 422


def test_submit_training_job_returns_422_for_missing_profile(client: TestClient) -> None:
    response = client.post("/api/v1/training/jobs", json={})
    assert response.status_code == 422


def test_submit_training_job_accepts_optional_base_model(client: TestClient) -> None:
    payload = {**_VALID_PAYLOAD, "base_model_id": "alpha-v1", "base_model_version": "1.0.0"}
    body = client.post("/api/v1/training/jobs", json=payload).json()
    assert body["status"] == "queued"
