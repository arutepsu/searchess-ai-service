from fastapi.testclient import TestClient

_VALID_PAYLOAD = {
    "candidate_model_id": "alpha-v1",
    "candidate_model_version": "1.0.0",
    "baseline_model_id": "legacy-v0",
    "baseline_model_version": "0.1.0",
    "number_of_games": 10,
}


def test_submit_evaluation_job_returns_202(client: TestClient) -> None:
    response = client.post("/api/v1/evaluation/jobs", json=_VALID_PAYLOAD)
    assert response.status_code == 202


def test_submit_evaluation_job_returns_expected_shape(client: TestClient) -> None:
    body = client.post("/api/v1/evaluation/jobs", json=_VALID_PAYLOAD).json()
    assert "evaluation_job_id" in body
    assert body["candidate_model_id"] == "alpha-v1"
    assert body["baseline_model_id"] == "legacy-v0"
    assert body["number_of_games"] == 10
    assert body["status"] == "queued"
    assert "submitted_at" in body


def test_submit_evaluation_job_returns_non_empty_job_id(client: TestClient) -> None:
    body = client.post("/api/v1/evaluation/jobs", json=_VALID_PAYLOAD).json()
    assert body["evaluation_job_id"]


def test_get_evaluation_job_returns_200_for_known_job(client: TestClient) -> None:
    job_id = client.post("/api/v1/evaluation/jobs", json=_VALID_PAYLOAD).json()[
        "evaluation_job_id"
    ]
    response = client.get(f"/api/v1/evaluation/jobs/{job_id}")
    assert response.status_code == 200


def test_get_evaluation_job_returns_expected_status(client: TestClient) -> None:
    job_id = client.post("/api/v1/evaluation/jobs", json=_VALID_PAYLOAD).json()[
        "evaluation_job_id"
    ]
    body = client.get(f"/api/v1/evaluation/jobs/{job_id}").json()
    assert body["evaluation_job_id"] == job_id
    assert body["status"] == "queued"
    assert body["candidate_model_id"] == "alpha-v1"
    assert body["baseline_model_id"] == "legacy-v0"
    assert body["number_of_games"] == 10
    assert "submitted_at" in body
    assert "updated_at" in body
    assert body["result"] is None


def test_get_evaluation_job_returns_404_for_unknown_job(client: TestClient) -> None:
    response = client.get("/api/v1/evaluation/jobs/does-not-exist")
    assert response.status_code == 404


def test_get_evaluation_job_404_has_structured_error_body(client: TestClient) -> None:
    body = client.get("/api/v1/evaluation/jobs/does-not-exist").json()
    assert body["type"] == "not_found"
    assert "does-not-exist" in body["error"]


def test_submit_evaluation_job_returns_422_for_zero_games(client: TestClient) -> None:
    response = client.post(
        "/api/v1/evaluation/jobs", json={**_VALID_PAYLOAD, "number_of_games": 0}
    )
    assert response.status_code == 422


def test_submit_evaluation_job_returns_422_for_missing_candidate(client: TestClient) -> None:
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "candidate_model_id"}
    response = client.post("/api/v1/evaluation/jobs", json=payload)
    assert response.status_code == 422


def test_submit_evaluation_job_returns_422_for_empty_model_id(client: TestClient) -> None:
    response = client.post(
        "/api/v1/evaluation/jobs", json={**_VALID_PAYLOAD, "baseline_model_id": ""}
    )
    assert response.status_code == 422


def test_submit_evaluation_job_accepts_optional_notes(client: TestClient) -> None:
    payload = {**_VALID_PAYLOAD, "notes": "dry run"}
    body = client.post("/api/v1/evaluation/jobs", json=payload).json()
    assert body["status"] == "queued"
