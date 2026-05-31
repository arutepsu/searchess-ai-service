"""Tests for OpenTelemetry tracing initialisation (searchess_ai.telemetry)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# configure_tracing — unit tests (no global OTel state mutation)
# ---------------------------------------------------------------------------

def test_configure_tracing_is_noop_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_tracing must not raise and must leave the tracer provider unchanged."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from opentelemetry import trace as otel_trace

    provider_before = otel_trace.get_tracer_provider()

    from searchess_ai.telemetry import configure_tracing
    configure_tracing()

    assert otel_trace.get_tracer_provider() is provider_before


def test_configure_tracing_installs_sdk_provider_when_endpoint_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """configure_tracing must install a TracerProvider when an endpoint is configured."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "searchess-test")

    from opentelemetry.sdk.trace import TracerProvider

    captured: list[object] = []

    # Mock the exporter to avoid a real gRPC connection and capture the provider
    # without mutating global OTel state.
    with (
        patch("searchess_ai.telemetry.OTLPSpanExporter", return_value=MagicMock()),
        patch("opentelemetry.trace.set_tracer_provider", side_effect=captured.append),
    ):
        from searchess_ai.telemetry import configure_tracing
        configure_tracing()

    assert len(captured) == 1
    assert isinstance(captured[0], TracerProvider)


def test_configure_tracing_uses_custom_service_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-custom-service")

    from opentelemetry.sdk.resources import SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider

    captured: list[TracerProvider] = []

    with (
        patch("searchess_ai.telemetry.OTLPSpanExporter", return_value=MagicMock()),
        patch("opentelemetry.trace.set_tracer_provider", side_effect=captured.append),
    ):
        from searchess_ai.telemetry import configure_tracing
        configure_tracing()

    provider = captured[0]
    assert provider.resource.attributes[SERVICE_NAME] == "my-custom-service"


def test_configure_tracing_uses_default_service_name_when_env_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)

    from opentelemetry.sdk.resources import SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider

    captured: list[TracerProvider] = []

    with (
        patch("searchess_ai.telemetry.OTLPSpanExporter", return_value=MagicMock()),
        patch("opentelemetry.trace.set_tracer_provider", side_effect=captured.append),
    ):
        from searchess_ai.telemetry import configure_tracing
        configure_tracing()

    provider = captured[0]
    assert provider.resource.attributes[SERVICE_NAME] == "searchess-python-ai-service"


# ---------------------------------------------------------------------------
# Inference endpoint smoke tests — tracing must not break existing behaviour
# ---------------------------------------------------------------------------

_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_VALID_PAYLOAD: dict = {
    "requestId": "550e8400-e29b-41d4-a716-446655440000",
    "gameId": "123e4567-e89b-12d3-a456-426614174000",
    "sessionId": "789abcde-f012-3456-b789-abcdef012345",
    "sideToMove": "white",
    "fen": _START_FEN,
    "legalMoves": [
        {"from": "e2", "to": "e4"},
        {"from": "d2", "to": "d4"},
    ],
    "limits": {"timeoutMillis": 3000},
}


def test_inference_endpoint_unaffected_by_absent_otel_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference must work normally when no OTEL endpoint is configured."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    response = client.post("/v1/move-suggestions", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "requestId" in body
    assert "move" in body


def test_health_endpoint_unaffected_by_absent_otel_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
