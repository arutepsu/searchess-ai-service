from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing() -> None:
    """Initialize OTel tracing from OTEL_* environment variables.

    No-op when OTEL_EXPORTER_OTLP_ENDPOINT is absent so the service starts
    cleanly in local development without a collector running.

    OTLPSpanExporter reads OTEL_EXPORTER_OTLP_ENDPOINT and
    OTEL_EXPORTER_OTLP_PROTOCOL from the environment automatically.
    """
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    service_name = os.getenv("OTEL_SERVICE_NAME", "searchess-python-ai-service")
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
