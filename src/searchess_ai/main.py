from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from searchess_ai.api.app import create_app
from searchess_ai.telemetry import configure_tracing
import uvicorn

configure_tracing()
app = create_app()
FastAPIInstrumentor.instrument_app(app)

if __name__ == "__main__":
    uvicorn.run(
        "searchess_ai.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
