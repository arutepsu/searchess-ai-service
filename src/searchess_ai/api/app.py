from fastapi import FastAPI

from searchess_ai.api.routes.health import router as health_router
from searchess_ai.api.routes.inference import router as inference_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="searchess-ai-service",
        version="0.1.0",
        description="AI service for Searchess.",
    )

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(inference_router, prefix="/api/v1")

    return app