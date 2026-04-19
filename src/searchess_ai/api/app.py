from fastapi import FastAPI

from searchess_ai.api.errors import (
    InferenceContractError,
    evaluation_job_not_found_handler,
    generic_error_handler,
    inference_contract_error_handler,
    model_not_found_handler,
    openspiel_adapter_error_handler,
    training_job_not_found_handler,
    value_error_handler,
)
from searchess_ai.api.routes.evaluation import router as evaluation_router
from searchess_ai.api.routes.health import router as health_router
from searchess_ai.api.routes.inference import router as inference_router
from searchess_ai.api.routes.models import router as models_router
from searchess_ai.api.routes.training import router as training_router
from searchess_ai.domain.evaluation import EvaluationJobNotFoundError
from searchess_ai.domain.model import ModelNotFoundError
from searchess_ai.domain.training import TrainingJobNotFoundError
from searchess_ai.infrastructure.inference.openspiel_inference_engine import OpenSpielAdapterError


def create_app() -> FastAPI:
    app = FastAPI(
        title="searchess-ai-service",
        version="0.1.0",
        description="AI service for Searchess.",
    )

    # Contract-compliant inference routes at /v1
    app.include_router(inference_router, prefix="/v1")

    # Infrastructure health check at /health (no version prefix)
    app.include_router(health_router, prefix="")

    # Non-inference routes unchanged
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(training_router, prefix="/api/v1")
    app.include_router(evaluation_router, prefix="/api/v1")

    # Contract error handler for inference routes
    app.add_exception_handler(InferenceContractError, inference_contract_error_handler)

    # Legacy handlers for non-inference routes
    app.add_exception_handler(ModelNotFoundError, model_not_found_handler)
    app.add_exception_handler(TrainingJobNotFoundError, training_job_not_found_handler)
    app.add_exception_handler(EvaluationJobNotFoundError, evaluation_job_not_found_handler)
    app.add_exception_handler(OpenSpielAdapterError, openspiel_adapter_error_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    return app
