from __future__ import annotations

from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.application.usecase.get_evaluation_job_status import GetEvaluationJobStatusUseCase
from searchess_ai.application.usecase.get_model_detail import GetModelDetailUseCase
from searchess_ai.application.usecase.get_training_job_status import GetTrainingJobStatusUseCase
from searchess_ai.application.usecase.list_models import ListModelsUseCase
from searchess_ai.application.usecase.submit_evaluation_job import SubmitEvaluationJobUseCase
from searchess_ai.application.usecase.submit_training_job import SubmitTrainingJobUseCase
from searchess_ai.infrastructure.evaluation.fake_evaluation_scheduler import FakeEvaluationScheduler
from searchess_ai.infrastructure.evaluation.in_memory_evaluation_job_repository import (
    InMemoryEvaluationJobRepository,
)
from searchess_ai.infrastructure.inference.fake_inference_engine import FakeInferenceEngine
from searchess_ai.infrastructure.model.in_memory_model_repository import InMemoryModelRepository
from searchess_ai.infrastructure.training.fake_training_scheduler import FakeTrainingScheduler
from searchess_ai.infrastructure.training.in_memory_training_job_repository import (
    InMemoryTrainingJobRepository,
)

# Singletons — job use cases share one repository so POST and GET see the same store.
_training_job_repository = InMemoryTrainingJobRepository()
_training_scheduler = FakeTrainingScheduler()
_evaluation_job_repository = InMemoryEvaluationJobRepository()
_evaluation_scheduler = FakeEvaluationScheduler()


def get_choose_move_use_case() -> ChooseMoveUseCase:
    return ChooseMoveUseCase(inference_engine=FakeInferenceEngine())


def get_list_models_use_case() -> ListModelsUseCase:
    return ListModelsUseCase(repository=InMemoryModelRepository())


def get_get_model_detail_use_case() -> GetModelDetailUseCase:
    return GetModelDetailUseCase(repository=InMemoryModelRepository())


def get_submit_training_job_use_case() -> SubmitTrainingJobUseCase:
    return SubmitTrainingJobUseCase(
        repository=_training_job_repository,
        scheduler=_training_scheduler,
    )


def get_get_training_job_status_use_case() -> GetTrainingJobStatusUseCase:
    return GetTrainingJobStatusUseCase(repository=_training_job_repository)


def get_submit_evaluation_job_use_case() -> SubmitEvaluationJobUseCase:
    return SubmitEvaluationJobUseCase(
        repository=_evaluation_job_repository,
        scheduler=_evaluation_scheduler,
    )


def get_get_evaluation_job_status_use_case() -> GetEvaluationJobStatusUseCase:
    return GetEvaluationJobStatusUseCase(repository=_evaluation_job_repository)
