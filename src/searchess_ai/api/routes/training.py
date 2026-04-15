from __future__ import annotations

from fastapi import APIRouter, Depends

from searchess_ai.api.dependencies import (
    get_get_training_job_status_use_case,
    get_submit_training_job_use_case,
)
from searchess_ai.api.dto.training import (
    TrainingJobAcceptedDto,
    TrainingJobRequestDto,
    TrainingJobStatusDto,
)
from searchess_ai.application.usecase.get_training_job_status import GetTrainingJobStatusUseCase
from searchess_ai.application.usecase.submit_training_job import SubmitTrainingJobUseCase
from searchess_ai.domain.model import ModelId, ModelVersion
from searchess_ai.domain.training import TrainingJobId, TrainingJobRequest, TrainingProfile

router = APIRouter(tags=["training"])


@router.post("/training/jobs", response_model=TrainingJobAcceptedDto, status_code=202)
def submit_training_job(
    request_dto: TrainingJobRequestDto,
    use_case: SubmitTrainingJobUseCase = Depends(get_submit_training_job_use_case),
) -> TrainingJobAcceptedDto:
    request = TrainingJobRequest(
        training_profile=TrainingProfile(request_dto.training_profile),
        base_model_id=ModelId(request_dto.base_model_id) if request_dto.base_model_id else None,
        base_model_version=(
            ModelVersion(request_dto.base_model_version)
            if request_dto.base_model_version
            else None
        ),
        notes=request_dto.notes,
    )
    accepted = use_case.execute(request)
    return TrainingJobAcceptedDto(
        training_job_id=accepted.training_job_id.value,
        training_profile=accepted.training_profile.value,
        status=accepted.status.value,
        submitted_at=accepted.submitted_at,
    )


@router.get("/training/jobs/{training_job_id}", response_model=TrainingJobStatusDto)
def get_training_job(
    training_job_id: str,
    use_case: GetTrainingJobStatusUseCase = Depends(get_get_training_job_status_use_case),
) -> TrainingJobStatusDto:
    job = use_case.execute(TrainingJobId(training_job_id))
    return TrainingJobStatusDto(
        training_job_id=job.training_job_id.value,
        training_profile=job.training_profile.value,
        status=job.status.value,
        submitted_at=job.submitted_at,
        updated_at=job.updated_at,
        base_model_id=job.base_model_id.value if job.base_model_id else None,
        base_model_version=job.base_model_version.value if job.base_model_version else None,
        produced_model_id=job.produced_model_id.value if job.produced_model_id else None,
        produced_model_version=(
            job.produced_model_version.value if job.produced_model_version else None
        ),
        notes=job.notes,
        failure_reason=job.failure_reason,
    )
