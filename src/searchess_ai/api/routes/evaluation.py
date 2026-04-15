from __future__ import annotations

from fastapi import APIRouter, Depends

from searchess_ai.api.dependencies import (
    get_get_evaluation_job_status_use_case,
    get_submit_evaluation_job_use_case,
)
from searchess_ai.api.dto.evaluation import (
    EvaluationJobAcceptedDto,
    EvaluationJobRequestDto,
    EvaluationJobStatusDto,
    EvaluationResultDto,
    ScoreSummaryDto,
)
from searchess_ai.application.usecase.get_evaluation_job_status import GetEvaluationJobStatusUseCase
from searchess_ai.application.usecase.submit_evaluation_job import SubmitEvaluationJobUseCase
from searchess_ai.domain.evaluation import EvaluationJobId, EvaluationRequest
from searchess_ai.domain.model import ModelId, ModelVersion

router = APIRouter(tags=["evaluation"])


@router.post("/evaluation/jobs", response_model=EvaluationJobAcceptedDto, status_code=202)
def submit_evaluation_job(
    request_dto: EvaluationJobRequestDto,
    use_case: SubmitEvaluationJobUseCase = Depends(get_submit_evaluation_job_use_case),
) -> EvaluationJobAcceptedDto:
    request = EvaluationRequest(
        candidate_model_id=ModelId(request_dto.candidate_model_id),
        candidate_model_version=ModelVersion(request_dto.candidate_model_version),
        baseline_model_id=ModelId(request_dto.baseline_model_id),
        baseline_model_version=ModelVersion(request_dto.baseline_model_version),
        number_of_games=request_dto.number_of_games,
        notes=request_dto.notes,
    )
    accepted = use_case.execute(request)
    return EvaluationJobAcceptedDto(
        evaluation_job_id=accepted.evaluation_job_id.value,
        candidate_model_id=accepted.candidate_model_id.value,
        candidate_model_version=accepted.candidate_model_version.value,
        baseline_model_id=accepted.baseline_model_id.value,
        baseline_model_version=accepted.baseline_model_version.value,
        number_of_games=accepted.number_of_games,
        status=accepted.status.value,
        submitted_at=accepted.submitted_at,
    )


@router.get("/evaluation/jobs/{evaluation_job_id}", response_model=EvaluationJobStatusDto)
def get_evaluation_job(
    evaluation_job_id: str,
    use_case: GetEvaluationJobStatusUseCase = Depends(get_get_evaluation_job_status_use_case),
) -> EvaluationJobStatusDto:
    job = use_case.execute(EvaluationJobId(evaluation_job_id))
    result_dto: EvaluationResultDto | None = None
    if job.result is not None:
        result_dto = EvaluationResultDto(
            games_played=job.result.games_played,
            candidate_wins=job.result.candidate_wins,
            baseline_wins=job.result.baseline_wins,
            draws=job.result.draws,
            score_summary=ScoreSummaryDto(
                candidate_score=job.result.score_summary.candidate_score,
                baseline_score=job.result.score_summary.baseline_score,
            ),
        )
    return EvaluationJobStatusDto(
        evaluation_job_id=job.evaluation_job_id.value,
        candidate_model_id=job.candidate_model_id.value,
        candidate_model_version=job.candidate_model_version.value,
        baseline_model_id=job.baseline_model_id.value,
        baseline_model_version=job.baseline_model_version.value,
        number_of_games=job.number_of_games,
        status=job.status.value,
        submitted_at=job.submitted_at,
        updated_at=job.updated_at,
        result=result_dto,
        notes=job.notes,
        failure_reason=job.failure_reason,
    )
