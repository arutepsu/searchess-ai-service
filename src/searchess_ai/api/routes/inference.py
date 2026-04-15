from __future__ import annotations

from fastapi import APIRouter

from searchess_ai.api.dto.inference import (
    MoveInferenceRequestDto,
    MoveInferenceResponseDto,
)
from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.domain.game import LegalMoveSet, Move, Position, SideToMove
from searchess_ai.domain.inference import InferenceRequest
from searchess_ai.domain.model import ModelId, ModelVersion, PolicyProfile
from searchess_ai.infrastructure.inference.fake_inference_engine import FakeInferenceEngine

router = APIRouter(tags=["inference"])

_use_case = ChooseMoveUseCase(inference_engine=FakeInferenceEngine())


@router.post("/inference/move", response_model=MoveInferenceResponseDto)
def choose_move(request_dto: MoveInferenceRequestDto) -> MoveInferenceResponseDto:
    request = InferenceRequest(
        request_id=request_dto.request_id,
        match_id=request_dto.match_id,
        position=Position(request_dto.board_state),
        side_to_move=SideToMove(request_dto.side_to_move),
        legal_moves=LegalMoveSet(tuple(Move(move) for move in request_dto.legal_moves)),
        policy_profile=(
            PolicyProfile(request_dto.policy_profile)
            if request_dto.policy_profile is not None
            else None
        ),
        model_id=ModelId(request_dto.model_id) if request_dto.model_id is not None else None,
        model_version=(
            ModelVersion(request_dto.model_version)
            if request_dto.model_version is not None
            else None
        ),
        remaining_time_millis=request_dto.remaining_time_millis,
    )

    decision = _use_case.execute(request)

    return MoveInferenceResponseDto(
        request_id=decision.request_id,
        decision_type=decision.decision_type.value,
        selected_move=decision.selected_move.value,
        model_id=decision.model_id.value,
        model_version=decision.model_version.value,
        decision_time_millis=decision.decision_time_millis,
        policy_profile=decision.policy_profile.value if decision.policy_profile else None,
        confidence=decision.confidence,
    )