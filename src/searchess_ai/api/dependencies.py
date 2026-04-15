from __future__ import annotations

from searchess_ai.application.usecase.choose_move import ChooseMoveUseCase
from searchess_ai.infrastructure.inference.fake_inference_engine import FakeInferenceEngine


def get_choose_move_use_case() -> ChooseMoveUseCase:
    return ChooseMoveUseCase(inference_engine=FakeInferenceEngine())
