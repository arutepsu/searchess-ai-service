import pytest

from searchess_ai.domain.evaluation import (
    EvaluationJobId,
    EvaluationJobStatus,
    EvaluationRequest,
    EvaluationResult,
    ScoreSummary,
)
from searchess_ai.domain.model import ModelId, ModelVersion


def test_evaluation_job_id_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        EvaluationJobId("")


def test_evaluation_job_id_rejects_whitespace_only() -> None:
    with pytest.raises(ValueError):
        EvaluationJobId("   ")


def test_evaluation_job_id_accepts_valid_value() -> None:
    assert EvaluationJobId("eval-123").value == "eval-123"


def test_evaluation_job_status_has_expected_values() -> None:
    assert EvaluationJobStatus.QUEUED.value == "queued"
    assert EvaluationJobStatus.RUNNING.value == "running"
    assert EvaluationJobStatus.COMPLETED.value == "completed"
    assert EvaluationJobStatus.FAILED.value == "failed"
    assert EvaluationJobStatus.CANCELLED.value == "cancelled"


def test_evaluation_request_rejects_zero_games() -> None:
    with pytest.raises(ValueError, match="number_of_games"):
        EvaluationRequest(
            candidate_model_id=ModelId("alpha-v1"),
            candidate_model_version=ModelVersion("1.0.0"),
            baseline_model_id=ModelId("legacy-v0"),
            baseline_model_version=ModelVersion("0.1.0"),
            number_of_games=0,
        )


def test_evaluation_request_rejects_negative_games() -> None:
    with pytest.raises(ValueError):
        EvaluationRequest(
            candidate_model_id=ModelId("alpha-v1"),
            candidate_model_version=ModelVersion("1.0.0"),
            baseline_model_id=ModelId("legacy-v0"),
            baseline_model_version=ModelVersion("0.1.0"),
            number_of_games=-10,
        )


def test_score_summary_rejects_negative_candidate_score() -> None:
    with pytest.raises(ValueError, match="candidate_score"):
        ScoreSummary(candidate_score=-1.0, baseline_score=5.0)


def test_score_summary_rejects_negative_baseline_score() -> None:
    with pytest.raises(ValueError, match="baseline_score"):
        ScoreSummary(candidate_score=5.0, baseline_score=-0.5)


def test_evaluation_result_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError, match="games_played"):
        EvaluationResult(
            games_played=10,
            candidate_wins=4,
            baseline_wins=4,
            draws=1,  # 4+4+1 = 9, not 10
            score_summary=ScoreSummary(candidate_score=4.5, baseline_score=4.5),
        )


def test_evaluation_result_rejects_negative_wins() -> None:
    with pytest.raises(ValueError):
        EvaluationResult(
            games_played=10,
            candidate_wins=-1,
            baseline_wins=6,
            draws=5,
            score_summary=ScoreSummary(candidate_score=1.5, baseline_score=8.5),
        )


def test_evaluation_result_accepts_valid_data() -> None:
    result = EvaluationResult(
        games_played=10,
        candidate_wins=4,
        baseline_wins=4,
        draws=2,
        score_summary=ScoreSummary(candidate_score=5.0, baseline_score=5.0),
    )
    assert result.games_played == 10
    assert result.candidate_wins + result.baseline_wins + result.draws == result.games_played
