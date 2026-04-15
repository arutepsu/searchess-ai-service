import pytest

from searchess_ai.domain.training import (
    TrainingJobId,
    TrainingJobStatus,
    TrainingProfile,
)


def test_training_job_id_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        TrainingJobId("")


def test_training_job_id_rejects_whitespace_only() -> None:
    with pytest.raises(ValueError):
        TrainingJobId("   ")


def test_training_job_id_accepts_valid_value() -> None:
    assert TrainingJobId("job-123").value == "job-123"


def test_training_job_status_has_expected_values() -> None:
    assert TrainingJobStatus.QUEUED.value == "queued"
    assert TrainingJobStatus.RUNNING.value == "running"
    assert TrainingJobStatus.COMPLETED.value == "completed"
    assert TrainingJobStatus.FAILED.value == "failed"
    assert TrainingJobStatus.CANCELLED.value == "cancelled"


def test_training_profile_has_expected_values() -> None:
    assert TrainingProfile.SUPERVISED.value == "supervised"
    assert TrainingProfile.REINFORCEMENT.value == "reinforcement"
    assert TrainingProfile.MIXED.value == "mixed"
