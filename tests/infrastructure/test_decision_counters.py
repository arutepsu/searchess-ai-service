"""Tests for _DecisionCounters in supervised_inference_engine.py.

These tests enforce that aggregate decision-mode tracking, per-reason
breakdown, derived rates, and periodic snapshot logging all work correctly.
"""

from __future__ import annotations

import logging

import pytest

from searchess_ai.infrastructure.inference.supervised_inference_engine import _DecisionCounters
from searchess_ai.model_runtime.decision import DecisionMode, FallbackReason


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestDecisionCounterInitialState:
    def test_all_counts_are_zero(self):
        snap = _DecisionCounters().snapshot()
        for key in (
            "total_requests",
            "model_decision_count",
            "fallback_count",
            "error_recovered_count",
            "model_failure_count",
            "invalid_output_count",
            "runtime_exception_count",
            "no_legal_moves_count",
        ):
            assert snap[key] == 0, f"{key} should be 0 at init"

    def test_snapshot_contains_all_required_keys(self):
        snap = _DecisionCounters().snapshot()
        required = {
            "total_requests",
            "model_decision_count",
            "fallback_count",
            "error_recovered_count",
            "model_failure_count",
            "invalid_output_count",
            "runtime_exception_count",
            "no_legal_moves_count",
            "model_decision_rate",
            "fallback_rate",
            "invalid_output_rate",
            "model_failure_rate",
        }
        assert required.issubset(snap.keys())

    def test_rates_are_zero_before_any_request(self):
        snap = _DecisionCounters().snapshot()
        for key in ("model_decision_rate", "fallback_rate", "invalid_output_rate", "model_failure_rate"):
            assert snap[key] == 0.0, f"{key} should be 0.0 before any requests"


# ---------------------------------------------------------------------------
# Decision mode tracking
# ---------------------------------------------------------------------------


class TestDecisionModeTracking:
    def test_model_decision_increments_total_and_model_count(self):
        c = _DecisionCounters()
        c.record(DecisionMode.MODEL)
        c.record(DecisionMode.MODEL)
        snap = c.snapshot()
        assert snap["total_requests"] == 2
        assert snap["model_decision_count"] == 2
        assert snap["fallback_count"] == 0
        assert snap["error_recovered_count"] == 0

    def test_fallback_increments_fallback_count(self):
        c = _DecisionCounters()
        c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        snap = c.snapshot()
        assert snap["total_requests"] == 1
        assert snap["fallback_count"] == 1
        assert snap["model_decision_count"] == 0

    def test_error_recovered_increments_error_recovered_count(self):
        c = _DecisionCounters()
        c.record(DecisionMode.ERROR_RECOVERED, FallbackReason.RUNTIME_EXCEPTION)
        snap = c.snapshot()
        assert snap["error_recovered_count"] == 1
        assert snap["fallback_count"] == 0

    def test_mixed_decisions_tracked_independently(self):
        c = _DecisionCounters()
        c.record(DecisionMode.MODEL)
        c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        c.record(DecisionMode.ERROR_RECOVERED, FallbackReason.RUNTIME_EXCEPTION)
        snap = c.snapshot()
        assert snap["total_requests"] == 3
        assert snap["model_decision_count"] == 1
        assert snap["fallback_count"] == 1
        assert snap["error_recovered_count"] == 1


# ---------------------------------------------------------------------------
# Per-reason breakdown
# ---------------------------------------------------------------------------


class TestFallbackReasonBreakdown:
    def test_model_failure_counter(self):
        c = _DecisionCounters()
        c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        assert c.snapshot()["model_failure_count"] == 2

    def test_invalid_output_counter(self):
        c = _DecisionCounters()
        c.record(DecisionMode.FALLBACK, FallbackReason.INVALID_OUTPUT)
        assert c.snapshot()["invalid_output_count"] == 1

    def test_runtime_exception_counter(self):
        c = _DecisionCounters()
        c.record(DecisionMode.ERROR_RECOVERED, FallbackReason.RUNTIME_EXCEPTION)
        assert c.snapshot()["runtime_exception_count"] == 1

    def test_no_legal_moves_counter(self):
        c = _DecisionCounters()
        c.record(DecisionMode.FALLBACK, FallbackReason.NO_LEGAL_MOVES)
        assert c.snapshot()["no_legal_moves_count"] == 1

    def test_per_reason_counters_are_independent(self):
        c = _DecisionCounters()
        c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        c.record(DecisionMode.FALLBACK, FallbackReason.INVALID_OUTPUT)
        c.record(DecisionMode.FALLBACK, FallbackReason.INVALID_OUTPUT)
        c.record(DecisionMode.FALLBACK, FallbackReason.NO_LEGAL_MOVES)
        c.record(DecisionMode.ERROR_RECOVERED, FallbackReason.RUNTIME_EXCEPTION)
        snap = c.snapshot()
        assert snap["model_failure_count"] == 1
        assert snap["invalid_output_count"] == 2
        assert snap["no_legal_moves_count"] == 1
        assert snap["runtime_exception_count"] == 1


# ---------------------------------------------------------------------------
# Derived rates
# ---------------------------------------------------------------------------


class TestDerivedRates:
    def test_model_decision_rate(self):
        c = _DecisionCounters()
        for _ in range(8):
            c.record(DecisionMode.MODEL)
        for _ in range(2):
            c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        snap = c.snapshot()
        assert snap["total_requests"] == 10
        assert snap["model_decision_rate"] == 0.8
        assert snap["fallback_rate"] == 0.2

    def test_fallback_rate_includes_error_recovered(self):
        c = _DecisionCounters()
        c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        c.record(DecisionMode.ERROR_RECOVERED, FallbackReason.RUNTIME_EXCEPTION)
        snap = c.snapshot()
        # fallback_rate = (fallback_count + error_recovered_count) / total
        assert snap["fallback_rate"] == 1.0

    def test_invalid_output_rate(self):
        c = _DecisionCounters()
        c.record(DecisionMode.MODEL)
        c.record(DecisionMode.MODEL)
        c.record(DecisionMode.FALLBACK, FallbackReason.INVALID_OUTPUT)
        snap = c.snapshot()
        assert round(snap["invalid_output_rate"], 4) == round(1 / 3, 4)

    def test_model_failure_rate(self):
        c = _DecisionCounters()
        for _ in range(5):
            c.record(DecisionMode.MODEL)
        c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        snap = c.snapshot()
        assert round(snap["model_failure_rate"], 4) == round(1 / 6, 4)

    def test_rates_sum_consistently(self):
        """model_decision_rate + fallback_rate must equal 1.0 over all requests."""
        c = _DecisionCounters()
        for _ in range(7):
            c.record(DecisionMode.MODEL)
        for _ in range(3):
            c.record(DecisionMode.FALLBACK, FallbackReason.INVALID_OUTPUT)
        snap = c.snapshot()
        total_rate = round(snap["model_decision_rate"] + snap["fallback_rate"], 10)
        assert total_rate == 1.0

    def test_all_fallback_gives_zero_model_rate(self):
        c = _DecisionCounters()
        for _ in range(5):
            c.record(DecisionMode.FALLBACK, FallbackReason.MODEL_FAILURE)
        snap = c.snapshot()
        assert snap["model_decision_rate"] == 0.0
        assert snap["fallback_rate"] == 1.0


# ---------------------------------------------------------------------------
# Periodic snapshot logging
# ---------------------------------------------------------------------------


class TestPeriodicSnapshotLogging:
    def test_snapshot_logged_exactly_at_interval(self, caplog):
        c = _DecisionCounters()
        c.LOG_INTERVAL = 5
        with caplog.at_level(logging.INFO, logger="searchess_ai.infrastructure.inference.supervised_inference_engine"):
            for _ in range(5):
                c.record(DecisionMode.MODEL)
        snapshot_logs = [r for r in caplog.records if "counters_snapshot" in r.message]
        assert len(snapshot_logs) == 1

    def test_snapshot_not_logged_before_interval(self, caplog):
        c = _DecisionCounters()
        c.LOG_INTERVAL = 10
        with caplog.at_level(logging.INFO, logger="searchess_ai.infrastructure.inference.supervised_inference_engine"):
            for _ in range(9):
                c.record(DecisionMode.MODEL)
        snapshot_logs = [r for r in caplog.records if "counters_snapshot" in r.message]
        assert len(snapshot_logs) == 0

    def test_snapshot_log_is_valid_json_with_event_key(self, caplog):
        import json

        c = _DecisionCounters()
        c.LOG_INTERVAL = 3
        with caplog.at_level(logging.INFO, logger="searchess_ai.infrastructure.inference.supervised_inference_engine"):
            for _ in range(3):
                c.record(DecisionMode.MODEL)
        snapshot_logs = [r for r in caplog.records if "counters_snapshot" in r.message]
        assert len(snapshot_logs) == 1
        payload = json.loads(snapshot_logs[0].message)
        assert payload["event"] == "counters_snapshot"
        assert "total_requests" in payload
        assert "fallback_rate" in payload
