"""Tests for Stage-2 retry + fallback rate tracking."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.classify.cost_tracker import CostTracker
from engine.classify.runpod_client import RunPodError, RunPodResponse
from engine.classify.stage2 import FallbackRateExceeded, Stage2Classifier
from engine.schema import IncidentRecord

_RUBRIC_JSON = '{"entries": [{"entry_id": "LLM01", "canonical_name": "Prompt Injection", "in_scope": "test"}]}'


def _make_incident(incident_id: str = "TEST-001") -> IncidentRecord:
    return IncidentRecord(
        id=incident_id, date="2025-01-01", text="Test", severity="High",
        source_class="advisory", corpus_stratum="security", quality="curated",
        native_labels=(), source_url="https://example.com",
    )


def _make_classifier(client: MagicMock, fallback_rate_window: int = 100) -> Stage2Classifier:
    return Stage2Classifier(
        client=client,
        cost_tracker=CostTracker(ceiling_usd=100.0),
        rubric_json=_RUBRIC_JSON,
        model_identity="test-model",
        weight_provenance_hash="abc",
        prng_seed=42,
        fallback_rate_window=fallback_rate_window,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)


class TestRetryOnError:
    def test_retries_once_on_runpod_error(self) -> None:
        client = MagicMock()
        client.run_sync.side_effect = [
            RunPodError("transient"),
            RunPodResponse(
                output_text='{"entry_id": "LLM01", "confidence": 0.9, "rationale": "test"}',
                job_id="j1",
                execution_time_ms=100.0,
            ),
        ]
        classifier = _make_classifier(client)
        result = classifier.classify(_make_incident(), "hash")

        assert result.entry_id == "LLM01"
        assert client.run_sync.call_count == 2

    def test_falls_back_after_two_failures(self) -> None:
        client = MagicMock()
        client.run_sync.side_effect = RunPodError("persistent")
        classifier = _make_classifier(client)
        result = classifier.classify(_make_incident(), "hash")

        assert result.entry_id == "out-of-scope"
        assert client.run_sync.call_count == 2


class TestFallbackRateTracking:
    def test_tracks_fallback_count(self) -> None:
        client = MagicMock()
        client.run_sync.side_effect = RunPodError("fail")
        classifier = _make_classifier(client)
        classifier.classify(_make_incident("T-001"), "hash")
        classifier.classify(_make_incident("T-002"), "hash")

        assert classifier.fallback_count == 2
        assert classifier.total_count == 2

    def test_abort_on_high_fallback_rate(self) -> None:
        client = MagicMock()
        client.run_sync.side_effect = RunPodError("fail")
        classifier = _make_classifier(client, fallback_rate_window=10)

        for i in range(11):
            classifier.classify(_make_incident(f"T-{i:03d}"), "hash")

        with pytest.raises(FallbackRateExceeded):
            classifier.classify(_make_incident("T-011"), "hash")
