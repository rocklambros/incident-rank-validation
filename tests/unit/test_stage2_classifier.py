from __future__ import annotations

import json

from engine.classify.cost_tracker import CostTracker
from engine.classify.runpod_client import RunPodError, RunPodResponse
from engine.classify.stage2 import Stage2Classifier
from engine.classify.stage2_protocol import Stage2Classification
from engine.schema import IncidentRecord


class _MockClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def run_sync(self, prompt: str, seed: int) -> RunPodResponse:
        self.calls.append(prompt)
        if not self._responses:
            raise RunPodError("No responses left")
        return RunPodResponse(
            output_text=self._responses.pop(0),
            job_id=f"mock-{len(self.calls)}",
            execution_time_ms=100.0,
        )

    def close(self) -> None:
        pass


def _make_incident(incident_id: str = "INC-001") -> IncidentRecord:
    return IncidentRecord(
        id=incident_id,
        date="2026-01-15",
        text="LLM prompt injection via crafted input",
        severity="High",
        source_class="cve",
        corpus_stratum="security",
        quality="curated",
        native_labels=("LLM01",),
        source_url="https://example.com/CVE-2026-0001",
    )


_GOOD_RESPONSE = json.dumps({
    "entry_id": "LLM01",
    "confidence": 0.92,
    "rationale": "Clear prompt injection pattern"
})


class TestStage2Classifier:
    def test_classify_single(self) -> None:
        client = _MockClient([_GOOD_RESPONSE])
        tracker = CostTracker(ceiling_usd=500.0)
        classifier = Stage2Classifier(
            client=client,
            cost_tracker=tracker,
            rubric_json='{"entries": []}',
            model_identity="test-model",
            weight_provenance_hash="abc123",
            prng_seed=42,
            cost_per_job_usd=0.01,
        )
        result = classifier.classify(_make_incident(), rubric_hash="hash123")
        assert isinstance(result, Stage2Classification)
        assert result.entry_id == "LLM01"
        assert result.confidence == 0.92
        assert result.incident_id == "INC-001"
        assert result.model_identity == "test-model"

    def test_classify_batch(self) -> None:
        client = _MockClient([_GOOD_RESPONSE, _GOOD_RESPONSE])
        tracker = CostTracker(ceiling_usd=500.0)
        classifier = Stage2Classifier(
            client=client,
            cost_tracker=tracker,
            rubric_json='{"entries": []}',
            model_identity="test-model",
            weight_provenance_hash="abc123",
            prng_seed=42,
            cost_per_job_usd=0.01,
        )
        incidents = (_make_incident("INC-001"), _make_incident("INC-002"))
        results = classifier.classify_batch(incidents, rubric_hash="hash123")
        assert len(results) == 2
        assert results[0].incident_id == "INC-001"
        assert results[1].incident_id == "INC-002"

    def test_cost_tracking(self) -> None:
        client = _MockClient([_GOOD_RESPONSE])
        tracker = CostTracker(ceiling_usd=500.0)
        classifier = Stage2Classifier(
            client=client,
            cost_tracker=tracker,
            rubric_json='{"entries": []}',
            model_identity="test-model",
            weight_provenance_hash="abc123",
            prng_seed=42,
            cost_per_job_usd=0.05,
        )
        classifier.classify(_make_incident(), rubric_hash="hash123")
        assert tracker.total_cost_usd == 0.05
        assert tracker.job_count == 1

    def test_malformed_response_handled(self) -> None:
        client = _MockClient(["not valid json"])
        tracker = CostTracker(ceiling_usd=500.0)
        classifier = Stage2Classifier(
            client=client,
            cost_tracker=tracker,
            rubric_json='{"entries": []}',
            model_identity="test-model",
            weight_provenance_hash="abc123",
            prng_seed=42,
            cost_per_job_usd=0.01,
        )
        result = classifier.classify(_make_incident(), rubric_hash="hash123")
        assert result.entry_id == "out-of-scope"
        assert result.confidence == 0.0

    def test_prompt_hash_captured(self) -> None:
        client = _MockClient([_GOOD_RESPONSE])
        tracker = CostTracker(ceiling_usd=500.0)
        classifier = Stage2Classifier(
            client=client,
            cost_tracker=tracker,
            rubric_json='{"entries": []}',
            model_identity="test-model",
            weight_provenance_hash="abc123",
            prng_seed=42,
            cost_per_job_usd=0.01,
        )
        result = classifier.classify(_make_incident(), rubric_hash="hash123")
        assert len(result.prompt_hash) == 64
