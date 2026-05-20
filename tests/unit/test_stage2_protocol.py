"""Unit tests for engine.classify.stage2_protocol."""

from __future__ import annotations

import pytest

from engine.classify.stage2_protocol import Stage2Classification, Stage2Protocol
from engine.schema import IncidentRecord


def make_incident() -> IncidentRecord:
    return IncidentRecord(
        id="TEST-001",
        date="2025-01-01",
        text="Test incident",
        severity="High",
        source_class="advisory",
        corpus_stratum="stratum_a",
        quality="curated",
        native_labels=("LLM01",),
        source_url="https://example.com",
    )


class TestStage2ClassificationDataclass:
    def test_fields_round_trip(self) -> None:
        c = Stage2Classification(
            incident_id="TEST-001",
            entry_id="LLM01",
            confidence=0.92,
            rationale="Matched LLM supply-chain pattern",
            model_identity="meta-llama/Llama-3.1-70B-Instruct",
            weight_provenance_hash="abc123",
            prompt_hash="def456",
        )
        assert c.incident_id == "TEST-001"
        assert c.entry_id == "LLM01"
        assert c.confidence == 0.92
        assert c.rationale == "Matched LLM supply-chain pattern"
        assert c.model_identity == "meta-llama/Llama-3.1-70B-Instruct"
        assert c.weight_provenance_hash == "abc123"
        assert c.prompt_hash == "def456"

    def test_frozen(self) -> None:
        c = Stage2Classification(
            incident_id="TEST-001",
            entry_id="LLM01",
            confidence=0.9,
            rationale="r",
            model_identity="m",
            weight_provenance_hash="w",
            prompt_hash="p",
        )
        with pytest.raises((AttributeError, TypeError)):
            c.entry_id = "OTHER"  # type: ignore[misc]


class TestStage2ProtocolInstantiation:
    def test_can_be_instantiated(self) -> None:
        proto = Stage2Protocol()
        assert proto is not None


class TestStage2ProtocolClassify:
    def test_classify_raises_not_implemented(self) -> None:
        proto = Stage2Protocol()
        inc = make_incident()
        with pytest.raises(NotImplementedError) as exc_info:
            proto.classify(inc, rubric_hash="rubric-sha256")
        assert "Plan 5 deliverable" in str(exc_info.value)
        assert "protocol stub defines the interface only" in str(exc_info.value)

    def test_classify_batch_raises_not_implemented(self) -> None:
        proto = Stage2Protocol()
        inc = make_incident()
        with pytest.raises(NotImplementedError) as exc_info:
            proto.classify_batch((inc,), rubric_hash="rubric-sha256")
        assert "Plan 5 deliverable" in str(exc_info.value)
        assert "protocol stub defines the interface only" in str(exc_info.value)

    def test_classify_empty_batch_raises_not_implemented(self) -> None:
        proto = Stage2Protocol()
        with pytest.raises(NotImplementedError):
            proto.classify_batch((), rubric_hash="rubric-sha256")
