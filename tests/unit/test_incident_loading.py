from __future__ import annotations

import pytest

from engine.schema import IncidentRecord


def test_incident_record_requires_all_fields() -> None:
    """F1: IncidentRecord needs all 9 positional fields."""
    with pytest.raises(TypeError):
        IncidentRecord(  # type: ignore[call-arg]
            id="INC-001",
            text="test",
            corpus_stratum="stratum_a",
            native_labels=("LLM01",),
        )


def test_incident_record_with_all_fields() -> None:
    rec = IncidentRecord(
        id="INC-001",
        date="2025-06-15",
        text="test incident",
        severity="High",
        source_class="advisory",
        corpus_stratum="stratum_a",
        quality="curated",
        native_labels=("LLM01",),
        source_url="https://example.com/inc-001",
    )
    assert rec.id == "INC-001"
    assert rec.date == "2025-06-15"
    assert rec.severity == "High"
    assert rec.source_class == "advisory"
    assert rec.quality == "curated"
    assert rec.source_url == "https://example.com/inc-001"
