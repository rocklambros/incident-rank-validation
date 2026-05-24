"""Tests for multi-model pre-labeling pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from engine.classify.multi_model import MultiModelPreLabeler, PreLabelResult
from engine.classify.runpod_client import RunPodResponse
from engine.schema import IncidentRecord

_RUBRIC_JSON = '{"entries": [{"entry_id": "LLM01", "canonical_name": "PI", "in_scope": "test"}, {"entry_id": "LLM06", "canonical_name": "EA", "in_scope": "test"}]}'


def _make_incident(incident_id: str = "GA-001") -> IncidentRecord:
    return IncidentRecord(
        id=incident_id, date="2025-01-01", text="Test", severity="High",
        source_class="advisory", corpus_stratum="security", quality="curated",
        native_labels=(), source_url="https://example.com",
    )


def _make_client(entry_id: str, confidence: float = 0.9) -> MagicMock:
    client = MagicMock()
    client.run_sync.return_value = RunPodResponse(
        output_text=json.dumps({
            "entry_id": entry_id,
            "confidence": confidence,
            "rationale": "test",
        }),
        job_id="j1",
        execution_time_ms=100.0,
    )
    return client


class TestPreLabel:
    def test_agree_tier_when_all_same(self) -> None:
        clients = [
            (_make_client("LLM01"), "model-A"),
            (_make_client("LLM01"), "model-B"),
            (_make_client("LLM01"), "model-C"),
        ]
        labeler = MultiModelPreLabeler(
            models=clients, rubric_json=_RUBRIC_JSON, prng_seed=42,
        )
        result = labeler.pre_label(_make_incident())

        assert result.consensus == "LLM01"
        assert result.triage_tier == "agree"
        assert len(result.model_votes) == 3

    def test_split_tier_when_two_agree(self) -> None:
        clients = [
            (_make_client("LLM01"), "model-A"),
            (_make_client("LLM01"), "model-B"),
            (_make_client("LLM06"), "model-C"),
        ]
        labeler = MultiModelPreLabeler(
            models=clients, rubric_json=_RUBRIC_JSON, prng_seed=42,
        )
        result = labeler.pre_label(_make_incident())

        assert result.consensus == "LLM01"
        assert result.triage_tier == "split"

    def test_disagree_tier_when_all_different(self) -> None:
        clients = [
            (_make_client("LLM01"), "model-A"),
            (_make_client("LLM06"), "model-B"),
            (_make_client("out-of-scope"), "model-C"),
        ]
        labeler = MultiModelPreLabeler(
            models=clients, rubric_json=_RUBRIC_JSON, prng_seed=42,
        )
        result = labeler.pre_label(_make_incident())

        assert result.triage_tier == "disagree"


class TestPreLabelBatch:
    def test_writes_checkpoint(self, tmp_path: Path) -> None:
        clients = [(_make_client("LLM01"), "model-A")]
        labeler = MultiModelPreLabeler(
            models=clients, rubric_json=_RUBRIC_JSON, prng_seed=42,
        )
        incidents = [_make_incident("GA-001"), _make_incident("GA-002")]
        checkpoint = tmp_path / "prelabels.jsonl"

        labeler.pre_label_batch(incidents, checkpoint)

        lines = checkpoint.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_resumes_from_checkpoint(self, tmp_path: Path) -> None:
        clients = [(_make_client("LLM01"), "model-A")]
        labeler = MultiModelPreLabeler(
            models=clients, rubric_json=_RUBRIC_JSON, prng_seed=42,
        )
        checkpoint = tmp_path / "prelabels.jsonl"
        existing = json.dumps({
            "incident_id": "GA-001",
            "model_votes": [{"model_id": "model-A", "entry_id": "LLM01",
                             "confidence": 0.9, "rationale": "test"}],
            "consensus": "LLM01", "agreement": "1-of-1", "triage_tier": "agree",
        })
        checkpoint.write_text(existing + "\n")

        incidents = [_make_incident("GA-001"), _make_incident("GA-002")]
        labeler.pre_label_batch(incidents, checkpoint)

        lines = checkpoint.read_text().strip().splitlines()
        assert len(lines) == 2
        assert clients[0][0].run_sync.call_count == 1
