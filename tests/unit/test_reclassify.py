"""Tests for engine.cli.reclassify helper functions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.schema import IncidentRecord


def _make_incident(iid: str, stratum: str = "security") -> IncidentRecord:
    return IncidentRecord(
        id=iid,
        date="2026-01-01",
        text=f"Incident {iid} text",
        severity="High",
        source_class="advisory",
        corpus_stratum=stratum,
        quality="auto",
        native_labels=(),
        source_url=f"https://example.com/{iid}",
    )


class TestConvertToLabeled:
    def test_merges_stage1_and_stage2(self, tmp_path: Path) -> None:
        from engine.cli.reclassify import _convert_to_labeled

        checkpoint = tmp_path / "checkpoint.jsonl"
        checkpoint.write_text(
            json.dumps({
                "incident_id": "INC-001",
                "model_votes": [
                    {"model_id": "m1", "entry_id": "NEW-MTIE", "confidence": 0.9, "rationale": "r1"},
                    {"model_id": "m2", "entry_id": "NEW-MTIE", "confidence": 0.85, "rationale": "r2"},
                ],
                "consensus": "NEW-MTIE",
                "agreement": "agree",
                "triage_tier": "agree",
            }) + "\n"
        )

        stage1 = {"INC-100": {"incident_id": "INC-100", "entry_id": "ROLL-CMSB", "stage": 1}}
        s2_incidents = [_make_incident("INC-001")]

        result = _convert_to_labeled(checkpoint, stage1, s2_incidents)

        assert len(result) == 2
        ids = {r["incident_id"] for r in result}
        assert ids == {"INC-100", "INC-001"}

        s2_result = next(r for r in result if r["incident_id"] == "INC-001")
        assert s2_result["entry_id"] == "NEW-MTIE"
        assert s2_result["stage"] == 2
        assert s2_result["confidence"] == 0.9
        assert "multi-model agree" in s2_result["rationale"]

    def test_null_consensus_defaults_to_out_of_scope(self, tmp_path: Path) -> None:
        from engine.cli.reclassify import _convert_to_labeled

        checkpoint = tmp_path / "checkpoint.jsonl"
        checkpoint.write_text(
            json.dumps({
                "incident_id": "INC-002",
                "model_votes": [],
                "consensus": None,
                "agreement": "disagree",
                "triage_tier": "disagree",
            }) + "\n"
        )

        result = _convert_to_labeled(checkpoint, {}, [_make_incident("INC-002")])
        assert result[0]["entry_id"] == "out-of-scope"

    def test_preserves_stratum(self, tmp_path: Path) -> None:
        from engine.cli.reclassify import _convert_to_labeled

        checkpoint = tmp_path / "checkpoint.jsonl"
        checkpoint.write_text(
            json.dumps({
                "incident_id": "INC-003",
                "model_votes": [
                    {"model_id": "m1", "entry_id": "ROLL-CFAS", "confidence": 0.7, "rationale": "r"},
                ],
                "consensus": "ROLL-CFAS",
                "agreement": "single",
                "triage_tier": "single",
            }) + "\n"
        )

        result = _convert_to_labeled(
            checkpoint, {},
            [_make_incident("INC-003", stratum="agentic")],
        )
        assert result[0]["stratum"] == "agentic"


class TestLoadModelConfigs:
    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from engine.cli.reclassify import _load_model_configs

        monkeypatch.setenv("RUNPOD_MODEL_1_ENDPOINT", "ep-aaa")
        monkeypatch.setenv("RUNPOD_MODEL_1_NAME", "llama-70b")
        monkeypatch.setenv("RUNPOD_MODEL_2_ENDPOINT", "ep-bbb")
        monkeypatch.setenv("RUNPOD_MODEL_2_NAME", "qwen-72b")
        monkeypatch.delenv("RUNPOD_MODEL_3_ENDPOINT", raising=False)
        monkeypatch.delenv("RUNPOD_MODEL_3_NAME", raising=False)

        configs = _load_model_configs()
        assert len(configs) == 2
        assert configs[0] == ("llama-70b", "ep-aaa")
        assert configs[1] == ("qwen-72b", "ep-bbb")

    def test_empty_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from engine.cli.reclassify import _load_model_configs

        for i in range(1, 10):
            monkeypatch.delenv(f"RUNPOD_MODEL_{i}_ENDPOINT", raising=False)
            monkeypatch.delenv(f"RUNPOD_MODEL_{i}_NAME", raising=False)

        assert _load_model_configs() == []
