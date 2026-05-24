"""Tests for two-frame adjudication tool."""
from __future__ import annotations

import json
from pathlib import Path

from tools.adjudicate import (
    load_prelabels,
    write_recall_adjudication,
    write_precision_verification,
)


class TestLoadPrelabels:
    def test_loads_jsonl(self, tmp_path: Path) -> None:
        line = json.dumps({
            "incident_id": "GA-001",
            "model_votes": [
                {"model_id": "A", "entry_id": "LLM01", "confidence": 0.9, "rationale": "x"},
            ],
            "consensus": "LLM01",
            "agreement": "1-of-1",
            "triage_tier": "agree",
        })
        path = tmp_path / "prelabels.jsonl"
        path.write_text(line + "\n")

        results = load_prelabels(path)
        assert len(results) == 1
        assert results[0]["incident_id"] == "GA-001"

    def test_sorts_by_triage_tier(self, tmp_path: Path) -> None:
        lines = [
            json.dumps({"incident_id": "GA-001", "triage_tier": "disagree",
                         "model_votes": [], "consensus": None, "agreement": ""}),
            json.dumps({"incident_id": "GA-002", "triage_tier": "agree",
                         "model_votes": [], "consensus": "LLM01", "agreement": ""}),
            json.dumps({"incident_id": "GA-003", "triage_tier": "split",
                         "model_votes": [], "consensus": "LLM01", "agreement": ""}),
        ]
        path = tmp_path / "prelabels.jsonl"
        path.write_text("\n".join(lines) + "\n")

        results = load_prelabels(path)
        tiers = [r["triage_tier"] for r in results]
        assert tiers == ["agree", "split", "disagree"]


class TestWriteRecallAdjudication:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "adjudicated.jsonl"
        write_recall_adjudication(
            out,
            incident_id="GA-001",
            llm_consensus="LLM01",
            adjudicated="accept",
            labels=["LLM01"],
            blind_label="LLM01",
            notes=None,
        )
        data = json.loads(out.read_text().strip())
        assert data["incident_id"] == "GA-001"
        assert data["adjudicated"] == "accept"
        assert data["blind_label"] == "LLM01"


class TestWritePrecisionVerification:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "precision.jsonl"
        write_precision_verification(
            out,
            incident_id="GA-002",
            claimed_entry_id="LLM06",
            is_correct=True,
            source="stage2-verified",
            adjudicator_id="RL",
        )
        data = json.loads(out.read_text().strip())
        assert data["is_correct"] is True
        assert data["claimed_entry_id"] == "LLM06"
