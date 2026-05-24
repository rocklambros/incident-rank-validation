"""Tests for gold calibration schema dataclasses."""
from __future__ import annotations

import pytest

from engine.calibrate.gold_schema import (
    GoldCalibration,
    GoldPrecisionLabel,
    GoldRecallLabel,
)


class TestGoldRecallLabel:
    def test_construction(self) -> None:
        label = GoldRecallLabel(
            incident_id="GA-04821",
            true_entry_ids=["LLM01"],
            classifier_entry_id="LLM01",
            source="manual-curated",
        )
        assert label.incident_id == "GA-04821"
        assert label.true_entry_ids == ["LLM01"]
        assert label.source == "manual-curated"

    def test_multi_label(self) -> None:
        label = GoldRecallLabel(
            incident_id="GA-07312",
            true_entry_ids=["LLM01", "LLM05"],
            classifier_entry_id="LLM05",
            source="llm-adjudicated",
        )
        assert len(label.true_entry_ids) == 2


class TestGoldPrecisionLabel:
    def test_correct_classification(self) -> None:
        label = GoldPrecisionLabel(
            incident_id="GA-04821",
            claimed_entry_id="LLM06",
            is_correct=True,
            source="stage2-verified",
        )
        assert label.is_correct is True

    def test_incorrect_classification(self) -> None:
        label = GoldPrecisionLabel(
            incident_id="GA-02198",
            claimed_entry_id="LLM09",
            is_correct=False,
            source="llm-prelabel-verified",
        )
        assert label.is_correct is False


class TestGoldCalibration:
    def test_construction_with_both_frames(self) -> None:
        recall = [
            GoldRecallLabel("GA-001", ["LLM01"], "LLM01", "manual-curated"),
        ]
        precision = [
            GoldPrecisionLabel("GA-002", "LLM06", True, "stage2-verified"),
        ]
        gold = GoldCalibration(
            recall_labels=recall,
            precision_labels=precision,
            provenance_hash="abc123",
            rubric_hash="def456",
            adjudicator_id="RL",
            session_count=1,
        )
        assert len(gold.recall_labels) == 1
        assert len(gold.precision_labels) == 1
        assert gold.adjudicator_id == "RL"

    def test_empty_precision_allowed(self) -> None:
        gold = GoldCalibration(
            recall_labels=[
                GoldRecallLabel("GA-001", ["LLM01"], None, "manual-curated"),
            ],
            precision_labels=[],
            provenance_hash="abc123",
            rubric_hash="def456",
            adjudicator_id="RL",
            session_count=1,
        )
        assert gold.precision_labels == []

    def test_rubric_hash_required(self) -> None:
        with pytest.raises(TypeError):
            GoldCalibration(
                recall_labels=[],
                precision_labels=[],
                provenance_hash="abc123",
                adjudicator_id="RL",
                session_count=1,
            )  # type: ignore[call-arg]
