"""Tests for engine.calibrate.tally — aggregation of coded labels."""
from __future__ import annotations

from engine.calibrate.batch import BatchHeader, BatchIncident, CodingBatch
from engine.calibrate.tally import (
    tally_batches,
)


def _precision_batch(
    entry_id: str = "LLM01",
    stratum: str = "security",
    incidents: list[dict] | None = None,
) -> CodingBatch:
    if incidents is None:
        incidents = [
            {"incident_id": "GA-001", "labels": ["LLM01"], "text": "t1"},
            {"incident_id": "GA-002", "labels": [], "text": "t2"},
            {"incident_id": "GA-003", "labels": ["LLM01", "LLM05"], "text": "t3"},
        ]
    return CodingBatch(
        header=BatchHeader(
            cycle_id="2026", batch_id=f"precision-{entry_id}-{stratum}",
            frame="precision", entry_id=entry_id, stratum=stratum,
            sample_hash="h", rubric_hash="r", manifest_lock_hash="l",
            coder_id="rock",
        ),
        incidents=[
            BatchIncident(
                incident_id=inc["incident_id"],
                text=inc["text"],
                labels=inc["labels"],
            )
            for inc in incidents
        ],
    )


def _recall_batch(
    stratum: str = "security",
    incidents: list[dict] | None = None,
) -> CodingBatch:
    if incidents is None:
        incidents = [
            {"incident_id": "GA-100", "labels": ["LLM01", "LLM05"], "text": "t"},
            {"incident_id": "GA-101", "labels": [], "text": "t"},
            {"incident_id": "GA-102", "labels": ["LLM09"], "text": "t"},
        ]
    return CodingBatch(
        header=BatchHeader(
            cycle_id="2026", batch_id=f"recall-all-{stratum}",
            frame="recall", entry_id=None, stratum=stratum,
            sample_hash="h", rubric_hash="r", manifest_lock_hash="l",
            coder_id="rock",
        ),
        incidents=[
            BatchIncident(
                incident_id=inc["incident_id"],
                text=inc["text"],
                labels=inc["labels"],
            )
            for inc in incidents
        ],
    )


class TestTally:
    def test_precision_counts(self) -> None:
        batch = _precision_batch("LLM01", "security")
        result = tally_batches([batch])
        key = ("LLM01", "security")
        assert key in result.precision_counts
        pt = result.precision_counts[key]
        # GA-001: LLM01 in labels → TP. GA-002: [] → FP. GA-003: LLM01 in labels → TP.
        assert pt.true_positives == 2
        assert pt.false_positives == 1
        assert pt.total == 3

    def test_recall_counts(self) -> None:
        batch = _recall_batch("security")
        result = tally_batches([batch])
        # LLM01: GA-100 has it → TP, GA-101 no → FN, GA-102 no → FN
        key_01 = ("LLM01", "security")
        assert key_01 in result.recall_counts
        rt = result.recall_counts[key_01]
        assert rt.true_positives == 1
        assert rt.false_negatives == 2
        assert rt.total_in_sample == 3

    def test_total_coded(self) -> None:
        batch = _precision_batch()
        result = tally_batches([batch])
        assert result.total_coded == 3

    def test_skips_null_labels(self) -> None:
        batch = CodingBatch(
            header=BatchHeader(
                cycle_id="2026", batch_id="p", frame="precision",
                entry_id="LLM01", stratum="security",
                sample_hash="h", rubric_hash="r", manifest_lock_hash="l",
                coder_id="rock",
            ),
            incidents=[
                BatchIncident(incident_id="GA-001", text="t", labels=None),
                BatchIncident(incident_id="GA-002", text="t", labels=["LLM01"]),
            ],
        )
        result = tally_batches([batch])
        assert result.precision_counts[("LLM01", "security")].total == 1
