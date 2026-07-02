"""Tests for engine/classify/bakeoff_inputs.py (Task 4 loaders)."""
from __future__ import annotations

import json
from pathlib import Path

from engine.classify.bakeoff_inputs import (
    compute_corpus_class_counts,
    load_floor_predictions,
    load_goldset_incidents,
)
from engine.schema import IncidentRecord

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_labeled_incidents(tmp: Path, records: list[dict[str, object]]) -> Path:
    p = tmp / "labeled_incidents.json"
    p.write_text(json.dumps(records))
    return p


def _write_goldset(tmp: Path, ids_classes: list[tuple[str, str]]) -> Path:
    """Write a minimal adjudicated_goldset.jsonl — only incident_id + label fields."""
    lines = []
    for inc_id, cls in ids_classes:
        lines.append(json.dumps({
            "incident_id": inc_id,
            "llm_consensus": cls,
            "adjudicated": "accept",
            "labels": [cls],
            "blind_label": cls,
            "notes": None,
        }))
    p = tmp / "goldset.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def _write_snapshot(tmp: Path, records: list[tuple[str, str]]) -> Path:
    """Write a minimal incidents.json snapshot — id + title only."""
    incidents = [
        {"id": inc_id, "title": text, "description": "", "impact": ""}
        for inc_id, text in records
    ]
    p = tmp / "incidents.json"
    p.write_text(json.dumps({"incidents": incidents}))
    return p


# ---------------------------------------------------------------------------
# load_floor_predictions
# ---------------------------------------------------------------------------

def test_load_floor_predictions_returns_incident_to_entry_id(tmp_path: Path) -> None:
    labeled = _write_labeled_incidents(tmp_path, [
        {"incident_id": "INC-001", "entry_id": "LLM01", "confidence": 0.9, "stage": 1},
        {"incident_id": "INC-002", "entry_id": "LLM02", "confidence": 0.8, "stage": 1},
        {"incident_id": "INC-003", "entry_id": "LLM01", "confidence": 0.7, "stage": 1},
    ])
    result = load_floor_predictions(labeled)
    assert result == {"INC-001": "LLM01", "INC-002": "LLM02", "INC-003": "LLM01"}


def test_load_floor_predictions_empty_file(tmp_path: Path) -> None:
    labeled = _write_labeled_incidents(tmp_path, [])
    result = load_floor_predictions(labeled)
    assert result == {}


# ---------------------------------------------------------------------------
# load_goldset_incidents  (R7 leakage firewall tests)
# ---------------------------------------------------------------------------

def test_load_goldset_incidents_keys_match_goldset_ids(tmp_path: Path) -> None:
    ids_classes = [(f"INC-{i:03d}", "LLM01") for i in range(5)]
    goldset = _write_goldset(tmp_path, ids_classes)
    snapshot_records = [(inc_id, f"text for {inc_id}") for inc_id, _ in ids_classes]
    snapshot = _write_snapshot(tmp_path, snapshot_records)

    result = load_goldset_incidents(goldset, snapshot)
    assert set(result.keys()) == {inc_id for inc_id, _ in ids_classes}


def test_load_goldset_incidents_text_hydrated_from_snapshot(tmp_path: Path) -> None:
    goldset = _write_goldset(tmp_path, [("INC-001", "A"), ("INC-002", "B")])
    snapshot = _write_snapshot(tmp_path, [
        ("INC-001", "first incident title"),
        ("INC-002", "second incident title"),
    ])
    result = load_goldset_incidents(goldset, snapshot)
    assert result["INC-001"].text == "first incident title"
    assert result["INC-002"].text == "second incident title"


def test_load_goldset_incidents_records_have_no_label_or_confidence_attr(tmp_path: Path) -> None:
    """R7: returned IncidentRecords must NOT carry adjudicated labels or confidence."""
    goldset = _write_goldset(tmp_path, [("INC-001", "LLM01"), ("INC-002", "LLM02")])
    snapshot = _write_snapshot(tmp_path, [("INC-001", "t1"), ("INC-002", "t2")])
    result = load_goldset_incidents(goldset, snapshot)
    for inc_id, record in result.items():
        assert isinstance(record, IncidentRecord)
        assert record.id == inc_id
        assert record.text  # non-empty
        # No label/confidence/consensus attributes on IncidentRecord
        assert not hasattr(record, "label")
        assert not hasattr(record, "confidence")
        assert not hasattr(record, "llm_consensus")
        assert not hasattr(record, "adjudicated")


def test_load_goldset_incidents_id_field_matches_key(tmp_path: Path) -> None:
    """R7: IncidentRecord.id == the dict key (both are the goldset incident_id)."""
    ids_classes = [("INC-010", "A"), ("INC-011", "B")]
    goldset = _write_goldset(tmp_path, ids_classes)
    snapshot = _write_snapshot(tmp_path, [("INC-010", "tx"), ("INC-011", "ty")])
    result = load_goldset_incidents(goldset, snapshot)
    for key, record in result.items():
        assert record.id == key


def test_load_goldset_incidents_bare_list_snapshot(tmp_path: Path) -> None:
    """Snapshot as a bare JSON array (no wrapper dict) must be accepted."""
    goldset = _write_goldset(tmp_path, [("INC-001", "A")])
    snapshot_path = tmp_path / "incidents.json"
    snapshot_path.write_text(json.dumps([
        {"id": "INC-001", "title": "bare list incident", "description": "", "impact": ""}
    ]))
    result = load_goldset_incidents(goldset, snapshot_path)
    assert "INC-001" in result
    assert result["INC-001"].text == "bare list incident"


def test_load_goldset_incidents_missing_from_snapshot_gives_empty_text(tmp_path: Path) -> None:
    """An incident id in the goldset but absent from snapshot gets empty text (no crash)."""
    goldset = _write_goldset(tmp_path, [("INC-MISSING", "A")])
    snapshot = _write_snapshot(tmp_path, [])  # empty snapshot
    result = load_goldset_incidents(goldset, snapshot)
    assert "INC-MISSING" in result
    assert result["INC-MISSING"].text == ""


# ---------------------------------------------------------------------------
# compute_corpus_class_counts
# ---------------------------------------------------------------------------

def test_compute_corpus_class_counts_basic(tmp_path: Path) -> None:
    labeled = _write_labeled_incidents(tmp_path, [
        {"incident_id": f"INC-{i:03d}", "entry_id": "LLM01", "stage": 1}
        for i in range(3)
    ] + [
        {"incident_id": f"INC-{i:03d}", "entry_id": "LLM02", "stage": 1}
        for i in range(3, 5)
    ])
    counts = compute_corpus_class_counts(labeled)
    assert counts == {"LLM01": 3, "LLM02": 2}


def test_compute_corpus_class_counts_empty(tmp_path: Path) -> None:
    labeled = _write_labeled_incidents(tmp_path, [])
    counts = compute_corpus_class_counts(labeled)
    assert counts == {}
