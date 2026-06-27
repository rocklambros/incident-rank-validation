"""F1: recall calibration scores the classifier's labels, not the old consensus."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.calibrate.gold_loader import (
    load_classifier_labels,
    load_gold_calibration,
)


def _write_adjudicated(tmp: Path) -> Path:
    gold_dir = tmp / "calibration"
    gold_dir.mkdir(parents=True)
    rows = [
        {"incident_id": "INC-1", "llm_consensus": "A", "adjudicated": "accept",
         "labels": ["A"], "blind_label": "A", "notes": None},
        {"incident_id": "INC-2", "llm_consensus": "A", "adjudicated": "override",
         "labels": ["B"], "blind_label": "B", "notes": None},
    ]
    (gold_dir / "adjudicated_goldset.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return gold_dir


def test_load_classifier_labels_maps_incident_to_entry(tmp_path: Path) -> None:
    p = tmp_path / "labeled_incidents.json"
    p.write_text(json.dumps([
        {"incident_id": "INC-1", "entry_id": "A", "stage": 2},
        {"incident_id": "INC-2", "entry_id": "B", "stage": 1},
    ]))
    assert load_classifier_labels(p) == {"INC-1": "A", "INC-2": "B"}


def test_classifier_labels_drive_classifier_entry_id(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    # Winner DISAGREES with consensus on INC-1 (consensus A, winner B).
    winner = {"INC-1": "B", "INC-2": "B"}
    gold = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t", classifier_labels=winner,
    )
    by_id = {r.incident_id: r for r in gold.recall_labels}
    assert by_id["INC-1"].classifier_entry_id == "B"  # winner, not consensus "A"
    assert by_id["INC-2"].classifier_entry_id == "B"


def test_default_is_backward_compatible_consensus(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    gold = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t",  # no classifier_labels
    )
    by_id = {r.incident_id: r for r in gold.recall_labels}
    assert by_id["INC-1"].classifier_entry_id == "A"  # consensus preserved
    assert by_id["INC-2"].classifier_entry_id == "A"


def test_coverage_guard_raises_on_missing_incident(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    with pytest.raises(ValueError, match="coverage guard"):
        load_gold_calibration(
            gold_dir=gold_dir, valid_entry_ids={"A", "B"},
            rubric_hash="r", adjudicator_id="t",
            classifier_labels={"INC-1": "A"},  # INC-2 missing
        )


def test_vocab_guard_raises_on_unknown_classifier_label(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    with pytest.raises(ValueError, match="not in rubric"):
        load_gold_calibration(
            gold_dir=gold_dir, valid_entry_ids={"A", "B"},
            rubric_hash="r", adjudicator_id="t",
            classifier_labels={"INC-1": "A", "INC-2": "B-TYPO"},
        )
