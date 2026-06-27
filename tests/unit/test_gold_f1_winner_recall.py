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


def test_missing_classifier_label_becomes_oos_recall_miss(tmp_path: Path) -> None:
    from engine.calibrate.gold_schema import OUT_OF_SCOPE

    gold_dir = _write_adjudicated(tmp_path)
    # INC-2 has no classifier label -> OOS sentinel (a recall miss for its truth).
    gold = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t",
        classifier_labels={"INC-1": "A"},  # INC-2 deliberately absent
    )
    by_id = {r.incident_id: r for r in gold.recall_labels}
    assert by_id["INC-2"].classifier_entry_id == OUT_OF_SCOPE
    # No precision row is generated for an OOS prediction.
    assert all(p.claimed_entry_id != OUT_OF_SCOPE for p in gold.precision_labels)


def test_vocab_guard_raises_on_unknown_classifier_label(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    with pytest.raises(ValueError, match="not in rubric"):
        load_gold_calibration(
            gold_dir=gold_dir, valid_entry_ids={"A", "B"},
            rubric_hash="r", adjudicator_id="t",
            classifier_labels={"INC-1": "A", "INC-2": "B-TYPO"},
        )


def test_winner_recall_differs_from_consensus_end_to_end(tmp_path: Path) -> None:
    from engine.calibrate.tally import TallyResult, calibrate_with_gold

    gold_dir = _write_adjudicated(tmp_path)
    base = TallyResult(
        precision_counts={}, recall_counts={}, rollup_counts={},
        total_coded=0, amendments_applied=0,
    )

    # Consensus channel: INC-1 consensus "A" == truth "A" -> recall HIT for A.
    gold_consensus = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t",
    )
    tally_c = calibrate_with_gold(base, gold_consensus, set(), {"A", "B"})
    rc_a = tally_c.recall_counts[("A", "security")]
    assert rc_a.true_positives == 1 and rc_a.false_negatives == 0

    # Winner channel: INC-1 winner "B" != truth "A" -> recall MISS for A.
    winner = {"INC-1": "B", "INC-2": "B"}
    gold_winner = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t", classifier_labels=winner,
    )
    tally_w = calibrate_with_gold(base, gold_winner, set(), {"A", "B"})
    rw_a = tally_w.recall_counts[("A", "security")]
    assert rw_a.true_positives == 0 and rw_a.false_negatives == 1

    # The two channels genuinely disagree -> F1 wiring changes the recall.
    assert rc_a != rw_a


def test_oos_prediction_is_recall_miss_no_precision_fp() -> None:
    from engine.calibrate.gold_schema import (
        OUT_OF_SCOPE,
        GoldCalibration,
        GoldRecallLabel,
    )
    from engine.calibrate.tally import TallyResult, calibrate_with_gold

    base = TallyResult(
        precision_counts={}, recall_counts={}, rollup_counts={},
        total_coded=0, amendments_applied=0,
    )
    gold = GoldCalibration(
        recall_labels=[GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A"],
            classifier_entry_id=OUT_OF_SCOPE, source="g",
        )],
        precision_labels=[],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )
    merged = calibrate_with_gold(base, gold, set(), {"A"})
    # Recall: a miss for A (it was truly A, classifier said out-of-scope).
    rc = merged.recall_counts[("A", "security")]
    assert rc.true_positives == 0 and rc.false_negatives == 1
    # Precision: NO false-positive cell for the OOS sentinel.
    assert (OUT_OF_SCOPE, "security") not in merged.precision_counts
