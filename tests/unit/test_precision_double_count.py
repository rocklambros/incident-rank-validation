"""F-A: adjudicated misclassifications must count ONE precision FP, not two."""
from __future__ import annotations

from engine.calibrate.gold_schema import (
    GoldCalibration,
    GoldPrecisionLabel,
    GoldRecallLabel,
)
from engine.calibrate.tally import TallyResult, calibrate_with_gold


def _base() -> TallyResult:
    return TallyResult(
        precision_counts={}, recall_counts={}, rollup_counts={},
        total_coded=0, amendments_applied=0,
    )


def test_adjudicated_misclassification_is_one_precision_fp() -> None:
    # Adjudicated path: the SAME incident produces a recall label (classifier=B,
    # truth=A) AND a loader precision_label (claimed=B, is_correct=False).
    # Before the fix both fire -> FP=2; after the fix -> FP=1.
    gold = GoldCalibration(
        recall_labels=[GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A"],
            classifier_entry_id="B", source="llm-adjudicated",
        )],
        precision_labels=[GoldPrecisionLabel(
            incident_id="i0", claimed_entry_id="B",
            is_correct=False, source="llm-adjudicated",
        )],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )
    merged = calibrate_with_gold(_base(), gold, set(), {"A", "B"})
    p_b = merged.precision_counts[("B", "security")]
    assert p_b.false_positives == 1  # not 2
    assert p_b.total == 1


def test_recall_only_path_still_derives_precision_fp() -> None:
    # Curation / F4 path: recall label, NO precision_labels -> recall-derived FP
    # MUST still fire (the dedup set is empty).
    gold = GoldCalibration(
        recall_labels=[GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A"],
            classifier_entry_id="B", source="manual",
        )],
        precision_labels=[],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )
    merged = calibrate_with_gold(_base(), gold, set(), {"A", "B"})
    assert merged.precision_counts[("B", "security")].false_positives == 1


def test_correct_adjudicated_claim_is_one_precision_tp() -> None:
    # Correct claim: precision_label is_correct=True -> 1 TP; recall-derived does
    # not fire (B in truth). No change from the fix, asserted for completeness.
    gold = GoldCalibration(
        recall_labels=[GoldRecallLabel(
            incident_id="i0", true_entry_ids=["B"],
            classifier_entry_id="B", source="llm-adjudicated",
        )],
        precision_labels=[GoldPrecisionLabel(
            incident_id="i0", claimed_entry_id="B",
            is_correct=True, source="llm-adjudicated",
        )],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )
    merged = calibrate_with_gold(_base(), gold, set(), {"A", "B"})
    p_b = merged.precision_counts[("B", "security")]
    assert p_b.true_positives == 1 and p_b.false_positives == 0 and p_b.total == 1
