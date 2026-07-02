"""Pin the INTENTIONAL single-label recall semantics (Plan 8ab-remediation F4).

The classifier is single-label (one `classifier_entry_id` per incident); truth may
be multi-label. Recall is the detection rate the measurement-error model consumes:
an incident truly {A, B} labeled A counts as a recall TP for A and a recall FN for B
(it is physically in A's observed count, not B's). This is correct, not a bug — these
tests lock it so a future "fix" cannot silently inflate co-occurring-entry recall.
"""
from __future__ import annotations

from engine.calibrate.gold_schema import GoldCalibration, GoldRecallLabel
from engine.calibrate.tally import PrecisionTally, RecallTally, TallyResult, calibrate_with_gold


def _empty_base() -> TallyResult:
    return TallyResult(
        precision_counts={}, recall_counts={}, rollup_counts={},
        total_coded=0, amendments_applied=0,
    )


def _gold(*labels: GoldRecallLabel) -> GoldCalibration:
    return GoldCalibration(
        recall_labels=list(labels), precision_labels=[],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )


def test_multilabel_incident_is_tp_for_chosen_and_fn_for_other() -> None:
    # Incident truly {A, B}; classifier labels A. A -> recall TP; B -> recall FN.
    gold = _gold(
        GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A", "B"],
            classifier_entry_id="A", source="g",
        )
    )
    merged = calibrate_with_gold(
        base_tally=_empty_base(), gold=gold, base_incident_ids=set(),
        all_entry_ids={"A", "B"}, merge_stratum="security",
    )
    # A: caught (TP), denominator 1.
    assert merged.recall_counts[("A", "security")] == RecallTally(
        true_positives=1, false_negatives=0, total_in_sample=1,
    )
    # B: truly-B but labeled A -> a real detection MISS (FN), denominator 1.
    # This is the intentional single-label semantics, NOT a per-label-hit credit.
    assert merged.recall_counts[("B", "security")] == RecallTally(
        true_positives=0, false_negatives=1, total_in_sample=1,
    )


def test_singlelabel_incident_is_a_plain_tp() -> None:
    gold = _gold(
        GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A"],
            classifier_entry_id="A", source="g",
        )
    )
    merged = calibrate_with_gold(
        base_tally=_empty_base(), gold=gold, base_incident_ids=set(),
        all_entry_ids={"A"}, merge_stratum="security",
    )
    assert merged.recall_counts[("A", "security")] == RecallTally(
        true_positives=1, false_negatives=0, total_in_sample=1,
    )


def test_wrong_label_is_fn_for_truth_and_precision_fp_for_claim() -> None:
    # Incident truly {A}; classifier labels B. A -> recall FN; B -> precision FP.
    gold = _gold(
        GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A"],
            classifier_entry_id="B", source="g",
        )
    )
    merged = calibrate_with_gold(
        base_tally=_empty_base(), gold=gold, base_incident_ids=set(),
        all_entry_ids={"A", "B"}, merge_stratum="security",
    )
    assert merged.recall_counts[("A", "security")] == RecallTally(
        true_positives=0, false_negatives=1, total_in_sample=1,
    )
    # The mislabel is a precision false positive for the claimed entry B.
    assert merged.precision_counts[("B", "security")] == PrecisionTally(
        true_positives=0, false_positives=1, total=1,
    )
