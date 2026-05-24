# tests/unit/test_calibrate_with_gold.py
"""Tests for calibrate_with_gold — merging gold labels into tally."""
from __future__ import annotations

from engine.calibrate.gold_schema import (
    GoldCalibration,
    GoldPrecisionLabel,
    GoldRecallLabel,
)
from engine.calibrate.tally import TallyResult, calibrate_with_gold


def _empty_tally() -> TallyResult:
    return TallyResult(
        precision_counts={},
        recall_counts={},
        rollup_counts={},
        total_coded=0,
        amendments_applied=0,
    )


class TestCalibrateWithGold:
    def test_recall_tp_when_classifier_matches(self) -> None:
        gold = GoldCalibration(
            recall_labels=[
                GoldRecallLabel("GA-001", ["LLM01"], "LLM01", "manual-curated"),
            ],
            precision_labels=[],
            provenance_hash="h", rubric_hash="r",
            adjudicator_id="RL", session_count=1,
        )
        result = calibrate_with_gold(_empty_tally(), gold, set(), {"LLM01", "LLM06"})

        key = ("LLM01", "security")
        assert key in result.recall_counts
        assert result.recall_counts[key].true_positives == 1
        assert result.recall_counts[key].false_negatives == 0

    def test_recall_fn_when_classifier_misses(self) -> None:
        gold = GoldCalibration(
            recall_labels=[
                GoldRecallLabel("GA-001", ["LLM01"], "LLM06", "manual-curated"),
            ],
            precision_labels=[],
            provenance_hash="h", rubric_hash="r",
            adjudicator_id="RL", session_count=1,
        )
        result = calibrate_with_gold(_empty_tally(), gold, set(), {"LLM01", "LLM06"})

        key_true = ("LLM01", "security")
        assert result.recall_counts[key_true].true_positives == 0
        assert result.recall_counts[key_true].false_negatives == 1

        key_wrong = ("LLM06", "security")
        assert key_wrong in result.precision_counts
        assert result.precision_counts[key_wrong].false_positives == 1

    def test_precision_tp(self) -> None:
        gold = GoldCalibration(
            recall_labels=[],
            precision_labels=[
                GoldPrecisionLabel("GA-002", "LLM06", True, "stage2-verified"),
            ],
            provenance_hash="h", rubric_hash="r",
            adjudicator_id="RL", session_count=1,
        )
        result = calibrate_with_gold(_empty_tally(), gold, set(), {"LLM06"})

        key = ("LLM06", "security")
        assert key in result.precision_counts
        assert result.precision_counts[key].true_positives == 1

    def test_precision_fp(self) -> None:
        gold = GoldCalibration(
            recall_labels=[],
            precision_labels=[
                GoldPrecisionLabel("GA-003", "LLM09", False, "stage2-verified"),
            ],
            provenance_hash="h", rubric_hash="r",
            adjudicator_id="RL", session_count=1,
        )
        result = calibrate_with_gold(_empty_tally(), gold, set(), {"LLM09"})

        key = ("LLM09", "security")
        assert result.precision_counts[key].false_positives == 1

    def test_recall_skips_when_classifier_entry_id_is_none(self) -> None:
        gold = GoldCalibration(
            recall_labels=[
                GoldRecallLabel("GA-001", ["LLM01"], None, "manual-curated"),
            ],
            precision_labels=[],
            provenance_hash="h", rubric_hash="r",
            adjudicator_id="RL", session_count=1,
        )
        result = calibrate_with_gold(_empty_tally(), gold, set(), {"LLM01"})

        assert ("LLM01", "security") not in result.recall_counts

    def test_deduplicates_against_base_ids(self) -> None:
        gold = GoldCalibration(
            recall_labels=[
                GoldRecallLabel("GA-001", ["LLM01"], "LLM01", "manual-curated"),
            ],
            precision_labels=[],
            provenance_hash="h", rubric_hash="r",
            adjudicator_id="RL", session_count=1,
        )
        result = calibrate_with_gold(
            _empty_tally(), gold, {"GA-001"}, {"LLM01"},
        )
        assert not result.recall_counts

    def test_merges_with_existing_tally(self) -> None:
        from engine.calibrate.tally import PrecisionTally, RecallTally

        base = TallyResult(
            precision_counts={("LLM01", "security"): PrecisionTally(5, 2, 7)},
            recall_counts={("LLM01", "security"): RecallTally(8, 2, 100)},
            rollup_counts={},
            total_coded=100,
            amendments_applied=0,
        )
        gold = GoldCalibration(
            recall_labels=[
                GoldRecallLabel("GA-100", ["LLM01"], "LLM01", "manual-curated"),
            ],
            precision_labels=[],
            provenance_hash="h", rubric_hash="r",
            adjudicator_id="RL", session_count=1,
        )
        result = calibrate_with_gold(base, gold, set(), {"LLM01"})

        assert ("LLM01", "security") in result.precision_counts
        assert ("LLM01", "security") in result.recall_counts
        assert result.recall_counts[("LLM01", "security")].true_positives == 9
        assert result.total_coded == 101

    def test_multi_label_recall(self) -> None:
        gold = GoldCalibration(
            recall_labels=[
                GoldRecallLabel("GA-001", ["LLM01", "LLM05"], "LLM01", "llm-adjudicated"),
            ],
            precision_labels=[],
            provenance_hash="h", rubric_hash="r",
            adjudicator_id="RL", session_count=1,
        )
        result = calibrate_with_gold(_empty_tally(), gold, set(), {"LLM01", "LLM05"})

        assert result.recall_counts[("LLM01", "security")].true_positives == 1
        assert result.recall_counts[("LLM05", "security")].false_negatives == 1
