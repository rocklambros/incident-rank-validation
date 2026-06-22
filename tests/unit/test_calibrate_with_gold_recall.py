from engine.calibrate.beta import BetaPosterior
from engine.calibrate.calibrate import compute_calibration
from engine.calibrate.gold_schema import GoldCalibration, GoldRecallLabel
from engine.calibrate.tally import RecallTally, TallyResult, calibrate_with_gold


def _base_tally_with_frame_padding() -> TallyResult:
    # Simulates the frame-size-padded recall the recall branch produces today:
    # ROLL-CFAS has exactly 1 true incident but a denominator of 100.
    return TallyResult(
        precision_counts={},
        recall_counts={("ROLL-CFAS", "ai-harm"): RecallTally(0, 100, 100)},
        rollup_counts={},
        total_coded=100,
        amendments_applied=0,
    )


def test_recall_derives_solely_from_gold_not_frame_padding():
    base = _base_tally_with_frame_padding()
    gold = GoldCalibration(
        recall_labels=[
            GoldRecallLabel(
                incident_id="INC-1",
                true_entry_ids=["ROLL-CFAS"],
                classifier_entry_id="out-of-scope",  # classifier MISSED it -> FN
                source="goldset",
            )
        ],
        precision_labels=[],
        provenance_hash="h",
        rubric_hash="r",
        adjudicator_id="RL",
        session_count=1,
    )
    merged = calibrate_with_gold(
        base_tally=base,
        gold=gold,
        base_incident_ids=set(),
        all_entry_ids={"ROLL-CFAS"},
        merge_stratum="ai-harm",
    )
    # Recall denominator must be the truth cell (1), NOT the frame size (100).
    rt = merged.recall_counts[("ROLL-CFAS", "ai-harm")]
    assert rt == RecallTally(true_positives=0, false_negatives=1, total_in_sample=1)

    cal, _ = compute_calibration(
        merged, all_entry_ids=["ROLL-CFAS"], frame_blind_ids=set(),
    )
    # Wide posterior: Beta(1, 2), mean 1/3 -- NOT the falsely-precise Beta(1, 101).
    assert cal.recall[("ROLL-CFAS", "ai-harm")] == BetaPosterior(alpha=1.0, beta=2.0)
