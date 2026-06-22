# tests/unit/test_confusion_overlap.py
from engine.calibrate.confusion import build_overlap_from_confusion
from engine.calibrate.gold_schema import GoldCalibration, GoldRecallLabel


def _gold(*triples) -> GoldCalibration:
    return GoldCalibration(
        recall_labels=[
            GoldRecallLabel(incident_id=f"i{n}", true_entry_ids=[t],
                            classifier_entry_id=p, source="g")
            for n, (t, p) in enumerate(triples)
        ],
        precision_labels=[], provenance_hash="h", rubric_hash="r",
        adjudicator_id="RL", session_count=1,
    )


def test_overlap_built_from_misclassifications():
    # Two incidents truly LLM02 but the classifier predicted LLM01 -> LLM01's FPs
    # leak from LLM02. One correct LLM01. So source=LLM01 has 2 FPs, all from LLM02.
    gold = _gold(("LLM02", "LLM01"), ("LLM02", "LLM01"), ("LLM01", "LLM01"))
    W = build_overlap_from_confusion(gold, ("LLM01", "LLM02"))
    # W[target=LLM02][source=LLM01] = 2 FPs from LLM02 / 2 total FPs of LLM01 = 1.0
    assert abs(W.weights["LLM02"]["LLM01"] - 1.0) < 1e-9
    # no self-loop
    assert "LLM01" not in W.weights.get("LLM01", {})
