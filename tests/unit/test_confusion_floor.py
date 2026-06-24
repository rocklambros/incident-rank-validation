from engine.calibrate.confusion import build_overlap_from_confusion
from engine.calibrate.gold_schema import GoldCalibration, GoldRecallLabel


def _gold(*triples: tuple[str, str]) -> GoldCalibration:
    return GoldCalibration(
        recall_labels=[
            GoldRecallLabel(incident_id=f"i{n}", true_entry_ids=[t],
                            classifier_entry_id=p, source="g")
            for n, (t, p) in enumerate(triples)
        ],
        precision_labels=[], provenance_hash="h", rubric_hash="r",
        adjudicator_id="t", session_count=1,
    )


def test_floor_drops_single_fp_column() -> None:
    # LLM01 has exactly 1 FP (truly LLM02) -> with floor 2, no leakage column.
    gold = _gold(("LLM02", "LLM01"))
    W = build_overlap_from_confusion(gold, ("LLM01", "LLM02"), min_fp_count=2)
    assert "LLM02" not in W.weights or "LLM01" not in W.weights.get("LLM02", {})


def test_floor_keeps_sufficient_column() -> None:
    # LLM01 has 2 FPs (both truly LLM02) -> meets floor 2, W=1.0 retained.
    gold = _gold(("LLM02", "LLM01"), ("LLM02", "LLM01"))
    W = build_overlap_from_confusion(gold, ("LLM01", "LLM02"), min_fp_count=2)
    assert abs(W.weights["LLM02"]["LLM01"] - 1.0) < 1e-9


def test_default_floor_is_backcompat() -> None:
    gold = _gold(("LLM02", "LLM01"))
    W = build_overlap_from_confusion(gold, ("LLM01", "LLM02"))  # default min_fp_count=1
    assert abs(W.weights["LLM02"]["LLM01"] - 1.0) < 1e-9
