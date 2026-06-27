"""Tests for floor + winner selection (Plan 8e T4)."""
from __future__ import annotations

from engine.classify.bakeoff import BAKEOFF_ALPHA, BakeoffResult, select_winner


def _truth() -> dict[str, frozenset[str]]:
    # 12 A, 12 B (both >= min_cell 5), all in lockbox for simplicity.
    t: dict[str, frozenset[str]] = {}
    for i in range(12):
        t[f"a{i}"] = frozenset({"A"})
    for i in range(12):
        t[f"b{i}"] = frozenset({"B"})
    return t


def test_winner_beats_floor() -> None:
    truth = _truth()
    lock = frozenset(truth)
    # Floor: gets A right, B wrong (predicts A for everything).
    floor = {k: "A" for k in truth}
    # Config "good": perfect.
    good = {k: ("A" if k.startswith("a") else "B") for k in truth}
    # Config "same": same as floor.
    same = dict(floor)
    result = select_winner(
        {"good": good, "same": same}, floor, truth, lock, alpha=BAKEOFF_ALPHA
    )
    assert isinstance(result, BakeoffResult)
    assert result.winner == "good"
    assert result.config_balanced_accuracy["good"] == 1.0
    assert result.floor_balanced_accuracy == 0.5
    assert "good" in result.eligible_configs
    assert "same" not in result.eligible_configs


def test_no_winner_when_none_beats_floor() -> None:
    truth = _truth()
    lock = frozenset(truth)
    floor = {k: ("A" if k.startswith("a") else "B") for k in truth}  # perfect floor
    weak = {k: "A" for k in truth}  # worse
    result = select_winner({"weak": weak}, floor, truth, lock)
    assert result.winner is None
    assert result.eligible_configs == ()


def test_sparse_class_excluded_from_metric() -> None:
    # Class C has only 2 incidents -> sparse -> excluded from selection metric.
    truth: dict[str, frozenset[str]] = {}
    for i in range(12):
        truth[f"a{i}"] = frozenset({"A"})
    for i in range(12):
        truth[f"b{i}"] = frozenset({"B"})
    truth["c0"] = frozenset({"C"})
    truth["c1"] = frozenset({"C"})
    lock = frozenset(truth)
    floor = {k: "A" for k in truth}
    good = {k: ("A" if k.startswith("a") else ("B" if k.startswith("b") else "C")) for k in truth}
    result = select_winner({"good": good}, floor, truth, lock)
    assert "C" in result.sparse_classes
    assert "C" not in result.selection_classes


def test_zero_lockbox_cell_class_excluded_from_metric() -> None:
    # C is non-sparse on the full goldset (6 >= 5) but has ZERO lockbox truth
    # -> it cannot be measured on the lockbox and must be excluded from the
    # selection metric (not scored as 0/0 = 0.0), and it is NOT "sparse".
    truth: dict[str, frozenset[str]] = {}
    for i in range(8):
        truth[f"a{i}"] = frozenset({"A"})
    for i in range(8):
        truth[f"b{i}"] = frozenset({"B"})
    for i in range(6):
        truth[f"c{i}"] = frozenset({"C"})
    # Lockbox excludes every C incident.
    lock = frozenset([f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)])
    floor = {k: "A" for k in truth}
    good = {
        k: ("A" if k.startswith("a") else ("B" if k.startswith("b") else "C"))
        for k in truth
    }
    result = select_winner({"good": good}, floor, truth, lock)
    assert "C" not in result.selection_classes
    assert "C" not in result.sparse_classes


def test_select_winner_raises_on_unknown_prediction_class() -> None:
    truth: dict[str, frozenset[str]] = {f"a{i}": frozenset({"A"}) for i in range(8)}
    truth.update({f"b{i}": frozenset({"B"}) for i in range(8)})
    lock = frozenset(truth)
    floor = {k: "A" for k in truth}
    bad = {k: ("A" if k.startswith("a") else "B-TYPO") for k in truth}
    import pytest

    with pytest.raises(ValueError, match="goldset vocabulary"):
        select_winner({"bad": bad}, floor, truth, lock)
