"""Tests for BH + two-proportion test (Plan 8e T3)."""
from __future__ import annotations

from engine.classify.bakeoff import benjamini_hochberg, two_proportion_pvalue


def test_two_proportion_identical_is_one() -> None:
    assert two_proportion_pvalue(5, 10, 5, 10) == 1.0


def test_two_proportion_strong_difference_is_small() -> None:
    p = two_proportion_pvalue(19, 20, 2, 20)
    assert p < 0.001


def test_two_proportion_empty_cell_is_one() -> None:
    assert two_proportion_pvalue(0, 0, 1, 5) == 1.0


def test_bh_all_significant() -> None:
    # All tiny p-values -> all rejected.
    assert benjamini_hochberg([0.001, 0.002, 0.003], 0.05) == [True, True, True]


def test_bh_none_significant() -> None:
    assert benjamini_hochberg([0.9, 0.8, 0.95], 0.05) == [False, False, False]


def test_bh_step_up_known_example() -> None:
    # Classic BH: sorted p = [0.01, 0.02, 0.03, 0.04, 0.05], alpha 0.05, m=5.
    # thresholds k/m*alpha = [0.01,0.02,0.03,0.04,0.05]; all <= -> all rejected.
    mask = benjamini_hochberg([0.05, 0.04, 0.03, 0.02, 0.01], 0.05)
    assert mask == [True, True, True, True, True]


def test_bh_partial_rejection_preserves_input_order() -> None:
    # p=[0.001, 0.5, 0.04], m=3, alpha=0.05. sorted=[0.001,0.04,0.5];
    # thresholds=[0.0167,0.0333,0.05]; 0.001<=0.0167 yes, 0.04<=0.0333 no,
    # 0.5<=0.05 no -> largest k with pass is k=1 -> reject sorted ranks<=1
    # -> only p=0.001 rejected. Mask aligned to input order.
    assert benjamini_hochberg([0.001, 0.5, 0.04], 0.05) == [True, False, False]
