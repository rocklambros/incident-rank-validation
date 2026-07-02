"""Tests for the oracle Bradley-Terry MM PL re-derivation (Plan 8d D2)."""
from __future__ import annotations

import numpy as np

from engine.verify.oracle import _pairwise_wins_halfcredit, oracle_pl_ranking_mm


def test_pairwise_wins_halfcredit() -> None:
    # 3 respondents, entries A,B,C. r0,r1: A>B>C ; r2: A=B>C
    rankings = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.5, 1.5, 3.0]])
    wins, comparisons = _pairwise_wins_halfcredit(rankings)
    # A: beats B twice (r0,r1) + 0.5 tie (r2) = 2.5 ; beats C 3 times = 3 -> 5.5
    assert wins[0] == 5.5
    # B: 0.5 tie with A (r2) + beats C 3 times = 3.5
    assert wins[1] == 3.5
    # C: never wins -> 0
    assert wins[2] == 0.0
    # every pair compared by all 3 respondents
    assert comparisons[0, 1] == 3
    assert comparisons[0, 2] == 3


def test_mm_recovers_strict_order() -> None:
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (6, 1))
    ranking = oracle_pl_ranking_mm(rankings, ("A", "B", "C"))
    assert ranking == ("A", "B", "C")


def test_mm_agrees_with_known_dominance() -> None:
    # A dominates; B and C close. A must be first.
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
        ]
    )
    ranking = oracle_pl_ranking_mm(rankings, ("A", "B", "C"))
    assert ranking[0] == "A"


def test_mm_is_deterministic() -> None:
    rankings = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0]])
    r1 = oracle_pl_ranking_mm(rankings, ("A", "B", "C"))
    r2 = oracle_pl_ranking_mm(rankings, ("A", "B", "C"))
    assert r1 == r2
