"""Tests for the oracle incidence re-derivation (Plan 8d D1)."""
from __future__ import annotations

import numpy as np

from engine.verify.oracle import oracle_incidence_intervals, oracle_incidence_ranking


def test_incidence_ranking_orders_by_lambda_times_size() -> None:
    # Two entries, single stratum size 10. A has higher lambda -> A ranks first.
    lambda_samples = np.array([[0.5, 0.1], [0.6, 0.2], [0.4, 0.15]])
    entry_ids = ("A", "B")
    entry_strata = {"A": ("security",), "B": ("security",)}
    stratum_sizes = {"security": 10}
    ranking = oracle_incidence_ranking(
        lambda_samples, entry_ids, entry_strata, stratum_sizes
    )
    assert ranking == ("A", "B")


def test_incidence_ranking_uses_multi_stratum_sum() -> None:
    # B has lower lambda but spans two strata; its total exposure beats A.
    lambda_samples = np.array([[0.30, 0.20], [0.30, 0.20]])
    entry_ids = ("A", "B")
    entry_strata = {"A": ("security",), "B": ("security", "ai-harm")}
    stratum_sizes = {"security": 100, "ai-harm": 100}
    # A: 0.30*100 = 30 ; B: 0.20*200 = 40 -> B first
    ranking = oracle_incidence_ranking(
        lambda_samples, entry_ids, entry_strata, stratum_sizes
    )
    assert ranking == ("B", "A")


def test_incidence_intervals_are_ordered_pairs() -> None:
    lambda_samples = np.array([[0.5, 0.1], [0.6, 0.2], [0.4, 0.15], [0.55, 0.12]])
    entry_ids = ("A", "B")
    entry_strata = {"A": ("security",), "B": ("security",)}
    stratum_sizes = {"security": 10}
    ci = oracle_incidence_intervals(
        lambda_samples, entry_ids, entry_strata, stratum_sizes
    )
    assert set(ci.keys()) == {"A", "B"}
    for lo, hi in ci.values():
        assert lo <= hi


def test_incidence_ranking_tiebreak_by_entry_id() -> None:
    # Identical incidence -> deterministic order by entry id.
    lambda_samples = np.array([[0.2, 0.2], [0.2, 0.2]])
    entry_ids = ("B", "A")
    entry_strata = {"A": ("s",), "B": ("s",)}
    stratum_sizes = {"s": 5}
    ranking = oracle_incidence_ranking(
        lambda_samples, entry_ids, entry_strata, stratum_sizes
    )
    assert ranking == ("A", "B")
