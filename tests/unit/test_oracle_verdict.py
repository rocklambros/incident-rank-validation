"""Tests for oracle tolerances, comparisons, and verdict (Plan 8d Task 4)."""
from __future__ import annotations

import pytest

from engine.verify.oracle import (
    ORACLE_SIGMA_U_BAND,
    ORACLE_TAU_INCIDENCE,
    ORACLE_TAU_PL,
    OracleVerdict,
    compare_ranking,
    compare_sigma_u,
    kendall_tau,
    top_tier_set,
)


def test_tolerance_constants() -> None:
    assert ORACLE_TAU_INCIDENCE == 0.95
    assert ORACLE_TAU_PL == 0.70
    assert ORACLE_SIGMA_U_BAND == 0.75


def test_kendall_tau_identical_is_one() -> None:
    assert kendall_tau(("A", "B", "C"), ("A", "B", "C")) == 1.0


def test_kendall_tau_reversed_is_negative_one() -> None:
    assert kendall_tau(("A", "B", "C"), ("C", "B", "A")) == -1.0


def test_kendall_tau_mismatched_sets_raises() -> None:
    with pytest.raises(ValueError):
        kendall_tau(("A", "B"), ("A", "C"))


def test_top_tier_set() -> None:
    assert top_tier_set(("A", "B", "C", "D", "E", "F")) == {"A", "B"}
    assert top_tier_set(("A", "B")) == {"A"}


def test_compare_ranking_pass_when_identical() -> None:
    d = compare_ranking(
        "incidence", ("A", "B", "C", "D"), ("A", "B", "C", "D"), ORACLE_TAU_INCIDENCE
    )
    assert d.status == "PASS"


def test_compare_ranking_fail_when_reversed() -> None:
    d = compare_ranking(
        "incidence", ("A", "B", "C", "D"), ("D", "C", "B", "A"), ORACLE_TAU_INCIDENCE
    )
    assert d.status == "FAIL"


def test_compare_sigma_u_pass_and_fail() -> None:
    assert compare_sigma_u(1.0, 1.5, ORACLE_SIGMA_U_BAND).status == "PASS"
    assert compare_sigma_u(1.0, 2.0, ORACLE_SIGMA_U_BAND).status == "FAIL"


def test_verdict_provisional_iff_any_fail() -> None:
    ok = compare_sigma_u(1.0, 1.0, ORACLE_SIGMA_U_BAND)
    bad = compare_sigma_u(1.0, 9.0, ORACLE_SIGMA_U_BAND)
    assert OracleVerdict(deliverables=(ok,)).provisional is False
    assert OracleVerdict(deliverables=(ok, bad)).provisional is True
