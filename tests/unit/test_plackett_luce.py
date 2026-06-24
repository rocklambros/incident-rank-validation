"""Tests for the Davidson tie-aware paired-comparison vote model (Plan 8c)."""
from __future__ import annotations

import numpy as np

from engine.vote.plackett_luce import (
    DEFAULT_RIDGE,
    DavidsonFit,
    _pairwise_counts,
    fit_davidson,
)


def test_pairwise_counts_wins_and_ties() -> None:
    # 3 respondents, 3 entries (A, B, C).
    # r0: A>B>C (ranks 1,2,3); r1: A>B>C; r2: A tied B, both above C (ranks 1.5,1.5,3)
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.5, 1.5, 3.0],
        ]
    )
    wins, ties = _pairwise_counts(rankings)
    # A preferred over B by r0, r1 (r2 tied) -> wins[0,1] == 2
    assert wins[0, 1] == 2.0
    assert wins[1, 0] == 0.0
    # A and B both preferred over C by all 3 -> wins[0,2] == 3, wins[1,2] == 3
    assert wins[0, 2] == 3.0
    assert wins[1, 2] == 3.0
    # A tied B once (r2) -> ties[0,1] == 1 (upper triangle)
    assert ties[0, 1] == 1.0
    # no C ties
    assert ties[0, 2] == 0.0


def test_fit_recovers_strict_order() -> None:
    # Everyone ranks A>B>C strictly -> worth(A) > worth(B) > worth(C).
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    fit = fit_davidson(rankings, ("A", "B", "C"))
    assert fit.ranking == ("A", "B", "C")
    assert fit.log_worths["A"] > fit.log_worths["B"] > fit.log_worths["C"]
    assert fit.converged is True


def test_fit_tie_parameter_positive_when_ties_present() -> None:
    # Mix of strict and tied ballots -> tie parameter is finite and > 0.
    rankings = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.5, 1.5],
            [1.5, 1.5],
        ]
    )
    fit = fit_davidson(rankings, ("A", "B"))
    assert np.isfinite(fit.tie_param)
    assert fit.tie_param > 0.0


def test_fit_handles_complete_separation_without_diverging() -> None:
    # A is ranked strictly above everything by all respondents (separation):
    # worth(A) stays finite because of the ridge penalty.
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
            [1.0, 2.0, 3.0],
        ]
    )
    fit = fit_davidson(rankings, ("A", "B", "C"))
    assert np.isfinite(fit.log_worths["A"])
    assert fit.worths["A"] > fit.worths["B"]
    assert fit.ranking[0] == "A"


def test_fit_drop_ties_reduces_to_bradley_terry() -> None:
    # include_ties=False ignores tied pairs; strict order still recovered,
    # tie_param reported as 0.0.
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.5, 1.5, 3.0],
        ]
    )
    fit = fit_davidson(rankings, ("A", "B", "C"), include_ties=False)
    assert fit.ranking == ("A", "B", "C")
    assert fit.tie_param == 0.0


def test_default_ridge_value() -> None:
    assert DEFAULT_RIDGE == 1e-3


def test_fit_returns_dataclass_with_aligned_entries() -> None:
    rankings = np.tile(np.array([1.0, 2.0]), (3, 1))
    fit = fit_davidson(rankings, ("X", "Y"))
    assert isinstance(fit, DavidsonFit)
    assert fit.entries == ("X", "Y")
    assert set(fit.worths.keys()) == {"X", "Y"}
    assert set(fit.ranking) == {"X", "Y"}
