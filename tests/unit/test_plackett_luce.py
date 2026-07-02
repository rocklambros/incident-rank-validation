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


from engine.vote.plackett_luce import (  # noqa: E402  (appended import block)
    N_BOOTSTRAP_DEFAULT,
    DavidsonPosterior,
    _ranking_to_rank_vector,
    bootstrap_davidson,
)


def test_ranking_to_rank_vector() -> None:
    # ranking best->worst is (B, A, C); rank vector is aligned to entry_ids order.
    vec = _ranking_to_rank_vector(("B", "A", "C"), ("A", "B", "C"))
    # A is 2nd, B is 1st, C is 3rd
    assert list(vec) == [2.0, 1.0, 3.0]


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (8, 1))
    p1 = bootstrap_davidson(rankings, ("A", "B", "C"), n_bootstrap=30, seed=42)
    p2 = bootstrap_davidson(rankings, ("A", "B", "C"), n_bootstrap=30, seed=42)
    assert p1.median_ranks == p2.median_ranks
    assert p1.top5_frequency == p2.top5_frequency
    assert p1.mean_kendall_tau_vs_point == p2.mean_kendall_tau_vs_point


def test_bootstrap_strong_signal_is_stable() -> None:
    # Everyone strictly ranks A>B>C: A is top in every resample.
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (12, 1))
    post = bootstrap_davidson(rankings, ("A", "B", "C"), n_bootstrap=50, seed=7)
    assert post.point_ranking == ("A", "B", "C")
    assert post.top5_frequency["A"] == 1.0
    assert post.median_ranks["A"] == 1.0
    # Kendall tau vs the point ranking is in [-1, 1] and high for clean data.
    assert -1.0 <= post.mean_kendall_tau_vs_point <= 1.0
    assert post.mean_kendall_tau_vs_point > 0.8


def test_bootstrap_reports_counts() -> None:
    rankings = np.tile(np.array([1.0, 2.0]), (6, 1))
    post = bootstrap_davidson(rankings, ("A", "B"), n_bootstrap=20, seed=1)
    assert isinstance(post, DavidsonPosterior)
    assert post.n_respondents == 6
    assert post.n_bootstrap == 20


def test_default_bootstrap_count() -> None:
    assert N_BOOTSTRAP_DEFAULT == 2000


def test_bootstrap_reports_nonconverged_count() -> None:
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (8, 1))
    post = bootstrap_davidson(rankings, ("A", "B", "C"), n_bootstrap=15, seed=3)
    assert isinstance(post.n_nonconverged, int)
    assert 0 <= post.n_nonconverged <= 15
