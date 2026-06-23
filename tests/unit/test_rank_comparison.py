# tests/unit/test_rank_comparison.py
"""Tests for per-entry rank comparison in concordance."""
from __future__ import annotations

import numpy as np

from engine.decide.concordance import _tier_agreement_label, compute_concordance
from engine.model.inference import InferenceResult
from engine.vote.bootstrap import VoteRankPosterior


def _make_entry_strata(
    entries: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """Build minimal single-stratum entry_strata and stratum_sizes for tests."""
    entry_strata: dict[str, tuple[str, ...]] = {e: ("default",) for e in entries}
    stratum_sizes = {"default": max(len(entries) * 10, 1)}
    return entry_strata, stratum_sizes


def _make_inference(
    entry_ids: tuple[str, ...],
    n_samples: int = 100,
    seed: int = 42,
) -> InferenceResult:
    rng = np.random.default_rng(seed)
    n = len(entry_ids)
    lam = rng.exponential(scale=1.0, size=(n_samples, n))
    return InferenceResult(
        lambda_samples=lam,
        entry_ids=entry_ids,
        r_hat={f"lambda[{i}]": 1.0 for i in range(n)},
        ess={f"lambda[{i}]": float(n_samples) for i in range(n)},
        divergences=0,
        num_warmup=100,
        num_samples=n_samples,
    )


def _make_vote_posterior(
    entries: tuple[str, ...],
    n_bootstrap: int = 100,
    seed: int = 99,
) -> VoteRankPosterior:
    rng = np.random.default_rng(seed)
    n = len(entries)
    rank_samples = np.zeros((n_bootstrap, n), dtype=np.float64)
    for b in range(n_bootstrap):
        order = rng.permutation(n)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        rank_samples[b] = ranks
    medians = {entries[i]: float(np.median(rank_samples[:, i])) for i in range(n)}
    return VoteRankPosterior(
        entries=entries,
        rank_samples=rank_samples,
        median_ranks=medians,
        n_respondents=50,
        n_bootstrap=n_bootstrap,
    )


def test_concordance_includes_per_entry_comparison() -> None:
    entries = tuple(f"E{i}" for i in range(10))
    inf = _make_inference(entries, n_samples=200)
    vote = _make_vote_posterior(entries, n_bootstrap=200)
    entry_strata, stratum_sizes = _make_entry_strata(entries)

    result = compute_concordance(
        inference_result=inf,
        vote_posterior=vote,
        tier_boundaries=(3, 7),
        flag_threshold_tau=0.3,
        measurable_count=10,
        total_count=10,
        meaningful_kappa_n=5,
        measurability_minimum=5,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )
    assert result.entry_comparisons is not None
    assert len(result.entry_comparisons) > 0
    first = result.entry_comparisons[0]
    assert "entry_id" in first
    assert "lambda_rank_median" in first
    assert "vote_rank_median" in first
    assert "tier_agreement" in first
    assert "direction" in first
    assert "action" in first


def test_tier_agreement_labels() -> None:
    assert _tier_agreement_label(0, 0) == "same"
    assert _tier_agreement_label(0, 1) == "±1"
    assert _tier_agreement_label(0, 2) == "±2+"
    assert _tier_agreement_label(2, 0) == "±2+"


def test_na_result_has_none_comparisons() -> None:
    entries = ("A", "B")
    inf = _make_inference(entries)
    vote = _make_vote_posterior(entries)
    entry_strata, stratum_sizes = _make_entry_strata(entries)

    result = compute_concordance(
        inference_result=inf,
        vote_posterior=vote,
        tier_boundaries=(1,),
        flag_threshold_tau=0.5,
        measurable_count=2,
        total_count=10,
        meaningful_kappa_n=5,
        measurability_minimum=3,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )
    assert result.entry_comparisons is None
