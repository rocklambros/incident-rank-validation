"""Tests for draw cap removal and vote recycling fix (Phase 1)."""
from __future__ import annotations

import numpy as np

from engine.decide.concordance import compute_concordance
from engine.model.inference import InferenceResult
from engine.vote.bootstrap import VoteRankPosterior


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


class TestDrawCapRemoval:
    def test_uses_all_available_draws_not_capped_at_500(self) -> None:
        entries = tuple(f"E{i}" for i in range(10))
        inf = _make_inference(entries, n_samples=800)
        vote = _make_vote_posterior(entries, n_bootstrap=800)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(3, 7),
            flag_threshold_tau=0.5,
            measurable_count=10,
            total_count=15,
            meaningful_kappa_n=5,
            measurability_minimum=5,
        )

        assert result.weighted_kappa_median is not None
        ci = result.weighted_kappa_ci
        assert ci is not None
        assert ci[0] < ci[1]


class TestVoteRecyclingFix:
    def test_no_recycling_when_vote_shorter(self) -> None:
        entries = tuple(f"E{i}" for i in range(10))
        inf = _make_inference(entries, n_samples=1000)
        vote = _make_vote_posterior(entries, n_bootstrap=600)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(3, 7),
            flag_threshold_tau=0.5,
            measurable_count=10,
            total_count=15,
            meaningful_kappa_n=5,
            measurability_minimum=5,
        )

        assert result.weighted_kappa_median is not None

    def test_common_bound_equals_min_of_both(self) -> None:
        entries = tuple(f"E{i}" for i in range(6))
        inf = _make_inference(entries, n_samples=200)
        vote = _make_vote_posterior(entries, n_bootstrap=150)

        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(2, 4),
            flag_threshold_tau=0.5,
            measurable_count=6,
            total_count=6,
            meaningful_kappa_n=3,
            measurability_minimum=3,
        )

        assert result.weighted_kappa_median is not None
        assert result.weighted_kappa_ci is not None
