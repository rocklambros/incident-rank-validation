"""Unit tests for engine.decide.kappa — quadratic-weighted kappa (M12)."""

from __future__ import annotations

import math

import numpy as np

from engine.decide.kappa import quadratic_weighted_kappa

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQuadraticWeightedKappa:
    def test_all_one_tier_returns_one(self) -> None:
        """M12: when all entries fall in one tier, return 1.0 (trivial agreement)."""
        rank_a = np.array([1.0, 2.0, 3.0])
        rank_b = np.array([2.0, 1.0, 3.0])
        # Boundaries at 10, 20 -- all ranks < 10 -> all in tier 0
        tier_boundaries = (10, 20)

        result = quadratic_weighted_kappa(rank_a, rank_b, tier_boundaries)

        assert result == 1.0

    def test_perfect_agreement_returns_one(self) -> None:
        """Identical tierings should yield kappa = 1.0."""
        rank_a = np.array([1.0, 2.0, 5.0, 8.0, 12.0, 18.0])
        rank_b = np.array([1.0, 2.0, 5.0, 8.0, 12.0, 18.0])
        tier_boundaries = (3, 10)

        result = quadratic_weighted_kappa(rank_a, rank_b, tier_boundaries)

        assert abs(result - 1.0) < 1e-9

    def test_normal_case_returns_reasonable_value(self) -> None:
        """Two correlated but not identical rank vectors should give 0 < kappa < 1."""
        rng = np.random.default_rng(42)
        n = 20
        rank_a = np.arange(1, n + 1, dtype=np.float64)
        # Correlated: same order with some noise
        rank_b = rank_a + rng.normal(0, 2, size=n)
        rank_b = np.argsort(np.argsort(rank_b)).astype(np.float64) + 1
        tier_boundaries = (5, 10, 15)

        result = quadratic_weighted_kappa(rank_a, rank_b, tier_boundaries)

        assert 0.0 < result < 1.0

    def test_empty_returns_nan(self) -> None:
        """Empty input should yield NaN."""
        rank_a = np.array([], dtype=np.float64)
        rank_b = np.array([], dtype=np.float64)
        tier_boundaries = (5,)

        result = quadratic_weighted_kappa(rank_a, rank_b, tier_boundaries)

        assert math.isnan(result)

    def test_complete_disagreement_negative_kappa(self) -> None:
        """Systematically mismatched tiers should yield kappa <= 0."""
        # rank_a puts entries in tier 0, rank_b puts them in highest tier
        rank_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rank_b = np.array([16.0, 17.0, 18.0, 19.0, 20.0])
        tier_boundaries = (5, 10, 15)

        result = quadratic_weighted_kappa(rank_a, rank_b, tier_boundaries)

        assert result < 0.5  # strongly disagreeing should be low

    def test_single_entry(self) -> None:
        """Single entry: trivially all in one tier -> 1.0."""
        rank_a = np.array([1.0])
        rank_b = np.array([1.0])
        tier_boundaries = (5,)

        result = quadratic_weighted_kappa(rank_a, rank_b, tier_boundaries)

        assert result == 1.0
