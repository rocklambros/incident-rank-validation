"""Unit tests for engine.decide.multiplicity — permutation null (M15)."""

from __future__ import annotations

import numpy as np

from engine.decide.multiplicity import MultiplicityDisclosure, permutation_null

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPermutationNull:
    def test_runs_and_returns_disclosure(self) -> None:
        """Basic smoke test: permutation null should return a valid disclosure."""
        n = 15
        incident_ranks = np.arange(1, n + 1, dtype=np.float64)
        vote_ranks = np.arange(1, n + 1, dtype=np.float64)
        tier_boundaries = (5, 10)

        result = permutation_null(
            incident_ranks, vote_ranks, tier_boundaries,
            observed_kappa=0.8, n_permutations=200, seed=42,
        )

        assert isinstance(result, MultiplicityDisclosure)
        assert result.n_permutations == 200

    def test_p_value_between_zero_and_one(self) -> None:
        """p_value must be in [0, 1]."""
        n = 20
        rng = np.random.default_rng(7)
        incident_ranks = np.arange(1, n + 1, dtype=np.float64)
        vote_ranks = rng.permutation(incident_ranks).astype(np.float64)
        tier_boundaries = (5, 10, 15)

        result = permutation_null(
            incident_ranks, vote_ranks, tier_boundaries,
            observed_kappa=0.3, n_permutations=500, seed=42,
        )

        assert 0.0 <= result.p_value <= 1.0

    def test_high_kappa_yields_low_p(self) -> None:
        """Perfect agreement should be rare under the null -> low p-value."""
        n = 20
        incident_ranks = np.arange(1, n + 1, dtype=np.float64)
        vote_ranks = np.arange(1, n + 1, dtype=np.float64)
        tier_boundaries = (5, 10, 15)

        from engine.decide.kappa import quadratic_weighted_kappa
        observed = quadratic_weighted_kappa(incident_ranks, vote_ranks, tier_boundaries)

        result = permutation_null(
            incident_ranks, vote_ranks, tier_boundaries,
            observed_kappa=observed, n_permutations=500, seed=42,
        )

        # With perfect agreement observed, p should be quite low
        assert result.p_value < 0.1

    def test_null_kappa_ci_ordered(self) -> None:
        """Null CI lower bound should be <= upper bound."""
        n = 10
        ranks = np.arange(1, n + 1, dtype=np.float64)
        tier_boundaries = (3, 7)

        result = permutation_null(
            ranks, ranks, tier_boundaries,
            observed_kappa=0.5, n_permutations=300, seed=42,
        )

        assert result.null_kappa_ci[0] <= result.null_kappa_ci[1]

    def test_deterministic_with_same_seed(self) -> None:
        """Two calls with the same seed must produce identical results."""
        n = 10
        ranks = np.arange(1, n + 1, dtype=np.float64)
        tier_boundaries = (3, 7)

        r1 = permutation_null(ranks, ranks, tier_boundaries, 0.5, 100, seed=42)
        r2 = permutation_null(ranks, ranks, tier_boundaries, 0.5, 100, seed=42)

        assert r1.p_value == r2.p_value
        assert r1.null_kappa_median == r2.null_kappa_median
