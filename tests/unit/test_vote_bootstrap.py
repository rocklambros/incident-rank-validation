"""Unit tests for engine.vote.bootstrap — vote-rank posterior."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from engine.vote.bootstrap import bootstrap_vote_ranks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rankings(
    n_respondents: int = 20,
    n_entries: int = 4,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Generate random respondent ranking matrices."""
    rng = np.random.default_rng(seed)
    rankings = np.zeros((n_respondents, n_entries), dtype=np.float64)
    for i in range(n_respondents):
        rankings[i] = rng.permutation(n_entries) + 1  # ranks 1..n_entries
    return rankings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBootstrapVoteRanks:
    def test_rank_samples_shape(self) -> None:
        """rank_samples must be (n_bootstrap, n_entries)."""
        n_bootstrap = 200
        n_entries = 4
        entry_ids = tuple(f"E{i}" for i in range(n_entries))
        rankings = _make_rankings(n_respondents=30, n_entries=n_entries)

        result = bootstrap_vote_ranks(
            rankings, entry_ids, n_bootstrap=n_bootstrap, seed=42
        )

        assert result.rank_samples.shape == (n_bootstrap, n_entries)

    def test_median_ranks_in_valid_range(self) -> None:
        """Median ranks should be between 1 and n_entries (inclusive)."""
        n_entries = 5
        entry_ids = tuple(f"E{i}" for i in range(n_entries))
        rankings = _make_rankings(n_respondents=50, n_entries=n_entries)

        result = bootstrap_vote_ranks(rankings, entry_ids, n_bootstrap=500, seed=7)

        for eid, med in result.median_ranks.items():
            assert 1.0 <= med <= float(n_entries), (
                f"Median rank {med} out of range [1, {n_entries}] for {eid}"
            )

    def test_deterministic_with_same_seed(self) -> None:
        """Two calls with the same seed must produce identical rank_samples."""
        entry_ids = ("A", "B", "C")
        rankings = _make_rankings(n_respondents=15, n_entries=3)

        r1 = bootstrap_vote_ranks(rankings, entry_ids, n_bootstrap=100, seed=99)
        r2 = bootstrap_vote_ranks(rankings, entry_ids, n_bootstrap=100, seed=99)

        np.testing.assert_array_equal(r1.rank_samples, r2.rank_samples)

    def test_different_seeds_give_different_results(self) -> None:
        """Different seeds should (overwhelmingly) produce different samples."""
        entry_ids = ("A", "B", "C")
        rankings = _make_rankings(n_respondents=30, n_entries=3)

        r1 = bootstrap_vote_ranks(rankings, entry_ids, n_bootstrap=200, seed=1)
        r2 = bootstrap_vote_ranks(rankings, entry_ids, n_bootstrap=200, seed=2)

        assert not np.array_equal(r1.rank_samples, r2.rank_samples)

    def test_clear_winner_ranked_first(self) -> None:
        """When all respondents agree on rank 1, that entry's median rank is 1."""
        # Entry 0 always ranked 1st by all respondents
        n_respondents = 30
        n_entries = 4
        rankings = np.tile(
            np.arange(1, n_entries + 1, dtype=np.float64), (n_respondents, 1)
        )
        entry_ids = tuple(f"E{i}" for i in range(n_entries))

        result = bootstrap_vote_ranks(
            rankings, entry_ids, n_bootstrap=500, seed=42
        )

        # E0 is always rank 1 — every bootstrap sample must give it rank 1
        assert result.median_ranks["E0"] == pytest.approx(1.0)

    def test_result_metadata(self) -> None:
        """VoteRankPosterior stores n_respondents and n_bootstrap correctly."""
        n_resp = 25
        n_boot = 300
        n_entries = 3
        entry_ids = ("X", "Y", "Z")
        rankings = _make_rankings(n_respondents=n_resp, n_entries=n_entries)

        result = bootstrap_vote_ranks(
            rankings, entry_ids, n_bootstrap=n_boot, seed=0
        )

        assert result.n_respondents == n_resp
        assert result.n_bootstrap == n_boot
        assert result.entries == entry_ids

    def test_frozen_dataclass_immutable(self) -> None:
        """VoteRankPosterior is a frozen dataclass."""
        entry_ids = ("A", "B")
        rankings = _make_rankings(n_respondents=10, n_entries=2)
        result = bootstrap_vote_ranks(rankings, entry_ids, n_bootstrap=50, seed=0)

        with pytest.raises((AttributeError, TypeError)):
            result.n_bootstrap = 999  # type: ignore[misc]

    def test_mismatched_entry_ids_raises(self) -> None:
        """entry_ids length mismatch raises AssertionError."""
        rankings = _make_rankings(n_respondents=10, n_entries=3)
        with pytest.raises(AssertionError):
            bootstrap_vote_ranks(rankings, ("A", "B"), n_bootstrap=10, seed=0)
