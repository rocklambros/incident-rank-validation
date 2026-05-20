"""Vote-rank posterior via bootstrap resampling.

See HANDOFF §5.4: "The vote is resampled by bootstrap over the Raw Results
(Anonymized) respondents to produce a posterior over the vote ranking and
vote tiers."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["VoteRankPosterior", "bootstrap_vote_ranks"]


@dataclass(frozen=True, slots=True)
class VoteRankPosterior:
    """Bootstrap posterior over vote rankings."""

    entries: tuple[str, ...]
    rank_samples: npt.NDArray[np.float64]  # (n_bootstrap, n_entries) — rank per sample
    median_ranks: dict[str, float]
    n_respondents: int
    n_bootstrap: int


def bootstrap_vote_ranks(
    respondent_rankings: npt.NDArray[np.float64],  # (n_respondents, n_entries)
    entry_ids: tuple[str, ...],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> VoteRankPosterior:
    """Bootstrap the vote ranking by resampling respondents.

    Each bootstrap sample draws n_respondents with replacement, computes mean
    rank per entry, then converts to ordinal ranks (1 = best / lowest mean).

    Parameters
    ----------
    respondent_rankings:
        2-D array of shape (n_respondents, n_entries).  Each row is one
        respondent's ranking vector (lower value = higher preference).
    entry_ids:
        Ordered entry identifiers matching the columns of respondent_rankings.
    n_bootstrap:
        Number of bootstrap replicates.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    VoteRankPosterior with rank_samples of shape (n_bootstrap, n_entries).
    """
    rng = np.random.default_rng(seed)
    n_resp, n_entries = respondent_rankings.shape
    assert n_entries == len(entry_ids), (
        f"entry_ids length {len(entry_ids)} does not match "
        f"respondent_rankings columns {n_entries}"
    )

    rank_samples = np.zeros((n_bootstrap, n_entries), dtype=np.float64)

    for b in range(n_bootstrap):
        # Resample respondents with replacement
        indices = rng.integers(0, n_resp, size=n_resp)
        sample = respondent_rankings[indices]
        # Mean rank per entry across resampled respondents
        mean_ranks = sample.mean(axis=0)
        # Convert to ordinal ranks (1 = best, i.e. lowest mean rank)
        order = np.argsort(mean_ranks)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, n_entries + 1, dtype=np.float64)
        rank_samples[b] = ranks

    median_ranks = {
        entry_ids[i]: float(np.median(rank_samples[:, i]))
        for i in range(n_entries)
    }

    return VoteRankPosterior(
        entries=entry_ids,
        rank_samples=rank_samples,
        median_ranks=median_ranks,
        n_respondents=n_resp,
        n_bootstrap=n_bootstrap,
    )
