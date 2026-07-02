"""Reproduce the as-shipped 2026 previous ranking (byte-pinned to concordance.json).

Mirrors compute_concordance's draw loop (concordance.py:193-213) over the FULL
common set (no measurability filter — concordance.py:178) using
_ranks_from_incidence + quadratic_weighted_kappa.

The reproduced kappa MUST be byte-equal to cycles/2026/results/concordance.json:
    weighted_kappa_median  = 0.2028985507246377
    weighted_kappa_ci      = [-0.1594202898550725, 0.5652173913043478]
    total_count (n_common) = 20
    tier_boundaries        = (6, 12)

This is the "previous" side of the U9 compare/contrast and the preserve-previous-
rankings mandate.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from engine.decide.concordance import _ranks_from_incidence
from engine.decide.kappa import quadratic_weighted_kappa


class PreviousRankingResult(NamedTuple):
    """Frozen previous-ranking output (byte-pinnable to concordance.json)."""

    ranking: tuple[str, ...]  # best -> worst (median lambda*size order)
    kappa_median: float
    kappa_ci_lo: float
    kappa_ci_hi: float
    n_common: int
    tier_boundaries: tuple[int, ...]


def _tier_bounds(n_common: int) -> tuple[int, ...]:
    """Replicates concordance.py:187-191 exactly."""
    if n_common <= 3:
        return tuple(range(1, n_common))
    third = n_common // 3
    return (third, 2 * third)


def compute_previous_ranking(
    lambda_samples: npt.NDArray[np.float64],
    vote_rank_samples: npt.NDArray[np.float64],
    inf_entry_ids: tuple[str, ...],
    vote_entry_ids: tuple[str, ...],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> PreviousRankingResult:
    """Reproduce the as-shipped incidence-ranking kappa over all common entries.

    No measurability filter — mirrors concordance.py:178 exactly.
    n_draws = min(len(lambda_samples), len(vote_rank_samples)).
    Tier boundaries: (n_common // 3, 2 * n_common // 3) for n > 3.

    Parameters
    ----------
    lambda_samples:
        (N, n_inf_entries) posterior lambda draws.
    vote_rank_samples:
        (M, n_vote_entries) bootstrap vote rank draws.
    inf_entry_ids:
        Ordered entry IDs from the inference result.
    vote_entry_ids:
        Ordered entry IDs from the vote posterior.
    entry_strata:
        Maps each entry to its observed strata (must cover all common entries).
    stratum_sizes:
        Maps each stratum name to its incident count.

    Returns
    -------
    PreviousRankingResult with kappa_median/ci byte-pinnable to concordance.json.
    """
    vote_set = set(vote_entry_ids)
    common: list[str] = [e for e in inf_entry_ids if e in vote_set]
    n_common = len(common)

    inf_idx: dict[str, int] = {e: i for i, e in enumerate(inf_entry_ids)}
    vote_idx: dict[str, int] = {e: i for i, e in enumerate(vote_entry_ids)}

    tier_boundaries = _tier_bounds(n_common)

    n_draws = min(len(lambda_samples), len(vote_rank_samples))
    kappas: list[float] = []

    for s in range(n_draws):
        inc_ranks = _ranks_from_incidence(
            lambda_samples[s], inf_idx, common, entry_strata, stratum_sizes
        )
        vote_ranks = np.array([vote_rank_samples[s][vote_idx[e]] for e in common])
        k = quadratic_weighted_kappa(inc_ranks, vote_ranks, tier_boundaries)
        if not np.isnan(k):
            kappas.append(k)

    if not kappas:
        raise ValueError(
            "compute_previous_ranking: all draws produced NaN kappa; "
            "check that entry_strata and stratum_sizes cover all common entries"
        )

    kappa_arr = np.array(kappas, dtype=np.float64)
    median_k = float(np.median(kappa_arr))
    ci_lo = float(np.percentile(kappa_arr, 2.5))
    ci_hi = float(np.percentile(kappa_arr, 97.5))

    # Derive ranking from median lambda (same method as oracle_incidence_ranking)
    median_lambda = np.median(lambda_samples, axis=0)
    inc_vals: dict[str, float] = {
        e: float(median_lambda[inf_idx[e]])
        * float(sum(stratum_sizes[st] for st in entry_strata[e]))
        for e in common
    }
    ranking = tuple(sorted(common, key=lambda e: (-inc_vals[e], e)))

    return PreviousRankingResult(
        ranking=ranking,
        kappa_median=median_k,
        kappa_ci_lo=ci_lo,
        kappa_ci_hi=ci_hi,
        n_common=n_common,
        tier_boundaries=tier_boundaries,
    )
