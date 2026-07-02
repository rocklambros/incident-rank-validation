"""Secondary measurable-subset kappa (F14 footnote, U3 Cluster A, T3).

Computes kappa restricted to measurable entries only, surfacing the
STANDING_CAVEAT contradiction: concordance.py:48 claims 'computed over
the measurable subset only', but the as-shipped concordance.json kappa
(0.2028985507246377, total_count=20) is over ALL 20 inference-union-vote
entries.  The measurable-subset kappa (~0.12) differs.

This contradiction is surfaced as a disclosure, never silently resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from engine.baselines.previous_ranking import _tier_bounds
from engine.decide.concordance import _ranks_from_incidence
from engine.decide.kappa import quadratic_weighted_kappa

STANDING_CAVEAT_CONTRADICTION = (
    "concordance.py:48 (STANDING_CAVEAT) claims 'computed over the measurable "
    "subset only', but the as-shipped concordance.json kappa (0.2028985507246377) "
    "is over all 20 inference-union-vote entries (total_count=20, measurable_count=17). "
    "The measurable-subset kappa (~0.12) differs from the shipped number. "
    "This contradiction is surfaced as a standing disclosure, not silently resolved."
)


@dataclass(frozen=True)
class MeasurableSubsetKappaResult:
    """Secondary footnote kappa restricted to measurable entries."""

    kappa_median: float
    kappa_ci_lo: float
    kappa_ci_hi: float
    n_measurable: int
    tier_boundaries: tuple[int, ...]
    standing_caveat_contradiction: str


def compute_measurable_subset_kappa(
    lambda_samples: npt.NDArray[np.float64],
    vote_rank_samples: npt.NDArray[np.float64],
    inf_entry_ids: tuple[str, ...],
    vote_entry_ids: tuple[str, ...],
    measurable_entry_ids: tuple[str, ...],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> MeasurableSubsetKappaResult:
    """Compute kappa restricted to measurable entries (F14 secondary footnote).

    Slices both lambda columns and vote columns to the measurable subset and
    recomputes incidence ranks over only those entries (F13 column alignment).
    Tier boundaries recomputed from n_measurable, not n_common.

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
    measurable_entry_ids:
        Subset of entry IDs deemed measurable (excludes frame-blind entries).
    entry_strata:
        Maps each entry to its observed strata.
    stratum_sizes:
        Maps each stratum name to its incident count.

    Returns
    -------
    MeasurableSubsetKappaResult with standing_caveat_contradiction baked in.
    """
    measurable_set = set(measurable_entry_ids)
    vote_set = set(vote_entry_ids)
    common: list[str] = [
        e for e in inf_entry_ids if e in vote_set and e in measurable_set
    ]
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
            "compute_measurable_subset_kappa: all draws produced NaN kappa"
        )

    kappa_arr = np.array(kappas, dtype=np.float64)
    return MeasurableSubsetKappaResult(
        kappa_median=float(np.median(kappa_arr)),
        kappa_ci_lo=float(np.percentile(kappa_arr, 2.5)),
        kappa_ci_hi=float(np.percentile(kappa_arr, 97.5)),
        n_measurable=n_common,
        tier_boundaries=tier_boundaries,
        standing_caveat_contradiction=STANDING_CAVEAT_CONTRADICTION,
    )
