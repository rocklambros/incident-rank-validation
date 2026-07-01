"""Bare-lambda sensitivity disclosure (U3 Cluster A, T2).

Computes the kappa using _ranks_from_lambda (bare lambda, no size weighting) and
its method-delta vs the incidence kappa.  This path is DEAD CODE in concordance.py
— it never produced a published number.

On 2026 data the kappa MEDIANS coincide (method_kappa_delta == 0.0) even though
individual draw rankings differ on some draws.  This finding is disclosed
and is NEVER credited as a method gain.  The exact count is computed and stored
in draws_differing / n_draws_total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from engine.baselines.previous_ranking import _tier_bounds
from engine.decide.concordance import _ranks_from_incidence, _ranks_from_lambda
from engine.decide.kappa import quadratic_weighted_kappa


@dataclass(frozen=True)
class BareLambdaSensitivityResult:
    """Bare-lambda kappa and method-delta (always disclosed, never credited)."""

    ranking: tuple[str, ...]  # bare-lambda median ranking, best -> worst
    kappa_median: float
    method_kappa_delta: float  # bare_kappa_median - incidence_kappa_median
    draws_differing: int  # number of draws where bare-λ ranking != λ·size ranking
    n_draws_total: int   # total draws used (= min(N_lambda, N_vote))
    disclosure: str


_DISCLOSURE_TEMPLATE = (
    "Bare-lambda (_ranks_from_lambda, dead code in concordance.py) kappa median "
    "differs from the incidence (lambda*size) kappa median by {delta:+.9f} on "
    "this data. This method delta is disclosed for transparency and is NOT "
    "credited as a method gain in any comparison. On 2026 OWASP-LLM data the "
    "delta is 0.0 (kappa medians coincide) even though individual draw rankings "
    "differ on {n_differing}/{n_draws} draws."
)


def compute_bare_lambda_sensitivity(
    lambda_samples: npt.NDArray[np.float64],
    vote_rank_samples: npt.NDArray[np.float64],
    inf_entry_ids: tuple[str, ...],
    vote_entry_ids: tuple[str, ...],
    incidence_kappa_median: float,
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> BareLambdaSensitivityResult:
    """Compute bare-lambda ranking kappa and its delta vs incidence kappa.

    Parameters
    ----------
    lambda_samples:
        (N, n_entries) posterior lambda draws.
    vote_rank_samples:
        (M, n_entries) bootstrap vote rank draws.
    inf_entry_ids:
        Ordered entry IDs from the inference result.
    vote_entry_ids:
        Ordered entry IDs from the vote posterior.
    incidence_kappa_median:
        The incidence kappa median (from compute_previous_ranking) to diff against.
    entry_strata:
        Maps each entry ID to its observed strata names (needed for incidence ranks).
    stratum_sizes:
        Maps each stratum name to its incident count (needed for incidence ranks).

    Returns
    -------
    BareLambdaSensitivityResult with disclosure text and computed draws_differing.
    """
    vote_set = set(vote_entry_ids)
    common: list[str] = [e for e in inf_entry_ids if e in vote_set]
    n_common = len(common)

    inf_idx: dict[str, int] = {e: i for i, e in enumerate(inf_entry_ids)}
    vote_idx: dict[str, int] = {e: i for i, e in enumerate(vote_entry_ids)}

    tier_boundaries = _tier_bounds(n_common)

    n_draws = min(len(lambda_samples), len(vote_rank_samples))
    kappas: list[float] = []
    draws_differing: int = 0

    for s in range(n_draws):
        lam_draw = lambda_samples[s]
        bare_ranks = _ranks_from_lambda(lam_draw, inf_idx, common)
        inc_ranks = _ranks_from_incidence(lam_draw, inf_idx, common, entry_strata, stratum_sizes)
        vote_ranks = np.array([vote_rank_samples[s][vote_idx[e]] for e in common])
        k = quadratic_weighted_kappa(bare_ranks, vote_ranks, tier_boundaries)
        if not np.isnan(k):
            kappas.append(k)
        # Count draws where the bare-λ and λ·size rankings differ
        if not np.array_equal(bare_ranks, inc_ranks):
            draws_differing += 1

    bare_median = float(np.median(kappas)) if kappas else float("nan")
    delta = bare_median - incidence_kappa_median

    # Bare-lambda median ranking (sort by median lambda, descending)
    median_lambda = np.median(lambda_samples, axis=0)
    bare_vals: dict[str, float] = {e: float(median_lambda[inf_idx[e]]) for e in common}
    ranking = tuple(sorted(common, key=lambda e: (-bare_vals[e], e)))

    return BareLambdaSensitivityResult(
        ranking=ranking,
        kappa_median=bare_median,
        method_kappa_delta=delta,
        draws_differing=draws_differing,
        n_draws_total=n_draws,
        disclosure=_DISCLOSURE_TEMPLATE.format(
            delta=delta,
            n_differing=draws_differing,
            n_draws=n_draws,
        ),
    )
