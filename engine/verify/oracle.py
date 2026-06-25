"""Independent consistency-check oracle (Plan 8d).

Re-derives the engine's headline deliverables by a DIFFERENT method on the
pinned numpy/scipy stack and compares within pre-declared tolerances.  This
module MUST NOT import engine estimator code (concordance / plackett_luce /
model / calibrate); it reads persisted artifacts as plain numpy/JSON so it is
a genuine cross-check, not a re-run.  It is a CONSISTENCY check, not
independent verification (shared author/conceptual source).
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


def _incidence_value(
    lam: float,
    entry: str,
    entry_strata: Mapping[str, tuple[str, ...]],
    stratum_sizes: Mapping[str, int],
) -> float:
    """lambda_e * sum of sizes of all strata entry e was observed in."""
    total_size = float(sum(stratum_sizes[s] for s in entry_strata[entry]))
    return lam * total_size


def oracle_incidence_ranking(
    lambda_samples: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    entry_strata: Mapping[str, tuple[str, ...]],
    stratum_sizes: Mapping[str, int],
) -> tuple[str, ...]:
    """Re-derive the incidence ranking (best->worst) from median lambda x size."""
    median_lambda = np.median(lambda_samples, axis=0)
    incidence = {
        e: _incidence_value(float(median_lambda[i]), e, entry_strata, stratum_sizes)
        for i, e in enumerate(entry_ids)
    }
    order = sorted(entry_ids, key=lambda e: (-incidence[e], e))
    return tuple(order)


def oracle_incidence_intervals(
    lambda_samples: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    entry_strata: Mapping[str, tuple[str, ...]],
    stratum_sizes: Mapping[str, int],
) -> dict[str, tuple[float, float]]:
    """Per-entry (2.5, 97.5) percentile incidence interval."""
    intervals: dict[str, tuple[float, float]] = {}
    for i, e in enumerate(entry_ids):
        total_size = float(sum(stratum_sizes[s] for s in entry_strata[e]))
        draws = lambda_samples[:, i] * total_size
        intervals[e] = (
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        )
    return intervals


def _pairwise_wins_halfcredit(
    rankings: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-entry win credit (1 strict / 0.5 tie) and pairwise comparison counts.

    Different tie handling than the engine's Davidson nu model: ties are split
    as half a win each.  This is intentional independence for the cross-check.
    """
    n_resp, n = rankings.shape
    wins = np.zeros(n, dtype=np.float64)
    comparisons = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = rankings[:, i] - rankings[:, j]
            i_wins = float(np.sum(diff < 0.0))  # lower rank = preferred
            j_wins = float(np.sum(diff > 0.0))
            ties = float(np.sum(diff == 0.0))
            wins[i] += i_wins + 0.5 * ties
            wins[j] += j_wins + 0.5 * ties
            comparisons[i, j] = float(n_resp)
            comparisons[j, i] = float(n_resp)
    return wins, comparisons


def oracle_pl_ranking_mm(
    rankings: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> tuple[str, ...]:
    """Bradley-Terry worths via MM/fixed-point (Hunter 2004), then rank.

    Update: pi_i <- w_i / sum_j!=i  n_ij / (pi_i + pi_j) ; renormalize to sum 1.
    A different optimizer family than the engine's scipy.optimize L-BFGS-B.
    """
    n = len(entry_ids)
    wins, comparisons = _pairwise_wins_halfcredit(rankings)
    pi = np.full(n, 1.0 / n, dtype=np.float64)
    eps = 1e-12
    for _ in range(max_iter):
        denom = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                denom[i] += comparisons[i, j] / (pi[i] + pi[j] + eps)
        new_pi = wins / (denom + eps)
        total = float(np.sum(new_pi))
        if total <= 0.0:
            break
        new_pi = new_pi / total
        if float(np.max(np.abs(new_pi - pi))) < tol:
            pi = new_pi
            break
        pi = new_pi
    worths = {e: float(pi[i]) for i, e in enumerate(entry_ids)}
    order = sorted(entry_ids, key=lambda e: (-worths[e], e))
    return tuple(order)
