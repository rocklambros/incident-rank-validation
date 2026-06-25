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


def oracle_sigma_u_surrogate(lambda_samples: npt.NDArray[np.float64]) -> float:
    """DerSimonian-Laird between-entry SD of log-lambda (random-effects moment).

    A closed-form surrogate for the engine's NUTS HalfNormal sigma_u posterior:
    y_e = median(log lambda_e), v_e = var(log lambda_e) (within-entry sampling
    variance).  tau^2 = max(0, (Q - (k-1)) / C) with DSL weights w_e = 1/v_e.
    No MCMC, no scipy.optimize.  Computed on the UNPOOLED poisson_flat samples
    so it is an independent estimate of the pooling SD, not a re-read of the
    hierarchical posterior.
    """
    k = lambda_samples.shape[1]
    if k < 2:
        return 0.0
    log_lambda = np.log(np.clip(lambda_samples, 1e-12, None))
    y = np.median(log_lambda, axis=0)
    v = np.var(log_lambda, axis=0, ddof=1)
    v = np.clip(v, 1e-12, None)
    w = 1.0 / v
    sum_w = float(np.sum(w))
    y_bar = float(np.sum(w * y) / sum_w)
    q = float(np.sum(w * (y - y_bar) ** 2))
    c = sum_w - float(np.sum(w**2)) / sum_w
    if c <= 0.0:
        return 0.0
    tau2 = max(0.0, (q - (k - 1)) / c)
    return float(np.sqrt(tau2))
