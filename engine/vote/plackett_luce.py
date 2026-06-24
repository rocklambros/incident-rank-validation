"""Davidson (1970) tie-aware paired-comparison vote model (Plan 8c).

The OWASP importance ballots are 1-5 scores over the candidate entries, which
average-rank into partial orders saturated with ties.  Plackett-Luce proper
models strict total orderings, so we use the Davidson tie-aware extension of
Bradley-Terry, the standard aggregation for tie-heavy paired-comparison data.

For an unordered pair (i, j) with worths pi_i = exp(theta_i), pi_j = exp(theta_j),
a tie parameter nu = exp(gamma) >= 0, and s = sqrt(pi_i * pi_j):

    P(i beats j) = pi_i / (pi_i + pi_j + nu * s)
    P(j beats i) = pi_j / (pi_i + pi_j + nu * s)
    P(i ties  j) = nu * s / (pi_i + pi_j + nu * s)

As nu -> 0 this reduces to Bradley-Terry / Plackett-Luce worths, so
``include_ties=False`` (drop tied pairs, fix nu = 0) is a free drop-ties
sensitivity lens.  Worths are identified only up to scale and complete
separation drives a dominant entry's worth to infinity; a single L2 ridge on
theta makes the penalized negative log-likelihood strictly convex AND bounds
separation.  Rao-Kupper is an equivalent threshold-parametrized tie model; one
tie-aware model is sufficient for a robustness lens (YAGNI).

CPU-deterministic: the point fit starts from theta = 0, gamma = 0 via L-BFGS-B.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

DEFAULT_RIDGE: float = 1e-3
# gamma = log(nu) is bounded so that all-strict ballots (which drive nu -> 0)
# do not stall the optimizer at gamma -> -inf.  nu in [~2e-9, ~4.9e8].
_GAMMA_BOUND: float = 20.0


@dataclass(frozen=True, slots=True)
class DavidsonFit:
    """Point fit of the Davidson tie-aware model."""

    entries: tuple[str, ...]
    log_worths: dict[str, float]
    worths: dict[str, float]
    tie_param: float
    ranking: tuple[str, ...]  # best -> worst by worth
    converged: bool


def _pairwise_counts(
    rankings: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reduce respondent rank vectors to pairwise win/tie counts.

    ``rankings`` is (n_respondents, n_entries); lower rank = more preferred,
    equal rank = tie.  Returns ``(wins, ties)`` where ``wins[a, b]`` counts
    respondents preferring entry a over entry b and ``ties[i, j]`` (for i < j)
    counts respondents who tied them.
    """
    _, n = rankings.shape
    wins = np.zeros((n, n), dtype=np.float64)
    ties = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = rankings[:, i] - rankings[:, j]
            wins[i, j] = float(np.sum(diff < 0.0))  # i preferred (lower rank)
            wins[j, i] = float(np.sum(diff > 0.0))  # j preferred
            ties[i, j] = float(np.sum(diff == 0.0))
    return wins, ties


def _neg_log_likelihood(
    params: npt.NDArray[np.float64],
    wins: npt.NDArray[np.float64],
    ties: npt.NDArray[np.float64],
    iu: npt.NDArray[np.intp],
    ju: npt.NDArray[np.intp],
    ridge: float,
    include_ties: bool,
) -> float:
    """Penalized negative log-likelihood of the Davidson model.

    ``iu``/``ju`` are the upper-triangle (i < j) pair indices.  When
    ``include_ties`` is True the last element of ``params`` is gamma = log(nu).
    """
    n = wins.shape[0]
    theta = params[:n]
    ti = theta[iu]
    tj = theta[ju]
    wins_ij = wins[iu, ju]
    wins_ji = wins[ju, iu]
    if include_ties:
        gamma = float(params[n])
        log_s = 0.5 * (ti + tj)
        denom = np.exp(ti) + np.exp(tj) + np.exp(gamma + log_s)
        log_denom = np.log(denom)
        ties_ij = ties[iu, ju]
        ll = (
            np.sum(wins_ij * (ti - log_denom))
            + np.sum(wins_ji * (tj - log_denom))
            + np.sum(ties_ij * (gamma + log_s - log_denom))
        )
    else:
        denom = np.exp(ti) + np.exp(tj)
        log_denom = np.log(denom)
        ll = np.sum(wins_ij * (ti - log_denom)) + np.sum(wins_ji * (tj - log_denom))
    penalty = ridge * float(np.sum(theta * theta))
    return float(-ll + penalty)


def fit_davidson(
    rankings: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    ridge: float = DEFAULT_RIDGE,
    include_ties: bool = True,
) -> DavidsonFit:
    """Fit the Davidson tie-aware model by ridge-penalized MLE (L-BFGS-B)."""
    n = len(entry_ids)
    if rankings.shape[1] != n:
        raise ValueError(
            f"entry_ids length {n} does not match rankings columns "
            f"{rankings.shape[1]}"
        )
    wins, ties = _pairwise_counts(rankings)
    iu, ju = np.triu_indices(n, k=1)

    if include_ties:
        x0 = np.zeros(n + 1, dtype=np.float64)
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * n + [
            (-_GAMMA_BOUND, _GAMMA_BOUND)
        ]
    else:
        x0 = np.zeros(n, dtype=np.float64)
        bounds = [(None, None)] * n

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(wins, ties, iu, ju, ridge, include_ties),
        method="L-BFGS-B",
        bounds=bounds,
    )
    x = np.asarray(result.x, dtype=np.float64)
    theta = x[:n]
    tie_param = float(np.exp(x[n])) if include_ties else 0.0

    log_worths = {entry_ids[k]: float(theta[k]) for k in range(n)}
    worths = {entry_ids[k]: float(np.exp(theta[k])) for k in range(n)}
    # Deterministic ranking: highest worth first, ties broken by entry id.
    order = sorted(range(n), key=lambda k: (-theta[k], entry_ids[k]))
    ranking = tuple(entry_ids[k] for k in order)

    return DavidsonFit(
        entries=entry_ids,
        log_worths=log_worths,
        worths=worths,
        tie_param=tie_param,
        ranking=ranking,
        converged=bool(result.success),
    )
