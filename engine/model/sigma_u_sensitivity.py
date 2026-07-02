"""sigma_u prior-sensitivity sweep + prior-dominance decision rule (Plan 8b).

With ~20 groups sigma_u is weakly identified; if its posterior tracks the prior
scale, pooling is prior-driven and should be abandoned in favor of independent
per-entry rates (RARR design Sec 5.3).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sweep_sigma_u(
    scales: tuple[float, ...],
    run_fn: Callable[[float], float],
) -> dict[float, float]:
    """Run the hierarchical fit at each prior scale; return scale -> sigma_u median."""
    return {scale: run_fn(scale) for scale in scales}


def is_prior_dominated(
    scales: tuple[float, ...],
    sigma_u_by_scale: dict[float, float],
    rel_tol: float = 0.25,
) -> bool:
    """True if the sigma_u posterior tracks the prior scale (prior-dominated).

    Heuristic: if the ratio sigma_u/scale is roughly constant across scales
    (coefficient of variation of the ratios below rel_tol), the posterior is
    following the prior rather than the data, so sigma_u is not identified.
    """
    ratios = np.array([sigma_u_by_scale[s] / s for s in scales if s > 0])
    if ratios.size < 2:
        return False
    mean = float(ratios.mean())
    if mean == 0.0:
        return False
    cv = float(ratios.std() / mean)
    return cv < rel_tol
