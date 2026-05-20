"""Quadratic-weighted Cohen's kappa (HANDOFF §5.5)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def quadratic_weighted_kappa(
    rank_a: npt.NDArray[np.float64],
    rank_b: npt.NDArray[np.float64],
    tier_boundaries: tuple[int, ...],
) -> float:
    """Compute quadratic-weighted kappa between two rank vectors.

    M12: when all entries fall in one tier, return 1.0 (perfect trivial agreement).
    """

    def tier_of(rank: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        t = np.zeros_like(rank, dtype=np.int32)
        for i, b in enumerate(tier_boundaries):
            t = np.where(rank > b, i + 1, t)
        return t

    ta = tier_of(rank_a)
    tb = tier_of(rank_b)
    n_tiers = len(tier_boundaries) + 1

    obs = np.zeros((n_tiers, n_tiers))
    for x, y in zip(ta, tb, strict=False):
        obs[int(x), int(y)] += 1

    n = float(obs.sum())
    if n == 0:
        return float("nan")

    obs /= n
    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    expected = np.outer(row, col)

    weights = (np.arange(n_tiers)[:, None] - np.arange(n_tiers)[None, :]) ** 2
    if weights.max() == 0:
        return 1.0

    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement < 1e-9:
        return 1.0  # M12: all entries in one tier

    return 1.0 - float((weights * obs).sum()) / expected_disagreement
