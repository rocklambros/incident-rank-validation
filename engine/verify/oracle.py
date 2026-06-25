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
