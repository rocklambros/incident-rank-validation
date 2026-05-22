"""Rollup sub-test (HANDOFF §5.2, §5.5).

Per rolled-up candidate: test whether it carries a large distinct incident
cluster the parent does not absorb. Direction and magnitude reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class RollupVerdict(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class RollupResult:
    parent_entry_id: str
    child_entry_id: str
    verdict: RollupVerdict
    p_distinct_cluster: float
    child_median_lambda: float
    parent_median_lambda: float
    ratio_median: float


def compute_rollup_subtest(
    parent_entry_id: str,
    child_entry_id: str,
    parent_lambda_samples: npt.NDArray[np.float64],
    child_lambda_samples: npt.NDArray[np.float64],
    threshold: float = 0.01,
    p_supported: float = 0.8,
    p_contradicted: float = 0.2,
) -> RollupResult:
    n = min(len(parent_lambda_samples), len(child_lambda_samples))
    parent = parent_lambda_samples[:n]
    child = child_lambda_samples[:n]

    ratio = child / np.clip(parent, 1e-10, None)
    p_above = float(np.mean(ratio > threshold))

    child_med = float(np.median(child))
    parent_med = float(np.median(parent))
    ratio_med = float(np.median(ratio))

    if p_above >= p_supported:
        verdict = RollupVerdict.SUPPORTED
    elif p_above <= p_contradicted:
        verdict = RollupVerdict.CONTRADICTED
    else:
        verdict = RollupVerdict.INDETERMINATE

    return RollupResult(
        parent_entry_id=parent_entry_id,
        child_entry_id=child_entry_id,
        verdict=verdict,
        p_distinct_cluster=p_above,
        child_median_lambda=child_med,
        parent_median_lambda=parent_med,
        ratio_median=ratio_med,
    )
