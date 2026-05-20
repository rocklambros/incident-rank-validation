"""Point-estimate robustness twin for incident prevalence.

See HANDOFF §5.5: non-Bayesian de-biased prevalence estimates without NUTS.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.calibrate.beta import Calibration
from engine.model.overlap import OverlapWeights

__all__ = ["TwinResult", "compute_twin"]


@dataclass(frozen=True, slots=True)
class TwinResult:
    """Point-estimate de-biased prevalence (no posterior)."""

    entry_ids: tuple[str, ...]
    prevalence_estimates: dict[str, float]  # entry_id -> de-biased prevalence
    rank: tuple[str, ...]  # entries sorted by prevalence, descending


def compute_twin(
    measurable_entries: tuple[str, ...],
    strata: tuple[str, ...],
    observed_counts: dict[tuple[str, str], int],
    stratum_sizes: dict[str, int],
    calibration: Calibration,
    overlap: OverlapWeights,
) -> TwinResult:
    """Non-Bayesian robustness twin: de-biased count estimates.

    For each entry and stratum:
        adjusted_count = max(0, observed - fp_estimate) / recall_estimate

    FP estimate comes from overlap weights applied to source entry observations
    scaled by (1 - source_precision).  Recall and precision estimates are the
    Beta posterior means, falling back to sensible defaults when absent.
    """
    prevalences: dict[str, float] = {}

    for entry in measurable_entries:
        total_adjusted = 0.0
        total_exposure = 0.0

        for stratum in strata:
            obs = observed_counts.get((entry, stratum), 0)
            size = stratum_sizes[stratum]

            # Calibration means with fallbacks
            key = (entry, stratum)
            recall_mean = (
                calibration.recall[key].mean if key in calibration.recall else 0.5
            )
            # precision_mean is fetched for completeness but is used via source
            # entries below; we keep the local variable for documentation clarity.

            # Estimate FPs landing in this entry from overlap
            fp_estimate = 0.0
            if entry in overlap.weights:
                for source, weight in overlap.weights[entry].items():
                    source_obs = observed_counts.get((source, stratum), 0)
                    source_prec_key = (source, stratum)
                    source_prec = (
                        calibration.precision[source_prec_key].mean
                        if source_prec_key in calibration.precision
                        else 0.9
                    )
                    fp_estimate += weight * source_obs * (1.0 - source_prec)

            # De-bias: (observed - FP) / recall
            adjusted = max(0.0, obs - fp_estimate) / max(recall_mean, 0.01)
            total_adjusted += adjusted
            total_exposure += size

        prevalences[entry] = total_adjusted / max(total_exposure, 1.0)

    ranked = tuple(
        sorted(prevalences, key=lambda e: prevalences[e], reverse=True)
    )
    return TwinResult(
        entry_ids=measurable_entries,
        prevalence_estimates=prevalences,
        rank=ranked,
    )
