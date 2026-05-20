"""Measurability map: per-entry verdicts with exact Beta CDF quantification.

Per HANDOFF §5.5: "Measurability map first. The report leads with the
per-entry verdict: measurable, classifier-blind-but-bounded, or
frame-blind-unmeasurable."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import beta as beta_dist

from engine.calibrate.beta import Calibration
from engine.model.censoring import CensoringResult, MeasurabilityVerdict

__all__ = ["MeasurabilityMap", "build_measurability_map"]


@dataclass(frozen=True, slots=True)
class MeasurabilityMap:
    """Per-entry measurability verdicts with quantitative detail."""

    verdict: dict[str, MeasurabilityVerdict]
    recall_p_above_threshold: dict[str, float]  # P(recall > threshold) per entry, from Beta CDF
    coverage_ratio: float                        # fraction of entries that are measurable
    measurable: tuple[str, ...]
    classifier_blind: tuple[str, ...]
    frame_blind: tuple[str, ...]
    below_prereg_minimum: bool                   # True if measurable count < minimum

    def to_coverage_json(self) -> str:
        """Serialize to coverage.json for cross-platform comparison (M5)."""
        return json.dumps(
            {
                "coverage_ratio": self.coverage_ratio,
                "measurable": sorted(self.measurable),
                "classifier_blind": sorted(self.classifier_blind),
                "frame_blind": sorted(self.frame_blind),
                "below_prereg_minimum": self.below_prereg_minimum,
                "recall_p_above_threshold": {
                    k: round(v, 6)
                    for k, v in sorted(self.recall_p_above_threshold.items())
                },
            },
            sort_keys=True,
            indent=2,
        ) + "\n"

    def write_coverage(self, path: Path) -> None:
        """Write coverage.json to disk, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_coverage_json())


def build_measurability_map(
    censoring: CensoringResult,
    calibration: Calibration | None,
    measurability_minimum: int,
    recall_threshold: float = 0.1,
) -> MeasurabilityMap:
    """Build the measurability map from censoring + calibration.

    For each measurable/classifier-blind entry, compute P(recall > threshold)
    using the exact Beta CDF. Frame-blind entries get P=0.0.

    Args:
        censoring: partition result from engine.model.censoring.partition_entries
        calibration: Beta posteriors (None on the synthetic pre-calibration path)
        measurability_minimum: minimum measurable count required by pre-registration
        recall_threshold: the recall level whose exceedance probability is computed

    Returns:
        MeasurabilityMap with per-entry verdicts and quantitative recall probabilities.
    """
    total = (
        len(censoring.measurable)
        + len(censoring.classifier_blind)
        + len(censoring.frame_blind)
    )
    coverage = len(censoring.measurable) / total if total > 0 else 0.0

    recall_p: dict[str, float] = {}

    for entry_id in censoring.frame_blind:
        recall_p[entry_id] = 0.0

    for entry_id in list(censoring.measurable) + list(censoring.classifier_blind):
        if calibration is not None:
            # Find recall posteriors for this entry across strata
            entry_betas = [
                v for (eid, _), v in calibration.recall.items() if eid == entry_id
            ]
            if entry_betas:
                # Use minimum P(recall > threshold) across strata (conservative)
                ps = [
                    1.0 - float(beta_dist.cdf(recall_threshold, b.alpha, b.beta))
                    for b in entry_betas
                ]
                recall_p[entry_id] = min(ps)
            else:
                recall_p[entry_id] = 0.0
        else:
            # No calibration — synthetic path, assume high recall
            recall_p[entry_id] = 1.0

    return MeasurabilityMap(
        verdict=dict(censoring.verdicts),
        recall_p_above_threshold=recall_p,
        coverage_ratio=coverage,
        measurable=censoring.measurable,
        classifier_blind=censoring.classifier_blind,
        frame_blind=censoring.frame_blind,
        below_prereg_minimum=len(censoring.measurable) < measurability_minimum,
    )
