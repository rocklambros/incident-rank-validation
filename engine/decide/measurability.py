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
from engine.calibrate.calibrate import CalibrationDiagnostic
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
    # F6 recall-cell flags (U2-2, additive, default empty — schema<3 cycles unaffected).
    thin_denominator_entries: tuple[str, ...] = ()
    under_detected_entries: tuple[str, ...] = ()

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
                # F6 additive keys (U2-2): always present (empty for schema<3 cycles).
                "thin_denominator_entries": sorted(self.thin_denominator_entries),
                "under_detected_entries": sorted(self.under_detected_entries),
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
    calibration_diagnostic: CalibrationDiagnostic | None = None,
    recall_min_denominator_gate: bool = False,
) -> MeasurabilityMap:
    """Build the measurability map from censoring + calibration.

    For each measurable/classifier-blind entry, compute P(recall > threshold)
    using the exact Beta CDF. Frame-blind entries get P=0.0.

    Args:
        censoring: partition result from engine.model.censoring.partition_entries
        calibration: Beta posteriors (None on the synthetic pre-calibration path)
        measurability_minimum: minimum measurable count required by pre-registration
        recall_threshold: the recall level whose exceedance probability is computed
        calibration_diagnostic: optional CalibrationDiagnostic carrying F6 thin/under_detected
            flags (U2-2).  When provided, the flags are persisted into coverage.json.
        recall_min_denominator_gate: when True AND calibration_diagnostic is provided,
            entries flagged as thin_denominator are hard-excluded from the measurable
            tuple (moved to classifier_blind).  Default False = keep-but-flag (no
            headline impact).

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

    # F6 flags (U2-2): collect thin/under_detected from diagnostic if available.
    _thin: tuple[str, ...] = ()
    _under: tuple[str, ...] = ()
    _measurable = censoring.measurable
    _classifier_blind = censoring.classifier_blind
    if calibration_diagnostic is not None:
        _thin = tuple(
            eid
            for eid, r in calibration_diagnostic.entry_reports.items()
            if r.thin_denominator
        )
        _under = tuple(
            eid
            for eid, r in calibration_diagnostic.entry_reports.items()
            if r.under_detected
        )
        # Gate: hard-exclude thin-flagged entries from measurable headline.
        # Only when recall_min_denominator_gate=True (default False = keep-but-flag).
        if recall_min_denominator_gate and _thin:
            _thin_set = set(_thin)
            _measurable = tuple(e for e in _measurable if e not in _thin_set)
            _classifier_blind = tuple(
                list(_classifier_blind)
                + [e for e in censoring.measurable if e in _thin_set]
            )

    return MeasurabilityMap(
        verdict=dict(censoring.verdicts),
        recall_p_above_threshold=recall_p,
        coverage_ratio=coverage,
        measurable=_measurable,
        classifier_blind=_classifier_blind,
        frame_blind=censoring.frame_blind,
        below_prereg_minimum=len(_measurable) < measurability_minimum,
        thin_denominator_entries=_thin,
        under_detected_entries=_under,
    )
