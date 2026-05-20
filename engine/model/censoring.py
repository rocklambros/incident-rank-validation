"""Frame-blind censoring: partition taxonomy entries by measurability.

Per HANDOFF §5.4: "Frame-blind entries are censored as unmeasurable and
excluded from the ranked posterior, listed explicitly in the measurability
map."

Per HANDOFF §3: "An entry without demonstrable frame-coverage is reported as
unmeasurable.  It is not assigned a low prevalence and it is not flagged."

Per HANDOFF §6 control 7: "Frame-coverage honesty: frame-blind entries are
reported unmeasurable and never flagged."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from engine.calibrate.beta import Calibration
from engine.schema import EntryDefinition

__all__ = [
    "MeasurabilityVerdict",
    "CensoringResult",
    "partition_entries",
]


class MeasurabilityVerdict(Enum):
    FRAME_BLIND_UNMEASURABLE = "frame_blind_unmeasurable"
    CLASSIFIER_BLIND_BOUNDED = "classifier_blind_bounded"
    MEASURABLE = "measurable"


@dataclass(frozen=True, slots=True)
class CensoringResult:
    """Partition of entries by measurability."""

    measurable: tuple[str, ...]        # entry_ids
    classifier_blind: tuple[str, ...]  # entry_ids — recall too low to rank but bounded
    frame_blind: tuple[str, ...]       # entry_ids — invisible to corpus frame
    verdicts: dict[str, MeasurabilityVerdict]


def partition_entries(
    entries: tuple[EntryDefinition, ...],
    calibration: Calibration | None,
    recall_floor: float = 0.1,
) -> CensoringResult:
    """Partition entries into measurability categories.

    Args:
        entries: all taxonomy entries
        calibration: gold-set calibration (None for synthetic pre-calibration)
        recall_floor: minimum recall mean across strata for "measurable" verdict.
            Below this, entry is classifier-blind (bounded but not rankable).

    Returns:
        CensoringResult with disjoint partitions.
    """
    measurable: list[str] = []
    classifier_blind: list[str] = []
    frame_blind: list[str] = []
    verdicts: dict[str, MeasurabilityVerdict] = {}

    for entry in entries:
        if entry.frame_blind:
            frame_blind.append(entry.entry_id)
            verdicts[entry.entry_id] = MeasurabilityVerdict.FRAME_BLIND_UNMEASURABLE
        elif calibration is not None:
            # Check recall across strata for this entry
            entry_recalls = [
                v.mean
                for (eid, _stratum), v in calibration.recall.items()
                if eid == entry.entry_id
            ]
            if entry_recalls and all(r >= recall_floor for r in entry_recalls):
                measurable.append(entry.entry_id)
                verdicts[entry.entry_id] = MeasurabilityVerdict.MEASURABLE
            else:
                classifier_blind.append(entry.entry_id)
                verdicts[entry.entry_id] = MeasurabilityVerdict.CLASSIFIER_BLIND_BOUNDED
        else:
            # No calibration yet — default non-frame-blind entries to measurable
            # (synthetic cycle pre-calibration path)
            measurable.append(entry.entry_id)
            verdicts[entry.entry_id] = MeasurabilityVerdict.MEASURABLE

    return CensoringResult(
        measurable=tuple(measurable),
        classifier_blind=tuple(classifier_blind),
        frame_blind=tuple(frame_blind),
        verdicts=verdicts,
    )
