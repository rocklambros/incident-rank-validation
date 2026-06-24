"""Build the FP-leakage overlap matrix W from goldset confusion (Plan 8a, RM12)."""
from __future__ import annotations

from engine.calibrate.gold_schema import GoldCalibration
from engine.model.overlap import OverlapWeights


def build_overlap_from_confusion(
    gold: GoldCalibration,
    measurable_entries: tuple[str, ...],
    min_fp_count: int = 1,
) -> OverlapWeights:
    entries = set(measurable_entries)
    # fp_counts[source][target] = # incidents predicted `source` but truly `target`
    fp_counts: dict[str, dict[str, int]] = {}
    for label in gold.recall_labels:
        pred = label.classifier_entry_id
        if pred is None or pred not in entries:
            continue
        for true_eid in label.true_entry_ids:
            if true_eid == pred or true_eid not in entries:
                continue
            fp_counts.setdefault(pred, {}).setdefault(true_eid, 0)
            fp_counts[pred][true_eid] += 1

    weights: dict[str, dict[str, float]] = {}
    for source, targets in fp_counts.items():
        total = sum(targets.values())
        if total < min_fp_count:
            continue  # insufficient evidence of leakage from this source (premortem F5)
        for target, n in targets.items():
            weights.setdefault(target, {})[source] = n / total
    return OverlapWeights(weights=weights)
