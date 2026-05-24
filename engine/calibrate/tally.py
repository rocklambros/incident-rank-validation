"""Tally aggregation: count coded labels into per-entry per-stratum tallies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.calibrate.batch import CodingBatch, ValidationError, validate_coded_batch
from engine.calibrate.gold_schema import GoldCalibration


@dataclass(frozen=True, slots=True)
class PrecisionTally:
    true_positives: int
    false_positives: int
    total: int


@dataclass(frozen=True, slots=True)
class RecallTally:
    true_positives: int
    false_negatives: int
    total_in_sample: int


@dataclass(frozen=True, slots=True)
class TallyResult:
    precision_counts: dict[tuple[str, str], PrecisionTally]
    recall_counts: dict[tuple[str, str], RecallTally]
    rollup_counts: dict[tuple[str, str], PrecisionTally]
    total_coded: int
    amendments_applied: int


def tally_batches(
    batches: list[CodingBatch],
    *,
    all_entry_ids: set[str] | None = None,
    rollup_children: dict[str, set[str]] | None = None,
) -> TallyResult:
    """Aggregate coded labels into per-entry per-stratum tallies.

    Parameters
    ----------
    all_entry_ids:
        Explicit set of ALL entry IDs to compute recall for. If None,
        falls back to discovering entries from labels (legacy behavior).
    rollup_children:
        Mapping of parent_entry_id -> set of rollup child IDs. Used to
        count rollup FPs: incidents in the parent's precision frame where
        the coder did NOT assign a rollup child count as FPs for that child.
    """
    precision_tp: dict[tuple[str, str], int] = {}
    precision_fp: dict[tuple[str, str], int] = {}
    precision_total: dict[tuple[str, str], int] = {}
    recall_hits: dict[tuple[str, str], int] = {}
    recall_total: dict[tuple[str, str], int] = {}
    rollup_tp: dict[tuple[str, str], int] = {}
    rollup_total: dict[tuple[str, str], int] = {}
    total_coded = 0
    amendments = 0

    discovered_recall_entries: set[str] = set()

    for batch in batches:
        entry_id = batch.header.entry_id
        stratum = batch.header.stratum or "unknown"
        frame = batch.header.frame

        if frame == "precision" and entry_id is not None:
            key = (entry_id, stratum)
            children = (
                (rollup_children or {}).get(entry_id, set())
            )
            for inc in batch.incidents:
                if inc.labels is None:
                    continue
                total_coded += 1
                precision_total[key] = precision_total.get(key, 0) + 1
                if entry_id in inc.labels:
                    precision_tp[key] = precision_tp.get(key, 0) + 1
                else:
                    precision_fp[key] = precision_fp.get(key, 0) + 1
                assigned_rollups = set(inc.rollup_sub_labels or [])
                for rl in assigned_rollups:
                    rk = (rl, stratum)
                    rollup_total[rk] = rollup_total.get(rk, 0) + 1
                    rollup_tp[rk] = rollup_tp.get(rk, 0) + 1
                for child in children - assigned_rollups:
                    rk = (child, stratum)
                    rollup_total[rk] = rollup_total.get(rk, 0) + 1
                if inc.amendment:
                    amendments += 1

        elif frame == "recall":
            for inc in batch.incidents:
                if inc.labels is None:
                    continue
                total_coded += 1
                labels_set = set(inc.labels)
                discovered_recall_entries.update(labels_set)
                for eid in labels_set:
                    rk = (eid, stratum)
                    recall_hits[rk] = recall_hits.get(rk, 0) + 1
                if inc.amendment:
                    amendments += 1

    recall_entry_ids = all_entry_ids if all_entry_ids is not None else discovered_recall_entries

    for batch in batches:
        if batch.header.frame != "recall":
            continue
        stratum = batch.header.stratum or "unknown"
        coded_count = sum(1 for inc in batch.incidents if inc.labels is not None)
        for eid in recall_entry_ids:
            rk = (eid, stratum)
            recall_total[rk] = recall_total.get(rk, 0) + coded_count

    precision_counts = {
        k: PrecisionTally(
            true_positives=precision_tp.get(k, 0),
            false_positives=precision_fp.get(k, 0),
            total=precision_total[k],
        )
        for k in precision_total
    }

    recall_counts = {
        k: RecallTally(
            true_positives=recall_hits.get(k, 0),
            false_negatives=recall_total.get(k, 0) - recall_hits.get(k, 0),
            total_in_sample=recall_total.get(k, 0),
        )
        for k in recall_total
    }

    rollup_counts_out = {
        k: PrecisionTally(
            true_positives=rollup_tp.get(k, 0),
            false_positives=rollup_total.get(k, 0) - rollup_tp.get(k, 0),
            total=rollup_total[k],
        )
        for k in rollup_total
    }

    return TallyResult(
        precision_counts=precision_counts,
        recall_counts=recall_counts,
        rollup_counts=rollup_counts_out,
        total_coded=total_coded,
        amendments_applied=amendments,
    )


def validate_and_tally(
    batch_paths: list[Path],
    *,
    valid_entry_ids: set[str],
    rollup_entry_ids: set[str],
    expected_sample_hashes: dict[str, str],
    expected_rubric_hash: str,
    expected_lock_hash: str,
    all_entry_ids: set[str] | None = None,
    rollup_children: dict[str, set[str]] | None = None,
    expected_incident_ids: set[str] | None = None,
) -> TallyResult:
    """Validate all coded batch files, then tally.

    Raises ValueError if any validation errors are found.
    """
    all_errors: list[ValidationError] = []
    batches: list[CodingBatch] = []

    for path in batch_paths:
        batch = CodingBatch.read(path)
        batch_id = batch.header.batch_id
        expected_hash = expected_sample_hashes.get(
            batch_id, batch.header.sample_hash,
        )
        errors = validate_coded_batch(
            path,
            valid_entry_ids=valid_entry_ids,
            rollup_entry_ids=rollup_entry_ids,
            expected_sample_hash=expected_hash,
            expected_rubric_hash=expected_rubric_hash,
            expected_lock_hash=expected_lock_hash,
            expected_incident_ids=expected_incident_ids,
        )
        hard_errors = [e for e in errors if "uncoded" not in e.message]
        all_errors.extend(hard_errors)
        batches.append(batch)

    if all_errors:
        msg = "\n".join(str(e) for e in all_errors)
        raise ValueError(f"Batch validation failed:\n{msg}")

    return tally_batches(
        batches,
        all_entry_ids=all_entry_ids,
        rollup_children=rollup_children,
    )


def calibrate_with_gold(
    base_tally: TallyResult,
    gold: GoldCalibration,
    base_incident_ids: set[str],
    all_entry_ids: set[str],
    merge_stratum: str = "security",
) -> TallyResult:
    """Merge gold calibration labels into an existing tally.

    Gold data is keyed under ``merge_stratum`` so that
    ``_build_observation_arrays`` picks it up when iterating corpus strata.
    """
    precision_counts = dict(base_tally.precision_counts)
    recall_counts = dict(base_tally.recall_counts)
    rollup_counts = dict(base_tally.rollup_counts)
    gold_coded = 0

    recall_tp: dict[tuple[str, str], int] = {}
    recall_fn: dict[tuple[str, str], int] = {}
    recall_total: dict[tuple[str, str], int] = {}
    precision_tp: dict[tuple[str, str], int] = {}
    precision_fp: dict[tuple[str, str], int] = {}
    precision_total: dict[tuple[str, str], int] = {}

    for label in gold.recall_labels:
        if label.incident_id in base_incident_ids:
            continue
        gold_coded += 1

        if label.classifier_entry_id is None:
            continue

        for true_eid in label.true_entry_ids:
            rk = (true_eid, merge_stratum)
            recall_total[rk] = recall_total.get(rk, 0) + 1

            if label.classifier_entry_id == true_eid:
                recall_tp[rk] = recall_tp.get(rk, 0) + 1
            else:
                recall_fn[rk] = recall_fn.get(rk, 0) + 1

        if label.classifier_entry_id not in label.true_entry_ids:
            pk = (label.classifier_entry_id, merge_stratum)
            precision_fp[pk] = precision_fp.get(pk, 0) + 1
            precision_total[pk] = precision_total.get(pk, 0) + 1

    for label in gold.precision_labels:
        pk = (label.claimed_entry_id, merge_stratum)
        precision_total[pk] = precision_total.get(pk, 0) + 1
        if label.is_correct:
            precision_tp[pk] = precision_tp.get(pk, 0) + 1
        else:
            precision_fp[pk] = precision_fp.get(pk, 0) + 1

    for k in recall_total:
        existing = recall_counts.get(k)
        if existing:
            recall_counts[k] = RecallTally(
                true_positives=existing.true_positives + recall_tp.get(k, 0),
                false_negatives=existing.false_negatives + recall_fn.get(k, 0),
                total_in_sample=existing.total_in_sample + recall_total[k],
            )
        else:
            recall_counts[k] = RecallTally(
                true_positives=recall_tp.get(k, 0),
                false_negatives=recall_fn.get(k, 0),
                total_in_sample=recall_total[k],
            )

    for k in precision_total:
        existing = precision_counts.get(k)
        if existing:
            precision_counts[k] = PrecisionTally(
                true_positives=existing.true_positives + precision_tp.get(k, 0),
                false_positives=existing.false_positives + precision_fp.get(k, 0),
                total=existing.total + precision_total[k],
            )
        else:
            precision_counts[k] = PrecisionTally(
                true_positives=precision_tp.get(k, 0),
                false_positives=precision_fp.get(k, 0),
                total=precision_total[k],
            )

    return TallyResult(
        precision_counts=precision_counts,
        recall_counts=recall_counts,
        rollup_counts=rollup_counts,
        total_coded=base_tally.total_coded + gold_coded,
        amendments_applied=base_tally.amendments_applied,
    )
