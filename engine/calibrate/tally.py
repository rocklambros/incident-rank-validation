"""Tally aggregation: count coded labels into per-entry per-stratum tallies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.calibrate.batch import CodingBatch, ValidationError, validate_coded_batch
from engine.calibrate.gold_schema import OUT_OF_SCOPE, GoldCalibration


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
    # Recall posteriors derive SOLELY from gold truth-vs-prediction (per-entry
    # truth-cell denominators). The recall-frame tally has no classifier
    # prediction, so its frame-size-padded recall counts are not real recall and
    # are intentionally dropped here. (Plan 8a, SD1/RM2.)
    recall_counts: dict[tuple[str, str], RecallTally] = {}
    rollup_counts = dict(base_tally.rollup_counts)
    gold_coded = 0

    recall_tp: dict[tuple[str, str], int] = {}
    recall_fn: dict[tuple[str, str], int] = {}
    recall_total: dict[tuple[str, str], int] = {}
    precision_tp: dict[tuple[str, str], int] = {}
    precision_fp: dict[tuple[str, str], int] = {}
    precision_total: dict[tuple[str, str], int] = {}

    # F-A: a claim that already carries an EXPLICIT precision verdict (a
    # GoldPrecisionLabel — e.g. every adjudicated in-scope claim on a truth-
    # labeled incident) is scored for precision via that label below; deriving a
    # second precision FP from the recall label would double-count.  Key on
    # (incident_id, claimed_entry) — NOT incident_id alone — so the recall-
    # derived FP is suppressed only for the SAME claim, never for a different
    # entry an incident might also be misclassified as.  The recall-derived FP
    # still fires for claims with no explicit precision verdict (the curation /
    # F4 path, and in-scope claims on truth-OOS incidents).
    precision_label_keys = {
        (p.incident_id, p.claimed_entry_id) for p in gold.precision_labels
    }

    for label in gold.recall_labels:
        if label.incident_id in base_incident_ids:
            continue
        gold_coded += 1

        if label.classifier_entry_id is None:
            continue

        # Single-label recall semantics (INTENTIONAL — Plan 8ab-remediation F4).
        # The classifier emits ONE entry_id per incident, but truth may be
        # multi-label. Recall here is the *detection rate* the measurement-error
        # model consumes (true_count ~= observed_count / recall): for entry X it
        # is "of incidents truly X, what fraction did the classifier label X?".
        # An incident truly {A, B} that the classifier labels A is in A's observed
        # count, NOT B's, so it is a genuine detection MISS for B (recall FN for
        # B), even though A is a true label. Crediting B as a hit here would
        # inflate B's recall and under-correct B's incidence for incidents absent
        # from B's observed count. Co-occurring entries therefore get inherently
        # lower recall — a real single-label-classifier limitation, honestly
        # reflected, not a bug. (Pinned by tests/unit/test_recall_single_label_semantics.py.)
        for true_eid in label.true_entry_ids:
            rk = (true_eid, merge_stratum)
            recall_total[rk] = recall_total.get(rk, 0) + 1

            if label.classifier_entry_id == true_eid:
                recall_tp[rk] = recall_tp.get(rk, 0) + 1
            else:
                recall_fn[rk] = recall_fn.get(rk, 0) + 1

        if (
            label.classifier_entry_id not in label.true_entry_ids
            and label.classifier_entry_id != OUT_OF_SCOPE
            and (label.incident_id, label.classifier_entry_id)
            not in precision_label_keys
        ):
            pk = (label.classifier_entry_id, merge_stratum)
            precision_fp[pk] = precision_fp.get(pk, 0) + 1
            precision_total[pk] = precision_total.get(pk, 0) + 1

    for plabel in gold.precision_labels:
        pk = (plabel.claimed_entry_id, merge_stratum)
        precision_total[pk] = precision_total.get(pk, 0) + 1
        if plabel.is_correct:
            precision_tp[pk] = precision_tp.get(pk, 0) + 1
        else:
            precision_fp[pk] = precision_fp.get(pk, 0) + 1

    for k in recall_total:
        recall_counts[k] = RecallTally(
            true_positives=recall_tp.get(k, 0),
            false_negatives=recall_fn.get(k, 0),
            total_in_sample=recall_total[k],
        )

    for k in precision_total:
        existing_p = precision_counts.get(k)
        if existing_p:
            precision_counts[k] = PrecisionTally(
                true_positives=existing_p.true_positives + precision_tp.get(k, 0),
                false_positives=existing_p.false_positives + precision_fp.get(k, 0),
                total=existing_p.total + precision_total[k],
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
