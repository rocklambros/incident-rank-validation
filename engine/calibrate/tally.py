"""Tally aggregation: count coded labels into per-entry per-stratum tallies."""
from __future__ import annotations

from dataclasses import dataclass

from engine.calibrate.batch import CodingBatch


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


def tally_batches(batches: list[CodingBatch]) -> TallyResult:
    precision_tp: dict[tuple[str, str], int] = {}
    precision_fp: dict[tuple[str, str], int] = {}
    precision_total: dict[tuple[str, str], int] = {}
    recall_hits: dict[tuple[str, str], int] = {}
    recall_miss: dict[tuple[str, str], int] = {}
    recall_total: dict[tuple[str, str], int] = {}
    rollup_tp: dict[tuple[str, str], int] = {}
    rollup_fp: dict[tuple[str, str], int] = {}
    rollup_total: dict[tuple[str, str], int] = {}
    total_coded = 0
    amendments = 0

    all_recall_entries: set[str] = set()

    for batch in batches:
        entry_id = batch.header.entry_id
        stratum = batch.header.stratum or "unknown"
        frame = batch.header.frame

        if frame == "precision" and entry_id is not None:
            key = (entry_id, stratum)
            for inc in batch.incidents:
                if inc.labels is None:
                    continue
                total_coded += 1
                precision_total[key] = precision_total.get(key, 0) + 1
                if entry_id in inc.labels:
                    precision_tp[key] = precision_tp.get(key, 0) + 1
                else:
                    precision_fp[key] = precision_fp.get(key, 0) + 1
                if inc.rollup_sub_labels:
                    for rl in inc.rollup_sub_labels:
                        rk = (rl, stratum)
                        rollup_total[rk] = rollup_total.get(rk, 0) + 1
                        rollup_tp[rk] = rollup_tp.get(rk, 0) + 1
                if inc.amendment:
                    amendments += 1

        elif frame == "recall":
            for inc in batch.incidents:
                if inc.labels is None:
                    continue
                total_coded += 1
                labels_set = set(inc.labels)
                all_recall_entries.update(labels_set)
                for eid in labels_set:
                    rk = (eid, stratum)
                    recall_hits[rk] = recall_hits.get(rk, 0) + 1
                if inc.amendment:
                    amendments += 1

    for batch in batches:
        if batch.header.frame != "recall":
            continue
        stratum = batch.header.stratum or "unknown"
        coded_count = sum(1 for inc in batch.incidents if inc.labels is not None)
        for eid in all_recall_entries:
            rk = (eid, stratum)
            hits = recall_hits.get(rk, 0)
            recall_total[rk] = recall_total.get(rk, 0) + coded_count
            recall_miss[rk] = recall_miss.get(rk, 0) + (coded_count - hits)

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
            false_negatives=recall_miss.get(k, 0),
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
