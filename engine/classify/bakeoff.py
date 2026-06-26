"""Classifier bake-off scoring + selection harness (Plan 8e, RARR spec §5.2).

Pure, deterministic, GPU-free.  Evaluates a config's predictions against the
adjudicated goldset on a held-back lockbox split via OOS-inclusive balanced
accuracy, controls multiplicity with Benjamini-Hochberg, excludes sparse truth
cells from the SELECTION metric only, and picks the winner that beats a
reproducible floor.  No live model or RunPod call lives here.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

OOS_CLASS: str = "out-of-scope"


def load_bakeoff_truth(goldset_path: Path) -> dict[str, frozenset[str]]:
    """Map incident_id -> set of true classes ({OOS_CLASS} for OOS/empty)."""
    truth: dict[str, frozenset[str]] = {}
    for line in goldset_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        incident_id = str(rec["incident_id"])
        labels = [str(x) for x in rec.get("labels", [])]
        if str(rec.get("llm_consensus", "")) == OOS_CLASS or not labels:
            truth[incident_id] = frozenset({OOS_CLASS})
        else:
            truth[incident_id] = frozenset(labels)
    return truth


def truth_cell_sizes(truth: Mapping[str, frozenset[str]]) -> dict[str, int]:
    """Per-class count of incidents whose truth set includes the class."""
    sizes: dict[str, int] = {}
    for classes in truth.values():
        for c in classes:
            sizes[c] = sizes.get(c, 0) + 1
    return sizes


def sparse_classes(
    truth: Mapping[str, frozenset[str]], min_n: int = 5
) -> frozenset[str]:
    """Classes with truth cell size < min_n (excluded from selection metric)."""
    return frozenset(c for c, n in truth_cell_sizes(truth).items() if n < min_n)


def per_class_recall(
    predictions: Mapping[str, str],
    truth: Mapping[str, frozenset[str]],
    classes: Iterable[str],
) -> dict[str, float]:
    """Recall per class over the incidents present in ``predictions``."""
    recall: dict[str, float] = {}
    for c in classes:
        denom = 0
        hits = 0
        for incident_id, pred in predictions.items():
            true_classes = truth.get(incident_id)
            if true_classes is None or c not in true_classes:
                continue
            denom += 1
            if pred == c:
                hits += 1
        recall[c] = hits / denom if denom > 0 else 0.0
    return recall


def balanced_accuracy_oos(
    predictions: Mapping[str, str],
    truth: Mapping[str, frozenset[str]],
    selection_classes: Iterable[str],
) -> float:
    """Mean per-class recall over selection_classes (includes OOS_CLASS)."""
    classes = list(selection_classes)
    if not classes:
        return 0.0
    recall = per_class_recall(predictions, truth, classes)
    return sum(recall[c] for c in classes) / len(classes)


LOCKBOX_FRACTION: float = 0.3


def _primary_class(classes: frozenset[str]) -> str:
    """Deterministic single stratification key for a (possibly multi-) truth set."""
    return sorted(classes)[0]


def lockbox_split(
    truth: Mapping[str, frozenset[str]],
    lockbox_fraction: float = LOCKBOX_FRACTION,
    seed: int = 42,
) -> tuple[frozenset[str], frozenset[str]]:
    """Stratified, seeded held-back split: (dev_ids, lockbox_ids)."""
    by_class: dict[str, list[str]] = {}
    for incident_id in sorted(truth):  # sorted -> deterministic base order
        by_class.setdefault(_primary_class(truth[incident_id]), []).append(incident_id)

    rng = np.random.default_rng(seed)
    lockbox: set[str] = set()
    for cls in sorted(by_class):
        ids = list(by_class[cls])
        n_lock = int(round(len(ids) * lockbox_fraction))
        if n_lock == 0:
            continue
        perm = rng.permutation(len(ids))
        for idx in perm[:n_lock]:
            lockbox.add(ids[int(idx)])
    dev = set(truth) - lockbox
    return frozenset(dev), frozenset(lockbox)


def lockbox_cell_sizes(
    lockbox_ids: Iterable[str], truth: Mapping[str, frozenset[str]]
) -> dict[str, int]:
    """Per-class truth count within the lockbox."""
    sizes: dict[str, int] = {}
    for incident_id in lockbox_ids:
        for c in truth.get(incident_id, frozenset()):
            sizes[c] = sizes.get(c, 0) + 1
    return sizes
