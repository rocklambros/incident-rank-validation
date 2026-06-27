"""Classifier bake-off scoring + selection harness (Plan 8e, RARR spec §5.2).

Pure, deterministic, GPU-free.  Evaluates a config's predictions against the
adjudicated goldset on a held-back lockbox split via OOS-inclusive balanced
accuracy, controls multiplicity with Benjamini-Hochberg, excludes sparse truth
cells from the SELECTION metric only, and picks the winner that beats a
reproducible floor.  No live model or RunPod call lives here.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

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


def two_proportion_pvalue(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided pooled z-test for a difference in two proportions."""
    if n_a <= 0 or n_b <= 0:
        return 1.0
    p_a = hits_a / n_a
    p_b = hits_b / n_b
    p_pool = (hits_a + hits_b) / (n_a + n_b)
    var = p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b)
    if var <= 0.0:
        return 1.0
    z = (p_a - p_b) / (var**0.5)
    return float(2.0 * norm.sf(abs(z)))


def benjamini_hochberg(pvalues: list[float], alpha: float) -> list[bool]:
    """BH step-up procedure; returns a rejection mask in the input order."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    k_max = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= (rank / m) * alpha:
            k_max = rank
    rejected = [False] * m
    if k_max > 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= k_max:
                rejected[idx] = True
    return rejected


BAKEOFF_ALPHA: float = 0.05
MIN_CELL: int = 5


@dataclass(frozen=True, slots=True)
class BakeoffResult:
    winner: str | None
    floor_balanced_accuracy: float
    config_balanced_accuracy: dict[str, float]
    selection_classes: tuple[str, ...]
    sparse_classes: tuple[str, ...]
    lockbox_cell_sizes: dict[str, int]
    eligible_configs: tuple[str, ...]
    alpha: float


def _restrict(
    predictions: Mapping[str, str], lockbox_ids: frozenset[str]
) -> dict[str, str]:
    return {k: v for k, v in predictions.items() if k in lockbox_ids}


def _class_hits(
    predictions: Mapping[str, str], truth: Mapping[str, frozenset[str]], c: str
) -> tuple[int, int]:
    """(hits, denom) for class c over the given predictions."""
    hits = 0
    denom = 0
    for incident_id, pred in predictions.items():
        true_classes = truth.get(incident_id)
        if true_classes is None or c not in true_classes:
            continue
        denom += 1
        if pred == c:
            hits += 1
    return hits, denom


def select_winner(
    config_predictions: Mapping[str, Mapping[str, str]],
    floor_predictions: Mapping[str, str],
    truth: Mapping[str, frozenset[str]],
    lockbox_ids: frozenset[str],
    alpha: float = BAKEOFF_ALPHA,
    min_cell: int = MIN_CELL,
) -> BakeoffResult:
    """Pick the config with the highest OOS-balanced-accuracy that beats the
    floor after BH correction; sparse truth cells excluded from the metric."""
    sparse = sparse_classes(truth, min_n=min_cell)
    lb_cells = lockbox_cell_sizes(lockbox_ids, truth)
    # A non-sparse class with ZERO lockbox truth cannot be measured on the
    # lockbox; including it would feed 0/0 -> 0.0 into the balanced-accuracy
    # mean and silently drag every config and the floor.  Exclude it from the
    # selection metric (it is still carried downstream, like sparse classes).
    selection = tuple(
        sorted(
            c
            for c in truth_cell_sizes(truth)
            if c not in sparse and lb_cells.get(c, 0) > 0
        )
    )

    floor_lb = _restrict(floor_predictions, lockbox_ids)
    floor_ba = balanced_accuracy_oos(floor_lb, truth, selection)

    config_ba: dict[str, float] = {}
    config_lb: dict[str, dict[str, str]] = {}
    for name, preds in config_predictions.items():
        lb = _restrict(preds, lockbox_ids)
        config_lb[name] = lb
        config_ba[name] = balanced_accuracy_oos(lb, truth, selection)

    # F5: predictions must use the goldset's class vocabulary, else they are
    # silently scored as all-misses (a wrong-winner footgun).
    # INVARIANT: this guard must stay BEFORE the winner is constructed/returned;
    # moving it below the BH block would let a vocabulary mismatch silently
    # produce a wrong winner instead of raising.
    allowed_classes = set(truth_cell_sizes(truth))
    floor_unknown = {c for c in floor_lb.values() if c not in allowed_classes}
    if floor_unknown:
        raise ValueError(
            f"floor predicts classes absent from the goldset vocabulary: "
            f"{sorted(floor_unknown)}"
        )
    for _name, _lb in config_lb.items():
        unknown = {c for c in _lb.values() if c not in allowed_classes}
        if unknown:
            raise ValueError(
                f"config {_name!r} predicts classes absent from the goldset "
                f"vocabulary: {sorted(unknown)}"
            )

    # Per-(config, class) two-proportion p-values vs floor, then BH across all.
    keys: list[tuple[str, str]] = []
    pvals: list[float] = []
    directions: list[bool] = []  # True = config recall > floor recall
    for name in sorted(config_lb):
        for c in selection:
            ch, cn = _class_hits(config_lb[name], truth, c)
            fh, fn = _class_hits(floor_lb, truth, c)
            keys.append((name, c))
            pvals.append(two_proportion_pvalue(ch, cn, fh, fn))
            directions.append((ch / cn if cn else 0.0) > (fh / fn if fn else 0.0))
    rejected = benjamini_hochberg(pvals, alpha)

    improved: dict[str, bool] = {name: False for name in config_lb}
    regressed: dict[str, bool] = {name: False for name in config_lb}
    for (name, _c), rej, up in zip(keys, rejected, directions, strict=True):
        if rej and up:
            improved[name] = True
        if rej and not up:
            regressed[name] = True

    eligible = tuple(
        sorted(
            name
            for name in config_lb
            if config_ba[name] > floor_ba and improved[name] and not regressed[name]
        )
    )
    winner = max(eligible, key=lambda n: config_ba[n]) if eligible else None

    return BakeoffResult(
        winner=winner,
        floor_balanced_accuracy=floor_ba,
        config_balanced_accuracy=config_ba,
        selection_classes=selection,
        sparse_classes=tuple(sorted(sparse)),
        lockbox_cell_sizes=lockbox_cell_sizes(lockbox_ids, truth),
        eligible_configs=eligible,
        alpha=alpha,
    )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    model_id: str
    revision_sha: str  # pinned HF commit SHA (spec §10)
    gpu_type: str
    gpu_count: int


def write_bakeoff_provenance(
    out_dir: Path,
    result: BakeoffResult,
    model_configs: Iterable[ModelConfig],
    label_file: Path,
    seed: int | None = None,
    lockbox_fraction: float | None = None,
    min_cell: int | None = None,
    goldset_meta: dict[str, object] | None = None,
) -> Path:
    """Write classify_provenance.json: label-file hash + resolved model SHAs +
    grid + winner/scores.  Returns the written path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_sha = hashlib.sha256(label_file.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "winner": result.winner,
        "floor_balanced_accuracy": result.floor_balanced_accuracy,
        "config_balanced_accuracy": result.config_balanced_accuracy,
        "selection_classes": list(result.selection_classes),
        "sparse_classes": list(result.sparse_classes),
        "lockbox_cell_sizes": result.lockbox_cell_sizes,
        "eligible_configs": list(result.eligible_configs),
        "alpha": result.alpha,
        "seed": seed,
        "lockbox_fraction": lockbox_fraction,
        "min_cell": min_cell,
        "goldset": goldset_meta,
        "label_file": str(label_file.name),
        "label_file_sha256": label_sha,
        "models": [
            {
                "name": m.name,
                "model_id": m.model_id,
                "revision_sha": m.revision_sha,
                "gpu_type": m.gpu_type,
                "gpu_count": m.gpu_count,
            }
            for m in model_configs
        ],
    }
    path = out_dir / "classify_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def goldset_provenance(goldset_path: Path) -> dict[str, object]:
    """Audit metadata for the goldset truth file (Plan 8e F3a/F9).

    Records the content hash, record count, the blind-label vs llm-consensus
    disagreement rate (a single-author truth-uncertainty signal), and the
    adjudication breakdown.  Makes the truth the winner is selected against
    auditable and bindable in the lock.
    """
    raw = goldset_path.read_bytes()
    n = 0
    blind_disagree = 0
    adjudicated: dict[str, int] = {}
    for line in raw.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n += 1
        if rec.get("blind_label") != rec.get("llm_consensus"):
            blind_disagree += 1
        adj = str(rec.get("adjudicated", ""))
        adjudicated[adj] = adjudicated.get(adj, 0) + 1
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_records": n,
        "blind_consensus_disagreement_rate": (blind_disagree / n) if n else 0.0,
        "adjudicated_counts": adjudicated,
        "adjudicator": "single-author",
    }


def goldset_corpus_divergence(
    goldset_truth: Mapping[str, frozenset[str]],
    corpus_class_counts: Mapping[str, int],
) -> float:
    """Total-variation distance between the goldset and corpus class mixes.

    0 = identical mix, 1 = disjoint.  A high value means the goldset the winner
    is SELECTED on is not representative of the corpus it is APPLIED to (F8).
    """
    gold_counts = truth_cell_sizes(goldset_truth)
    gold_total = sum(gold_counts.values())
    corpus_total = sum(corpus_class_counts.values())
    if gold_total == 0 or corpus_total == 0:
        return 0.0
    classes = set(gold_counts) | set(corpus_class_counts)
    tv = 0.0
    for c in classes:
        p = gold_counts.get(c, 0) / gold_total
        q = corpus_class_counts.get(c, 0) / corpus_total
        tv += abs(p - q)
    return 0.5 * tv
