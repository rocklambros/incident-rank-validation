"""K-fold cross-validation for calibration stability.

See HANDOFF §6 control 11(c). This is a transparency disclosure,
not a quality gate — high fold variance does not block the pipeline.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from engine.calibrate.beta import BetaPosterior

__all__ = ["CVResult", "cross_validate_calibration"]


@dataclass(frozen=True, slots=True)
class CVResult:
    n_folds: int
    fold_variances: dict[tuple[str, str], float]
    interpretation: dict[tuple[str, str], str]
    min_per_fold: dict[tuple[str, str], int]


def _interpret(variance: float, min_per_fold: int) -> str:
    if min_per_fold < 5:
        return "unstable — interpret with caution"
    if variance < 0.01:
        return "stable"
    if variance < 0.05:
        return "moderate"
    return "unstable — interpret with caution"


def cross_validate_calibration(
    precision_labels: dict[tuple[str, str], list[bool]],
    recall_labels: dict[tuple[str, str], list[bool]],
    n_folds: int = 5,
) -> CVResult:
    all_labels: dict[tuple[str, str], list[bool]] = {}
    for key, vals in precision_labels.items():
        all_labels.setdefault(key, []).extend(vals)
    for key, vals in recall_labels.items():
        all_labels.setdefault(key, []).extend(vals)

    fold_variances: dict[tuple[str, str], float] = {}
    interpretation: dict[tuple[str, str], str] = {}
    min_per_fold_out: dict[tuple[str, str], int] = {}

    for key, labels in all_labels.items():
        n = len(labels)
        if n == 0:
            continue

        shuffled = list(labels)
        random.Random(0).shuffle(shuffled)
        labels = shuffled

        fold_size = n // n_folds
        remainder = n % n_folds
        fold_means: list[float] = []

        start = 0
        min_fold_n = n
        for i in range(n_folds):
            end = start + fold_size + (1 if i < remainder else 0)
            fold = labels[start:end]
            if len(fold) == 0:
                continue
            min_fold_n = min(min_fold_n, len(fold))
            successes = sum(fold)
            failures = len(fold) - successes
            bp = BetaPosterior.from_counts(successes, failures)
            fold_means.append(bp.mean)
            start = end

        var = 0.0 if len(fold_means) < 2 else statistics.variance(fold_means)

        fold_variances[key] = var
        min_per_fold_out[key] = min_fold_n
        interpretation[key] = _interpret(var, min_fold_n)

    return CVResult(
        n_folds=n_folds,
        fold_variances=fold_variances,
        interpretation=interpretation,
        min_per_fold=min_per_fold_out,
    )
