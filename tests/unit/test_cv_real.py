"""Tests for engine.calibrate.cv — real k-fold cross-validation."""
from __future__ import annotations

from engine.calibrate.cv import cross_validate_calibration


class TestCrossValidateCalibration:
    def test_stable_labels(self) -> None:
        precision = {("LLM01", "security"): [True] * 35 + [False] * 5}
        recall = {("LLM01", "security"): [True] * 30 + [False] * 10}
        result = cross_validate_calibration(precision, recall, n_folds=5)
        assert result.n_folds == 5
        assert ("LLM01", "security") in result.fold_variances
        assert result.interpretation[("LLM01", "security")] == "stable"

    def test_unstable_small_sample(self) -> None:
        precision = {("LLM01", "security"): [True, False, True]}
        recall: dict[tuple[str, str], list[bool]] = {}
        result = cross_validate_calibration(precision, recall, n_folds=5)
        assert result.min_per_fold[("LLM01", "security")] < 5
        assert "unstable" in result.interpretation[("LLM01", "security")]

    def test_empty_labels(self) -> None:
        result = cross_validate_calibration({}, {}, n_folds=5)
        assert result.n_folds == 5
        assert result.fold_variances == {}

    def test_interpretation_thresholds(self) -> None:
        precision = {("LLM01", "security"): [True] * 100 + [False] * 100}
        result = cross_validate_calibration(precision, {}, n_folds=5)
        interp = result.interpretation[("LLM01", "security")]
        assert interp in ("stable", "moderate", "unstable — interpret with caution")
