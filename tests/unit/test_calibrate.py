"""Unit tests for engine.calibrate — BetaPosterior, Calibration, Sampler protocol, CV stub."""

from __future__ import annotations

import dataclasses
import math

import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.calibrate.cv import CVResult, cross_validate_calibration
from engine.calibrate.sampler import Sampler, StratifiedSampler

# ---------------------------------------------------------------------------
# BetaPosterior
# ---------------------------------------------------------------------------


class TestBetaPosterior:
    def test_from_counts_default_priors(self) -> None:
        bp = BetaPosterior.from_counts(50, 10)
        assert bp.alpha == 51.0
        assert bp.beta == 11.0

    def test_from_counts_custom_priors(self) -> None:
        bp = BetaPosterior.from_counts(10, 5, prior_alpha=2.0, prior_beta=3.0)
        assert bp.alpha == 12.0
        assert bp.beta == 8.0

    def test_mean(self) -> None:
        bp = BetaPosterior.from_counts(50, 10)
        # alpha=51, beta=11 → mean = 51/62
        expected = 51.0 / 62.0
        assert math.isclose(bp.mean, expected)

    def test_variance(self) -> None:
        bp = BetaPosterior.from_counts(50, 10)
        # alpha=51, beta=11 → var = (51*11) / (62^2 * 63)
        a, b = 51.0, 11.0
        expected = (a * b) / ((a + b) ** 2 * (a + b + 1))
        assert math.isclose(bp.variance, expected)

    def test_rejects_non_positive_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha=0"):
            BetaPosterior(alpha=0.0, beta=1.0)

    def test_rejects_non_positive_beta(self) -> None:
        with pytest.raises(ValueError, match="beta=-1"):
            BetaPosterior(alpha=1.0, beta=-1.0)

    def test_rejects_negative_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha=-0.5"):
            BetaPosterior(alpha=-0.5, beta=2.0)

    def test_is_frozen(self) -> None:
        bp = BetaPosterior(alpha=2.0, beta=3.0)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            bp.alpha = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_calibration_stores_dicts(self) -> None:
        recall = {("e1", "security"): BetaPosterior(2.0, 3.0)}
        precision = {("e1", "security"): BetaPosterior(4.0, 1.0)}
        cal = Calibration(recall=recall, precision=precision)
        assert cal.recall[("e1", "security")].mean == pytest.approx(2.0 / 5.0)
        assert cal.precision[("e1", "security")].mean == pytest.approx(4.0 / 5.0)

    def test_calibration_is_frozen(self) -> None:
        cal = Calibration(recall={}, precision={})
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cal.recall = {}  # type: ignore[misc]

    def test_calibration_empty(self) -> None:
        cal = Calibration(recall={}, precision={})
        assert len(cal.recall) == 0
        assert len(cal.precision) == 0


# ---------------------------------------------------------------------------
# Sampler protocol shape (M20)
# ---------------------------------------------------------------------------


class TestSamplerProtocol:
    def test_stratified_sampler_implements_sampler_protocol(self) -> None:
        sampler: Sampler = StratifiedSampler()
        try:
            sampler.draw([], {"a": 1}, seed=0)
        except NotImplementedError as e:
            assert "Plan 4" in str(e)
        else:
            raise AssertionError("StratifiedSampler should raise NotImplementedError")

    def test_stratified_sampler_isinstance_sampler(self) -> None:
        assert isinstance(StratifiedSampler(), Sampler)

    def test_not_implemented_message_mentions_plan_1(self) -> None:
        sampler = StratifiedSampler()
        with pytest.raises(NotImplementedError, match="Plan 1"):
            sampler.draw([], {}, seed=42)


# ---------------------------------------------------------------------------
# CV (real k-fold implementation)
# ---------------------------------------------------------------------------


class TestCV:
    def test_cvresult_is_frozen_dataclass(self) -> None:
        result = CVResult(
            n_folds=5,
            fold_variances={("e1", "s1"): 0.01},
            interpretation={("e1", "s1"): "stable"},
            min_per_fold={("e1", "s1"): 10},
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.n_folds = 3  # type: ignore[misc]

    def test_cvresult_stores_fold_variances(self) -> None:
        fv: dict[tuple[str, str], float] = {("e1", "security"): 0.002}
        result = CVResult(
            n_folds=5,
            fold_variances=fv,
            interpretation={("e1", "security"): "stable"},
            min_per_fold={("e1", "security"): 10},
        )
        assert result.n_folds == 5
        assert result.fold_variances[("e1", "security")] == pytest.approx(0.002)

    def test_cross_validate_empty_returns_cvresult(self) -> None:
        result = cross_validate_calibration({}, {}, n_folds=5)
        assert result.n_folds == 5
        assert result.fold_variances == {}
