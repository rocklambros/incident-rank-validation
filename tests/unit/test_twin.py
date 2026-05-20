"""Unit tests for engine.model.twin — point-estimate robustness twin."""

from __future__ import annotations

import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.overlap import OverlapWeights
from engine.model.twin import TwinResult, compute_twin

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _calibration(
    recall_mean: float = 0.8,
    precision_mean: float = 0.9,
    entries: tuple[str, ...] = ("A", "B"),
    strata: tuple[str, ...] = ("s1",),
) -> Calibration:
    """Build a simple uniform Calibration for the given entries/strata."""
    recall: dict[tuple[str, str], BetaPosterior] = {}
    precision: dict[tuple[str, str], BetaPosterior] = {}
    for e in entries:
        for s in strata:
            # Encode mean via alpha=mean*10, beta=(1-mean)*10 (sum=10)
            r_a = recall_mean * 10
            r_b = (1.0 - recall_mean) * 10
            recall[(e, s)] = BetaPosterior(alpha=r_a, beta=r_b)
            p_a = precision_mean * 10
            p_b = (1.0 - precision_mean) * 10
            precision[(e, s)] = BetaPosterior(alpha=p_a, beta=p_b)
    return Calibration(recall=recall, precision=precision)


def _empty_overlap() -> OverlapWeights:
    return OverlapWeights(weights={})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeTwin:
    def test_de_biased_prevalence_is_non_negative(self) -> None:
        """Adjusted prevalence must never be negative."""
        cal = _calibration()
        overlap = _empty_overlap()
        result = compute_twin(
            measurable_entries=("A", "B"),
            strata=("s1",),
            observed_counts={("A", "s1"): 10, ("B", "s1"): 2},
            stratum_sizes={"s1": 100},
            calibration=cal,
            overlap=overlap,
        )
        for eid, prev in result.prevalence_estimates.items():
            assert prev >= 0.0, f"Negative prevalence for {eid}: {prev}"

    def test_high_prevalence_entry_ranked_first(self) -> None:
        """Entry with more counts should rank above the low-count entry."""
        cal = _calibration(entries=("A", "B"), strata=("s1",))
        overlap = _empty_overlap()
        result = compute_twin(
            measurable_entries=("A", "B"),
            strata=("s1",),
            observed_counts={("A", "s1"): 50, ("B", "s1"): 5},
            stratum_sizes={"s1": 100},
            calibration=cal,
            overlap=overlap,
        )
        assert result.rank[0] == "A", (
            f"Expected 'A' ranked first, got {result.rank}"
        )
        assert result.prevalence_estimates["A"] > result.prevalence_estimates["B"]

    def test_overlap_weights_reduce_estimate(self) -> None:
        """FP leakage from source entry should lower target's adjusted prevalence."""
        entries = ("source", "target")
        strata = ("s1",)
        cal = _calibration(
            recall_mean=0.8,
            precision_mean=0.8,  # 20% FP rate on source
            entries=entries,
            strata=strata,
        )
        # Without overlap
        overlap_none = _empty_overlap()
        result_no_overlap = compute_twin(
            measurable_entries=entries,
            strata=strata,
            observed_counts={("source", "s1"): 100, ("target", "s1"): 10},
            stratum_sizes={"s1": 1000},
            calibration=cal,
            overlap=overlap_none,
        )

        # With 50% of source FPs landing in target
        overlap_some = OverlapWeights(weights={"target": {"source": 0.5}})
        result_with_overlap = compute_twin(
            measurable_entries=entries,
            strata=strata,
            observed_counts={("source", "s1"): 100, ("target", "s1"): 10},
            stratum_sizes={"s1": 1000},
            calibration=cal,
            overlap=overlap_some,
        )

        assert (
            result_with_overlap.prevalence_estimates["target"]
            < result_no_overlap.prevalence_estimates["target"]
        ), "Overlap should reduce target's de-biased prevalence"

    def test_empty_overlap_works(self) -> None:
        """compute_twin succeeds with an empty OverlapWeights."""
        cal = _calibration()
        result = compute_twin(
            measurable_entries=("A",),
            strata=("s1",),
            observed_counts={("A", "s1"): 7},
            stratum_sizes={"s1": 100},
            calibration=cal,
            overlap=_empty_overlap(),
        )
        assert isinstance(result, TwinResult)
        assert result.rank == ("A",)

    def test_missing_calibration_uses_fallback(self) -> None:
        """When calibration keys are absent the fallback defaults are used."""
        # No keys in calibration — should not raise
        cal = Calibration(recall={}, precision={})
        result = compute_twin(
            measurable_entries=("X",),
            strata=("s1",),
            observed_counts={("X", "s1"): 20},
            stratum_sizes={"s1": 100},
            calibration=cal,
            overlap=_empty_overlap(),
        )
        assert result.prevalence_estimates["X"] > 0.0

    def test_result_is_frozen_dataclass(self) -> None:
        """TwinResult must be immutable."""
        cal = _calibration()
        result = compute_twin(
            measurable_entries=("A",),
            strata=("s1",),
            observed_counts={},
            stratum_sizes={"s1": 100},
            calibration=cal,
            overlap=_empty_overlap(),
        )
        with pytest.raises((AttributeError, TypeError)):
            result.rank = ("B",)  # type: ignore[misc]

    def test_multi_stratum_sums_correctly(self) -> None:
        """Prevalence is accumulated across strata and normalised by total size."""
        # BetaPosterior requires strictly positive parameters, so use near-perfect
        # recall/precision (0.9999) rather than exact 1.0.  With recall ≈ 1 and
        # precision ≈ 1 the math is essentially: adjusted ≈ obs; prev ≈ 30/300.
        cal = _calibration(
            recall_mean=0.9999,
            precision_mean=0.9999,
            entries=("A",),
            strata=("s1", "s2"),
        )
        result = compute_twin(
            measurable_entries=("A",),
            strata=("s1", "s2"),
            observed_counts={("A", "s1"): 10, ("A", "s2"): 20},
            stratum_sizes={"s1": 100, "s2": 200},
            calibration=cal,
            overlap=_empty_overlap(),
        )
        # adjusted ≈ obs (recall ≈ 1, no FPs); prev ≈ 30/300 = 0.1
        assert abs(result.prevalence_estimates["A"] - 0.1) < 1e-3
