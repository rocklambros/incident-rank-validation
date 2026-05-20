"""Unit tests for engine.model.overlap.OverlapWeights."""

from __future__ import annotations

import pytest

from engine.model.overlap import OverlapWeights


class TestSelfLoopRejection:
    def test_self_loop_raises(self) -> None:
        with pytest.raises(ValueError, match="self-loop"):
            OverlapWeights(weights={"E01": {"E01": 0.5}})

    def test_self_loop_among_other_sources_raises(self) -> None:
        with pytest.raises(ValueError, match="self-loop"):
            OverlapWeights(weights={"E01": {"E02": 0.3, "E01": 0.2}})


class TestColumnStochasticity:
    def test_column_sum_over_one_raises(self) -> None:
        # E02 → E01 at 0.7, E02 → E03 at 0.5: column sum for E02 = 1.2 > 1
        with pytest.raises(ValueError, match="sum to"):
            OverlapWeights(
                weights={
                    "E01": {"E02": 0.7},
                    "E03": {"E02": 0.5},
                }
            )

    def test_column_sum_exactly_one_accepted(self) -> None:
        ow = OverlapWeights(
            weights={
                "E01": {"E02": 0.6},
                "E03": {"E02": 0.4},
            }
        )
        assert ow.weights["E01"]["E02"] == pytest.approx(0.6)

    def test_column_sum_just_over_tolerance_raises(self) -> None:
        # 1.0 + 1e-5 > 1e-6 tolerance
        with pytest.raises(ValueError, match="sum to"):
            OverlapWeights(
                weights={
                    "E01": {"E02": 0.5},
                    "E03": {"E02": 0.5 + 1e-5},
                }
            )


class TestValidWeights:
    def test_empty_weights_accepted(self) -> None:
        ow = OverlapWeights(weights={})
        assert ow.weights == {}

    def test_single_target_leakage_accepted(self) -> None:
        ow = OverlapWeights(weights={"E01": {"E02": 0.3}})
        assert ow.weights["E01"]["E02"] == pytest.approx(0.3)

    def test_multi_target_leakage_within_bounds(self) -> None:
        # E02 leaks into E01 at 0.3 and E03 at 0.4: column sum = 0.7 <= 1
        ow = OverlapWeights(
            weights={
                "E01": {"E02": 0.3},
                "E03": {"E02": 0.4},
            }
        )
        assert ow.weights["E01"]["E02"] == pytest.approx(0.3)
        assert ow.weights["E03"]["E02"] == pytest.approx(0.4)

    def test_multiple_sources_no_cross_column_violation(self) -> None:
        # E02 → E01 at 0.3; E03 → E01 at 0.5; E03 → E04 at 0.4
        # col(E02)=0.3, col(E03)=0.9 — both fine
        ow = OverlapWeights(
            weights={
                "E01": {"E02": 0.3, "E03": 0.5},
                "E04": {"E03": 0.4},
            }
        )
        assert ow.weights["E01"]["E03"] == pytest.approx(0.5)

    def test_frozen(self) -> None:
        ow = OverlapWeights(weights={"E01": {"E02": 0.3}})
        with pytest.raises((AttributeError, TypeError)):
            ow.weights = {}  # type: ignore[misc]
