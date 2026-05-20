"""Unit tests for engine.decide.selection_bias — Kruskal-Wallis (M14)."""

from __future__ import annotations

import math

import numpy as np

from engine.decide.selection_bias import compute_selection_bias

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeSelectionBias:
    def test_kruskal_fires_with_significant_difference(self) -> None:
        """When groups have clearly different rank distributions, p < 0.05."""
        verdicts = {
            **{f"m{i}": "measurable" for i in range(20)},
            **{f"fb{i}": "frame_blind_unmeasurable" for i in range(20)},
        }
        # Measurable entries get low ranks, frame-blind get high ranks
        rng = np.random.default_rng(42)
        median_ranks: dict[str, float] = {}
        for k in verdicts:
            if k.startswith("m"):
                median_ranks[k] = float(rng.uniform(1, 10))
            else:
                median_ranks[k] = float(rng.uniform(30, 40))

        result = compute_selection_bias(verdicts, median_ranks)

        assert result.statistic_name == "kruskal_wallis_h"
        assert result.p_value < 0.05
        assert result.severity in {"moderate", "high"}
        assert result.is_concerning()

    def test_severity_tags(self) -> None:
        """Severity should be low when distributions overlap heavily."""
        rng = np.random.default_rng(0)
        verdicts = {
            **{f"m{i}": "measurable" for i in range(15)},
            **{f"cb{i}": "classifier_blind_bounded" for i in range(15)},
        }
        # Both groups drawn from same distribution -> expect low severity
        median_ranks = {k: float(rng.uniform(1, 30)) for k in verdicts}

        result = compute_selection_bias(verdicts, median_ranks)

        # With same distribution, should typically be low
        assert result.severity in {"low", "moderate"}

    def test_nan_when_groups_too_small(self) -> None:
        """Return NaN when fewer than 2 non-empty groups have >= 2 entries."""
        verdicts = {"a": "measurable", "b": "measurable"}
        median_ranks = {"a": 1.0, "b": 2.0}

        result = compute_selection_bias(verdicts, median_ranks)

        assert math.isnan(result.statistic_value)
        assert math.isnan(result.p_value)
        assert result.severity == "low"
        assert not result.is_concerning()

    def test_nan_when_one_group_has_single_entry(self) -> None:
        """A group with only 1 entry cannot participate in Kruskal-Wallis."""
        verdicts = {
            "m0": "measurable",
            "m1": "measurable",
            "fb0": "frame_blind_unmeasurable",  # only 1 entry
        }
        median_ranks = {"m0": 1.0, "m1": 2.0, "fb0": 3.0}

        result = compute_selection_bias(verdicts, median_ranks)

        assert math.isnan(result.statistic_value)
        assert result.severity == "low"

    def test_entries_without_vote_ranks_are_skipped(self) -> None:
        """Entries missing from median_vote_ranks should be silently skipped."""
        verdicts = {"a": "measurable", "b": "measurable", "c": "frame_blind_unmeasurable"}
        median_ranks = {"a": 1.0, "b": 2.0}  # "c" has no rank

        result = compute_selection_bias(verdicts, median_ranks)

        assert result.n_entries_per_group["frame_blind_unmeasurable"] == 0

    def test_n_entries_per_group_reported(self) -> None:
        """Group sizes should be accurate in the disclosure."""
        verdicts = {
            "a": "measurable",
            "b": "measurable",
            "c": "classifier_blind_bounded",
            "d": "classifier_blind_bounded",
            "e": "frame_blind_unmeasurable",
            "f": "frame_blind_unmeasurable",
        }
        median_ranks = {k: float(i) for i, k in enumerate(verdicts)}

        result = compute_selection_bias(verdicts, median_ranks)

        assert result.n_entries_per_group["measurable"] == 2
        assert result.n_entries_per_group["classifier_blind_bounded"] == 2
        assert result.n_entries_per_group["frame_blind_unmeasurable"] == 2
