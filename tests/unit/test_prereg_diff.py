# tests/unit/test_prereg_diff.py
from __future__ import annotations

from engine.report.diff import PreregDiff, compute_prereg_diff


class TestPreregDiff:
    def test_no_deviations(self) -> None:
        diff = compute_prereg_diff(
            prereg_primary_spec="negative_binomial_per_stratum",
            actual_primary_spec="negative_binomial_per_stratum",
            prereg_flag_tau=0.8,
            actual_flag_tau=0.8,
            prereg_measurability_min=4,
            actual_measurability_min=4,
            additional_deviations=(),
        )
        assert not diff.has_deviations

    def test_spec_deviation(self) -> None:
        diff = compute_prereg_diff(
            prereg_primary_spec="negative_binomial_per_stratum",
            actual_primary_spec="poisson_flat",
            prereg_flag_tau=0.8,
            actual_flag_tau=0.8,
            prereg_measurability_min=4,
            actual_measurability_min=4,
            additional_deviations=(),
        )
        assert diff.has_deviations
        assert any("primary_spec" in d for d in diff.deviations)

    def test_additional_deviations(self) -> None:
        diff = compute_prereg_diff(
            prereg_primary_spec="negative_binomial_per_stratum",
            actual_primary_spec="negative_binomial_per_stratum",
            prereg_flag_tau=0.8,
            actual_flag_tau=0.8,
            prereg_measurability_min=4,
            actual_measurability_min=4,
            additional_deviations=("Post-hoc temporal filter applied",),
        )
        assert diff.has_deviations
        assert "Post-hoc temporal filter applied" in diff.deviations

    def test_markdown_output(self) -> None:
        diff = compute_prereg_diff(
            prereg_primary_spec="nb",
            actual_primary_spec="poisson",
            prereg_flag_tau=0.8,
            actual_flag_tau=0.8,
            prereg_measurability_min=4,
            actual_measurability_min=4,
            additional_deviations=(),
        )
        md = diff.to_markdown()
        assert "Deviations" in md
