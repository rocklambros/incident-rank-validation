"""Unit tests for engine.report.render and engine.report.diff."""

from __future__ import annotations

from engine.decide.concordance import STANDING_CAVEAT, ConcordanceResult
from engine.decide.measurability import MeasurabilityMap
from engine.decide.robustness_multiplicity import (
    FlagDirection,
    FlagFinding,
    RobustnessSpread,
    SpecResult,
)
from engine.decide.rollup import RollupResult, RollupVerdict
from engine.decide.selection_bias import SelectionBiasDisclosure
from engine.model.censoring import MeasurabilityVerdict
from engine.report.diff import PreregDiff
from engine.report.render import ReportInputs, render_report


def _make_measurability_map(
    below_min: bool = False,
) -> MeasurabilityMap:
    return MeasurabilityMap(
        verdict={
            "LLM01": MeasurabilityVerdict.MEASURABLE,
            "LLM02": MeasurabilityVerdict.CLASSIFIER_BLIND_BOUNDED,
            "LLM03": MeasurabilityVerdict.FRAME_BLIND_UNMEASURABLE,
        },
        recall_p_above_threshold={"LLM01": 0.95, "LLM02": 0.5, "LLM03": 0.0},
        coverage_ratio=0.333,
        measurable=("LLM01",),
        classifier_blind=("LLM02",),
        frame_blind=("LLM03",),
        below_prereg_minimum=below_min,
    )


def _make_concordance(
    with_flags: bool = False,
    kappa: float | None = 0.65,
) -> ConcordanceResult:
    flags: tuple[FlagFinding, ...] = ()
    if with_flags:
        flags = (
            FlagFinding(
                entry_id="LLM01",
                direction=FlagDirection.VOTE_OVER_RANKS,
                probability=0.88,
            ),
        )
    return ConcordanceResult(
        weighted_kappa_median=kappa,
        weighted_kappa_ci=(0.55, 0.75) if kappa is not None else None,
        measurable_count=8,
        total_count=10,
        coverage_ratio=0.8,
        below_prereg_minimum=False,
        meaningful_kappa_n=5,
        flags=flags,
        standing_caveat=STANDING_CAVEAT,
    )


def _make_selection_bias() -> SelectionBiasDisclosure:
    return SelectionBiasDisclosure(
        statistic_name="kruskal_wallis_h",
        statistic_value=3.14,
        p_value=0.07,
        n_entries_per_group={"measurable": 5, "classifier_blind_bounded": 3, "frame_blind": 2},
        severity="low",
    )


def _make_inputs(
    non_publishable: bool = False,
    with_flags: bool = False,
    kappa: float | None = 0.65,
    rollup_results: tuple[RollupResult, ...] = (),
    prereg_diff: PreregDiff | None = None,
    robustness: RobustnessSpread | None = None,
    runpod_cost_usd: float | None = None,
    cost_ceiling_usd: float | None = None,
) -> ReportInputs:
    return ReportInputs(
        cycle_id="cycle-001",
        engine_version="0.1.0",
        measurability_map=_make_measurability_map(),
        concordance=_make_concordance(with_flags=with_flags, kappa=kappa),
        selection_bias=_make_selection_bias(),
        robustness=robustness,
        twin_agreement=None,
        non_publishable=non_publishable,
        rollup_results=rollup_results,
        prereg_diff=prereg_diff,
        runpod_cost_usd=runpod_cost_usd,
        cost_ceiling_usd=cost_ceiling_usd,
    )


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


class TestRenderReport:
    def test_contains_cycle_id(self) -> None:
        md = render_report(_make_inputs())
        assert "cycle-001" in md

    def test_contains_engine_version(self) -> None:
        md = render_report(_make_inputs())
        assert "0.1.0" in md

    def test_measurability_section(self) -> None:
        md = render_report(_make_inputs())
        assert "## Measurability Map" in md
        assert "Coverage ratio:" in md

    def test_concordance_section_with_kappa(self) -> None:
        md = render_report(_make_inputs())
        assert "## Concordance" in md
        assert "Weighted kappa:" in md
        assert "[0.55, 0.75]" in md

    def test_concordance_section_without_kappa(self) -> None:
        md = render_report(_make_inputs(kappa=None))
        assert "N/A: insufficient measurable subset" in md

    def test_standing_caveat_present(self) -> None:
        md = render_report(_make_inputs())
        assert STANDING_CAVEAT in md

    def test_selection_bias_section(self) -> None:
        md = render_report(_make_inputs())
        assert "## Selection Bias" in md
        assert "kruskal_wallis_h" in md
        assert "H = 3.1400" in md

    def test_threats_section(self) -> None:
        md = render_report(_make_inputs())
        assert "## Threats to Validity" in md
        assert "F-defenseindepth" in md
        assert "F1-ingestion-frame" in md

    def test_pre_publish_checklist_footer(self) -> None:
        md = render_report(_make_inputs())
        assert "PRE-PUBLISH CHECKLIST" in md
        assert "docs/REVIEWERS.md" in md
        assert "internal-only unless the checklist passes" in md

    def test_non_publishable_stamp(self) -> None:
        md = render_report(_make_inputs(non_publishable=True))
        assert "NON-PUBLISHABLE" in md
        assert "single-author rubric" in md

    def test_publishable_no_stamp(self) -> None:
        md = render_report(_make_inputs(non_publishable=False))
        assert "NON-PUBLISHABLE" not in md

    def test_flags_section_when_present(self) -> None:
        md = render_report(_make_inputs(with_flags=True))
        assert "## Flags" in md
        assert "LLM01" in md
        assert "P(tier mismatch) = 0.88" in md

    def test_no_flags_section_when_absent(self) -> None:
        md = render_report(_make_inputs(with_flags=False))
        assert "## Flags" not in md


# ---------------------------------------------------------------------------
# PreregDiff
# ---------------------------------------------------------------------------


class TestPreregDiff:
    def test_no_deviations(self) -> None:
        d = PreregDiff(deviations=())
        assert not d.has_deviations
        assert "No deviations" in d.to_markdown()

    def test_with_deviations(self) -> None:
        d = PreregDiff(deviations=("changed threshold", "added stratum"))
        assert d.has_deviations
        md = d.to_markdown()
        assert "## Pre-registration Deviations" in md
        assert "- changed threshold" in md
        assert "- added stratum" in md


# ---------------------------------------------------------------------------
# New report sections (Plan 5 Task 12)
# ---------------------------------------------------------------------------


def test_report_includes_rollup_section() -> None:
    rollup = (
        RollupResult(
            parent_entry_id="LLM06",
            child_entry_id="mcp-tool",
            verdict=RollupVerdict.SUPPORTED,
            p_distinct_cluster=0.85,
            child_median_lambda=0.04,
            parent_median_lambda=0.12,
            ratio_median=0.33,
        ),
    )
    inputs = _make_inputs(rollup_results=rollup)
    report = render_report(inputs)
    assert "Rollup" in report
    assert "mcp-tool" in report
    assert "SUPPORTED" in report or "supported" in report


def test_report_includes_prereg_diff_section() -> None:
    diff = PreregDiff(deviations=("flag_threshold_tau changed: 0.8 to 0.7",))
    inputs = _make_inputs(prereg_diff=diff)
    report = render_report(inputs)
    assert "Pre-registration" in report or "Deviation" in report


def test_report_no_prereg_diff_when_clean() -> None:
    diff = PreregDiff(deviations=())
    inputs = _make_inputs(prereg_diff=diff)
    report = render_report(inputs)
    assert "No deviations" in report or "Deviation" not in report


def test_report_includes_robustness_section() -> None:
    spread = RobustnessSpread(
        primary=SpecResult(
            spec_name="negative_binomial_per_stratum",
            weighted_kappa_median=0.80,
            weighted_kappa_ci=(0.65, 0.92),
            flags=(),
        ),
        robustness=(
            SpecResult(
                spec_name="poisson_flat",
                weighted_kappa_median=0.72,
                weighted_kappa_ci=(0.55, 0.88),
                flags=(),
            ),
        ),
    )
    inputs = _make_inputs(robustness=spread)
    report = render_report(inputs)
    assert "Robustness" in report
    assert "poisson_flat" in report
    assert "0.72" in report


def test_report_runpod_cost_disclosure() -> None:
    inputs = _make_inputs(runpod_cost_usd=87.50, cost_ceiling_usd=500.0)
    report = render_report(inputs)
    assert "87.50" in report or "87.5" in report
