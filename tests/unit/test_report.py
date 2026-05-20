"""Unit tests for engine.report.render and engine.report.diff."""

from __future__ import annotations

from engine.decide.concordance import STANDING_CAVEAT, ConcordanceResult
from engine.decide.measurability import MeasurabilityMap
from engine.decide.robustness_multiplicity import FlagDirection, FlagFinding
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
) -> ReportInputs:
    return ReportInputs(
        cycle_id="cycle-001",
        engine_version="0.1.0",
        measurability_map=_make_measurability_map(),
        concordance=_make_concordance(with_flags=with_flags, kappa=kappa),
        selection_bias=_make_selection_bias(),
        robustness=None,
        twin_agreement=None,
        non_publishable=non_publishable,
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
