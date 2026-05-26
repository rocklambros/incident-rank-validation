"""Tests for engine.report.narrative — standalone narrative report generator."""
from __future__ import annotations
import json
import re
from pathlib import Path
import pytest

CYCLE_DIR = Path("projects/owasp-llm/cycles/2026")
SKIP_NO_CYCLE = pytest.mark.skipif(not CYCLE_DIR.exists(), reason="Cycle data not present")


@SKIP_NO_CYCLE
class TestNarrativeDataLoading:
    def test_load_data_returns_dict(self) -> None:
        from engine.report.narrative_data import load_narrative_data
        data = load_narrative_data(CYCLE_DIR)
        assert isinstance(data, dict)

    def test_load_data_has_required_keys(self) -> None:
        from engine.report.narrative_data import load_narrative_data
        data = load_narrative_data(CYCLE_DIR)
        required = {"rubric", "incidents", "prelabels", "goldset", "precision_verification", "posteriors", "diagnostic", "inference_summary", "lambda_samples", "concordance", "selection_bias", "rank_comparison_md"}
        missing = required - set(data.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_lambda_samples_shape(self) -> None:
        from engine.report.narrative_data import load_narrative_data
        data = load_narrative_data(CYCLE_DIR)
        assert data["lambda_samples"].shape == (16000, 20)

    def test_concordance_has_ci_method(self) -> None:
        from engine.report.narrative_data import load_narrative_data
        data = load_narrative_data(CYCLE_DIR)
        assert "ci_method" in data["concordance"]


REPORT_STEM = "2026_top_10_llm_update_what_the_data_says"

EXPECTED_FIGURES = [
    "stratum_bar.png",
    "tier_donut.png",
    "confusion_heatmap.png",
    "precision_bars.png",
    "precision_posteriors.png",
    "ridge_plot.png",
    "dumbbell_chart.png",
    "plotly_rankings.png",
    "bump_chart.png",
    "ci_overlap.png",
    "paired_dots.png",
    "theme_bars_llm09.png",
    "theme_bars_new_wla.png",
    "oos_treemap.png",
    "sankey_confusion.png",
    "confusion_matrix_3x3.png",
]

ACT_HEADINGS = [
    "Act 1", "Act 2", "Act 3", "Act 4", "Act 5",
    "Act 6", "Act 7", "Act 8", "Act 9", "Act 10",
]

AI_SLOP_PATTERNS = [
    r"\bdelve\b",
    r"\btapestry\b",
    r"\bunlock\b.*\bpotential\b",
    r"\bsynergy\b",
    r"\bholistic\b",
    r"\bparadigm shift\b",
    r"\bgame.?changer\b",
    r"\beverchanging\b",
]


@SKIP_NO_CYCLE
class TestNarrativeIntegration:
    @pytest.fixture(scope="class")
    def narrative_output(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        from engine.report.narrative import generate_narrative_report
        output_dir = tmp_path_factory.mktemp("narrative")
        generate_narrative_report(CYCLE_DIR, output_dir)
        return output_dir

    def test_report_exists(self, narrative_output: Path) -> None:
        assert (narrative_output / f"{REPORT_STEM}.md").exists()

    def test_pdf_exists(self, narrative_output: Path) -> None:
        import shutil
        if shutil.which("pandoc") is None:
            pytest.skip("pandoc not installed; PDF compile skipped")
        pdf_path = narrative_output / f"{REPORT_STEM}.pdf"
        assert pdf_path.exists(), "PDF was not produced"
        assert pdf_path.stat().st_size > 10 * 1024, "PDF suspiciously small (<10KB)"

    def test_has_abstract_and_toc(self, narrative_output: Path) -> None:
        text = (narrative_output / f"{REPORT_STEM}.md").read_text()
        assert "abstract:" in text, "Pandoc YAML abstract missing"
        assert "Table of Contents" in text, "Manual TOC heading missing"
        assert "toc: true" in text, "Pandoc TOC flag missing in frontmatter"

    def test_all_figures_present(self, narrative_output: Path) -> None:
        figures_dir = narrative_output / "figures"
        for fig_name in EXPECTED_FIGURES:
            fig_path = figures_dir / fig_name
            assert fig_path.exists(), f"Missing figure: {fig_name}"
            assert fig_path.stat().st_size > 1024, f"Figure too small (<1KB): {fig_name}"

    def test_all_act_headings_present(self, narrative_output: Path) -> None:
        report_text = (narrative_output / f"{REPORT_STEM}.md").read_text()
        for heading in ACT_HEADINGS:
            assert heading in report_text, f"Missing heading: {heading}"

    def test_no_ai_slop(self, narrative_output: Path) -> None:
        report_text = (narrative_output / f"{REPORT_STEM}.md").read_text()
        for pattern in AI_SLOP_PATTERNS:
            matches = re.findall(pattern, report_text, re.IGNORECASE)
            assert not matches, f"AI slop detected: {pattern} -> {matches}"

    def test_figure_references_valid(self, narrative_output: Path) -> None:
        report_text = (narrative_output / f"{REPORT_STEM}.md").read_text()
        refs = re.findall(r"!\[.*?\]\((figures/[^)]+)\)", report_text)
        for ref in refs:
            fig_path = narrative_output / ref
            assert fig_path.exists(), f"Broken image ref: {ref}"

    def test_non_publishable_banner(self, narrative_output: Path) -> None:
        report_text = (narrative_output / f"{REPORT_STEM}.md").read_text()
        assert "NON-PUBLISHABLE" in report_text

    def test_threats_section_has_f_aiharm_precision(self, narrative_output: Path) -> None:
        report_text = (narrative_output / f"{REPORT_STEM}.md").read_text()
        assert "F-aiharm-precision" in report_text

    def test_kappa_congruence_with_concordance_json(self, narrative_output: Path) -> None:
        report_text = (narrative_output / f"{REPORT_STEM}.md").read_text()
        conc_data = json.loads((CYCLE_DIR / "results" / "concordance.json").read_text())
        kappa_str = f"{conc_data['weighted_kappa_median']:.4f}"
        assert kappa_str in report_text, f"Kappa {kappa_str} not found in narrative report"
        ci = conc_data.get("weighted_kappa_ci", [])
        if ci:
            assert f"{ci[0]:.4f}" in report_text, f"CI lower bound not in report"
            assert f"{ci[1]:.4f}" in report_text, f"CI upper bound not in report"
        flags = conc_data.get("flags", [])
        assert f"{len(flags)} entries flagged" in report_text or len(flags) == 0

    def test_bump_chart_has_real_data(self, narrative_output: Path) -> None:
        report_text = (narrative_output / f"{REPORT_STEM}.md").read_text()
        assert "No rank data available" not in report_text
