"""Tests for the 3 new preprint charts added in Task 3.

Each test asserts that the function:
  - returns a Path equal to figures_dir / <expected filename>
  - the file exists and is >1 KB (a real PNG, not a stub)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
CYCLE = REPO / "projects" / "owasp-llm" / "cycles" / "2026"

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def figures_dir(tmp_path: Path) -> Path:
    d = tmp_path / "figures"
    d.mkdir()
    return d


@pytest.fixture()
def minimal_blended() -> list[dict[str, Any]]:
    """Minimal blended ranking list (shape from engine.report.blend_2025_2026)."""
    return [
        {"entry_id": "LLM01", "vote_rank": 1, "lambda_rank": 2, "blend": 0.90, "blend_rank": 1},
        {"entry_id": "LLM02", "vote_rank": 3, "lambda_rank": 1, "blend": 0.80, "blend_rank": 2},
        {"entry_id": "LLM03", "vote_rank": 2, "lambda_rank": 4, "blend": 0.70, "blend_rank": 3},
        {"entry_id": "LLM04", "vote_rank": 5, "lambda_rank": 5, "blend": 0.60, "blend_rank": 4},
        {"entry_id": "LLM05", "vote_rank": 4, "lambda_rank": 6, "blend": 0.55, "blend_rank": 5},
        {"entry_id": "NEW-ITSCD", "vote_rank": 6, "lambda_rank": 3, "blend": 0.50, "blend_rank": 6},
        {"entry_id": "ROLL-CFAS", "vote_rank": 7, "lambda_rank": 7, "blend": 0.40, "blend_rank": 7},
    ]


@pytest.fixture()
def minimal_entry_names() -> dict[str, str]:
    return {
        "LLM01": "Prompt Injection",
        "LLM02": "Sensitive Info Disclosure",
        "LLM03": "Supply Chain",
        "LLM04": "Data and Model Poisoning",
        "LLM05": "Improper Output Handling",
        "NEW-ITSCD": "IT Supply Chain Disruption",
        "ROLL-CFAS": "CFAS Rollup",
    }


@pytest.fixture()
def minimal_entries() -> list[dict[str, Any]]:
    """Minimal entries list (shape from engine.report.blend_2025_2026.load_entries)."""
    return [
        {
            "entry_id": "LLM01",
            "canonical_name": "Prompt Injection",
            "group": "incumbent",
            "rolled_into": None,
        },
        {
            "entry_id": "LLM02",
            "canonical_name": "Sensitive Info Disclosure",
            "group": "incumbent",
            "rolled_into": None,
        },
        {
            "entry_id": "LLM03",
            "canonical_name": "Supply Chain",
            "group": "incumbent",
            "rolled_into": None,
        },
        {
            "entry_id": "NEW-ITSCD",
            "canonical_name": "IT Supply Chain Disruption",
            "group": "new",
            "rolled_into": None,
        },
        {
            "entry_id": "NEW-MA",
            "canonical_name": "Model Abuse",
            "group": "new",
            "rolled_into": None,
        },
        {
            "entry_id": "ROLL-CFAS",
            "canonical_name": "CFAS",
            "group": "rollup",
            "rolled_into": "LLM02",
        },
        {
            "entry_id": "ROLL-CMSB",
            "canonical_name": "CMSB",
            "group": "rollup",
            "rolled_into": "LLM01",
        },
    ]


@pytest.fixture()
def minimal_robustness() -> dict[str, Any]:
    """Minimal robustness dict (shape from robustness_validation.json)."""
    return {
        "ranking_fidelity_spearman_vs_truth": {
            "floor": 0.9183295277830658,
            "deepseek-v3": 0.9035055296946769,
            "llama-405b": 0.945030388410907,
            "mistral-large-2411": 0.9186749592520371,
            "qwen3-235b": 0.793597043108591,
            "ensemble": 0.9427495960611311,
        }
    }


# ---------------------------------------------------------------------------
# render_rank_change_2025_2026
# ---------------------------------------------------------------------------


class TestRenderRankChange20252026:
    def test_creates_png_at_expected_path(
        self,
        minimal_blended: list[dict[str, Any]],
        minimal_entry_names: dict[str, str],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_rank_change_2025_2026

        out = render_rank_change_2025_2026(minimal_blended, minimal_entry_names, figures_dir)
        assert out == figures_dir / "rank_change_2025_2026.png"
        assert out.exists()

    def test_png_exceeds_1kb(
        self,
        minimal_blended: list[dict[str, Any]],
        minimal_entry_names: dict[str, str],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_rank_change_2025_2026

        out = render_rank_change_2025_2026(minimal_blended, minimal_entry_names, figures_dir)
        assert out.stat().st_size > 1024, f"PNG too small: {out.stat().st_size} bytes"

    def test_returns_path_type(
        self,
        minimal_blended: list[dict[str, Any]],
        minimal_entry_names: dict[str, str],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_rank_change_2025_2026

        out = render_rank_change_2025_2026(minimal_blended, minimal_entry_names, figures_dir)
        assert isinstance(out, Path)


# ---------------------------------------------------------------------------
# render_entry_expansion_map
# ---------------------------------------------------------------------------


class TestRenderEntryExpansionMap:
    def test_creates_png_at_expected_path(
        self,
        minimal_entries: list[dict[str, Any]],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_entry_expansion_map

        out = render_entry_expansion_map(minimal_entries, figures_dir)
        assert out == figures_dir / "entry_expansion_map.png"
        assert out.exists()

    def test_png_exceeds_1kb(
        self,
        minimal_entries: list[dict[str, Any]],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_entry_expansion_map

        out = render_entry_expansion_map(minimal_entries, figures_dir)
        assert out.stat().st_size > 1024, f"PNG too small: {out.stat().st_size} bytes"

    def test_returns_path_type(
        self,
        minimal_entries: list[dict[str, Any]],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_entry_expansion_map

        out = render_entry_expansion_map(minimal_entries, figures_dir)
        assert isinstance(out, Path)


# ---------------------------------------------------------------------------
# render_rarr_robustness
# ---------------------------------------------------------------------------


class TestRenderRarrRobustness:
    def test_creates_png_at_expected_path(
        self,
        minimal_robustness: dict[str, Any],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_rarr_robustness

        out = render_rarr_robustness(minimal_robustness, figures_dir)
        assert out == figures_dir / "rarr_robustness.png"
        assert out.exists()

    def test_png_exceeds_1kb(
        self,
        minimal_robustness: dict[str, Any],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_rarr_robustness

        out = render_rarr_robustness(minimal_robustness, figures_dir)
        assert out.stat().st_size > 1024, f"PNG too small: {out.stat().st_size} bytes"

    def test_returns_path_type(
        self,
        minimal_robustness: dict[str, Any],
        figures_dir: Path,
    ) -> None:
        from engine.report.narrative_charts import render_rarr_robustness

        out = render_rarr_robustness(minimal_robustness, figures_dir)
        assert isinstance(out, Path)


# ---------------------------------------------------------------------------
# _rank_change_rows (slopegraph row model)
# ---------------------------------------------------------------------------
class TestRankChangeRows:
    def _blended(self) -> list[dict[str, Any]]:
        # incumbent-only blended list with a mix of moves incl. LLM07 (nc, renamed)
        # published rank = int(LLMkk); move = published - blend_rank
        return [
            {"entry_id": "LLM02", "blend_rank": 1},  # pub 2 -> +1  hold
            {"entry_id": "LLM01", "blend_rank": 2},  # pub 1 -> -1  hold
            {"entry_id": "LLM06", "blend_rank": 3},  # pub 6 -> +3  mover
            {"entry_id": "LLM04", "blend_rank": 4},  # pub 4 -> 0   nc
            {"entry_id": "LLM03", "blend_rank": 5},  # pub 3 -> -2  mover
            {"entry_id": "LLM10", "blend_rank": 6},  # pub 10 -> +4 mover
            {"entry_id": "LLM07", "blend_rank": 7},  # pub 7 -> 0   nc + renamed
            {"entry_id": "LLM09", "blend_rank": 8},  # pub 9 -> +1  hold
            {"entry_id": "LLM08", "blend_rank": 9},  # pub 8 -> -1  hold
            {"entry_id": "LLM05", "blend_rank": 10}, # pub 5 -> -5  mover
        ]

    def _names(self) -> dict[str, str]:
        return {
            "LLM01": "Prompt Injection", "LLM02": "Sensitive Information Disclosure",
            "LLM03": "Supply Chain", "LLM04": "Data and Model Poisoning",
            "LLM05": "Improper Output Handling", "LLM06": "Excessive Agency",
            "LLM07": "Hidden Context Exposure", "LLM08": "Vector and Embedding Weaknesses",
            "LLM09": "Misinformation", "LLM10": "Unbounded Consumption",
        }

    def test_new_code_equals_blend_position_and_moves(self) -> None:
        from engine.report.narrative_charts import _rank_change_rows
        rows = {r["right_code"]: r for r in _rank_change_rows(self._blended(), self._names())}
        assert rows["LLM01"]["right_name"] == "Sensitive Information Disclosure"
        assert rows["LLM01"]["move"] == 1
        # right_code is keyed by blend position, not by the original entry_id:
        # LLM06 (pub 6) blends to rank 3 -> right_code "LLM03"; LLM10 (pub 10)
        # blends to rank 6 -> right_code "LLM06". (Brief had these two swapped
        # to the original entry_id; corrected here — see task-1-report.md.)
        assert rows["LLM03"]["move"] == 3 and rows["LLM03"]["style"] == "mover"
        assert rows["LLM06"]["move"] == 4 and rows["LLM06"]["style"] == "mover"
        assert rows["LLM06"]["right_name"] == "Unbounded Consumption"

    def test_renamed_only_llm07(self) -> None:
        from engine.report.narrative_charts import _rank_change_rows
        rows = _rank_change_rows(self._blended(), self._names())
        renamed = {r["right_code"] for r in rows if r["renamed"]}
        assert renamed == {"LLM07"}

    def test_style_bands(self) -> None:
        from engine.report.narrative_charts import _rank_change_rows
        rows = {r["right_code"]: r for r in _rank_change_rows(self._blended(), self._names())}
        assert rows["LLM04"]["style"] == "nc"      # move 0
        assert rows["LLM02"]["style"] == "hold"    # |move| == 1
        assert rows["LLM05"]["style"] == "mover"   # |move| == 5

    def test_left_side_uses_published_2025_name_and_rank(self) -> None:
        from engine.report.narrative_charts import _rank_change_rows
        rows = {r["right_code"]: r for r in _rank_change_rows(self._blended(), self._names())}
        # LLM07 published 2025 name is "System Prompt Leakage"; left_num == published rank
        assert rows["LLM07"]["left_code"] == "LLM07"
        assert rows["LLM07"]["left_name"] == "System Prompt Leakage"
        assert rows["LLM07"]["left_num"] == 7


class TestRankChangeShape:
    def test_landscape_not_tall(
        self,
        minimal_blended: list[dict[str, Any]],
        minimal_entry_names: dict[str, str],
        figures_dir: Path,
    ) -> None:
        from PIL import Image

        from engine.report.narrative_charts import render_rank_change_2025_2026
        out = render_rank_change_2025_2026(minimal_blended, minimal_entry_names, figures_dir)
        # Context manager avoids a dangling FileIO handle — this repo's pytest
        # config runs with filterwarnings = ["error"], which turns the
        # ResourceWarning from an unclosed Image.open() into a hard failure.
        with Image.open(out) as img:
            w, h = img.size
        assert h / w <= 0.85, f"slopegraph too tall: h/w={h/w:.2f}"


@pytest.mark.integration
class TestBumpChartFocused:
    """Fig 9: focused expert-vs-incident slopegraph, real cycle data."""

    def _data(self) -> dict[str, Any]:
        from engine.report.narrative_data import load_narrative_data
        return load_narrative_data(CYCLE)

    def test_renders_and_is_not_tall(self, figures_dir: Path) -> None:
        from PIL import Image

        from engine.report.narrative_charts import render_bump_chart
        render_bump_chart(self._data(), figures_dir)
        out = figures_dir / "bump_chart.png"
        assert out.exists() and out.stat().st_size > 1024
        # Context manager avoids a dangling FileIO handle — this repo's pytest
        # config runs with filterwarnings = ["error"], which turns the
        # ResourceWarning from an unclosed Image.open() into a hard failure.
        with Image.open(out) as img:
            w, h = img.size
        assert h / w <= 0.90, f"bump chart too tall: h/w={h/w:.2f}"
