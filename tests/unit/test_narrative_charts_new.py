"""Tests for the 3 new preprint charts added in Task 3.

Each test asserts that the function:
  - returns a Path equal to figures_dir / <expected filename>
  - the file exists and is >1 KB (a real PNG, not a stub)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
