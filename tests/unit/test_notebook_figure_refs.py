from __future__ import annotations

import json
from pathlib import Path

NB = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "2026_top_10_llm_update_what_the_data_says.ipynb"
)


def _sources() -> list[str]:
    nb = json.loads(NB.read_text())
    return ["".join(c["source"]) for c in nb["cells"]]


def _all() -> str:
    return "\n".join(_sources())


def test_plotly_rankings_fully_removed() -> None:
    a = _all()
    assert "plotly_rankings.png" not in a
    assert "render_plotly_rankings" not in a
    assert "interactive companion" not in a


def test_ranking_figures_in_blend_section() -> None:
    # The probabilistic relock retired the rank_change slopegraph and the interim
    # "Blended #" table; the ranking is now shown by the two uncertainty figures.
    srcs = _sources()
    blend = [s for s in srcs if "The 2026 blended Top 10" in s][0]
    part1 = [s for s in srcs if "What changed from 2025 to 2026" in s][0]
    assert "blend_position_intervals.png" in blend
    assert "blend_top_k_probs.png" in blend
    assert "| Blended # |" not in blend  # interim table removed
    assert "rank_change_2025_2026.png" not in _all()  # slopegraph fully retired
    assert "blend_position_intervals.png" not in part1  # ranking figures live in the blend section


def test_layout_attributes_applied() -> None:
    a = _all()
    expected = {
        "stratum_bar.png": "width=42% wrap=right",
        "tier_donut.png": "width=40% wrap=right",
        "precision_bars.png": "width=60%",
        "paired_dots.png": "width=46% wrap=right",
        "theme_bars_llm09.png": "width=42% wrap=left",
        "sankey_confusion.png": "width=90%",
        "rarr_robustness.png": "width=48% wrap=left",
        "blend_position_intervals.png": "width=70% wrap=left",
        "blend_top_k_probs.png": "width=62%",
    }
    for fname, attr in expected.items():
        assert f"{fname}){{{attr}}}" in a, f"missing/incorrect attrs for {fname}"
