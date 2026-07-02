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


def test_slopegraph_in_blend_section_not_part1() -> None:
    srcs = _sources()
    blend = [s for s in srcs if "The 2026 blended Top 10" in s][0]
    part1 = [s for s in srcs if "What changed from 2025 to 2026" in s][0]
    assert "rank_change_2025_2026.png" in blend
    assert "| Blended # |" not in blend  # table removed
    assert "under the figure" in blend  # reworded from "under the table"
    assert "rank_change_2025_2026.png" not in part1  # moved out of Part I


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
        "rank_change_2025_2026.png": "width=92%",
    }
    for fname, attr in expected.items():
        assert f"{fname}){{{attr}}}" in a, f"missing/incorrect attrs for {fname}"
