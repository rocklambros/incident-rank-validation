from __future__ import annotations

from pathlib import Path
from typing import cast

from engine.report.blend_2025_2026 import BlendedEntry, blended_ranking, load_entries, rank_moves

TAX = Path("projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json")


def test_groups_10_6_4() -> None:
    g: dict[str, int] = {}
    for e in load_entries(TAX):
        g[e["group"]] = g.get(e["group"], 0) + 1
    assert g == {"incumbent": 10, "new": 6, "rollup": 4}


def test_blend_formula_llm01_folded() -> None:
    # LLM01 vote_rank 1, folded lambda_rank 9 -> blend 3.00 (methodology doc worked example)
    ranks = blended_ranking({"LLM01": 1, "LLM02": 2}, {"LLM01": 9, "LLM02": 2})
    out = {r["entry_id"]: r for r in ranks}
    assert abs(out["LLM01"]["blend"] - 3.00) < 1e-9
    assert abs(out["LLM02"]["blend"] - 2.00) < 1e-9


def test_rank_moves_sign() -> None:
    blended = cast(
        list[BlendedEntry],
        [{"entry_id": "A", "blend_rank": 1}, {"entry_id": "B", "blend_rank": 2}],
    )
    assert rank_moves(["B", "A"], blended) == {"B": -1, "A": 1}
