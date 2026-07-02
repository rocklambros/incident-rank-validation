"""Blend 2025→2026 ranking compute module.

Pure functions — no I/O except ``load_entries`` reading a taxonomy JSON.
All inputs (``vote_ranks``, ``lambda_ranks``) are passed by callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NotRequired, TypedDict


class TaxonomyEntry(TypedDict):
    """One entry from taxonomy.json, annotated with a group label."""

    entry_id: str
    canonical_name: str
    group: str  # "incumbent" | "new" | "rollup"
    rolled_into: NotRequired[str | None]


class BlendedEntry(TypedDict):
    """Result row produced by :func:`blended_ranking`."""

    entry_id: str
    vote_rank: int
    lambda_rank: int
    blend: float
    blend_rank: int


def load_entries(taxonomy_path: Path) -> list[TaxonomyEntry]:
    """Load taxonomy entries and annotate each with a group label.

    ``group`` ∈ {"incumbent","new","rollup"} derived from
    ``is_incumbent`` / ``is_rollup_candidate`` flags.  Rollup entries
    also carry ``rolled_into``.
    """
    data: dict[str, object] = json.loads(taxonomy_path.read_text())
    raw_entries = data["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("taxonomy.json 'entries' must be a list")

    result: list[TaxonomyEntry] = []
    for e in raw_entries:
        if not isinstance(e, dict):
            raise ValueError(f"Unexpected entry type: {type(e)}")
        if e.get("is_incumbent"):
            group: str = "incumbent"
        elif e.get("is_rollup_candidate"):
            group = "rollup"
        else:
            group = "new"

        entry: TaxonomyEntry = {
            "entry_id": str(e["entry_id"]),
            "canonical_name": str(e["canonical_name"]),
            "group": group,
        }
        if group == "rollup":
            raw = e.get("rolled_into")
            entry["rolled_into"] = str(raw) if raw is not None else None
        result.append(entry)

    return result


def blended_ranking(
    vote_ranks: dict[str, int],
    lambda_ranks: dict[str, int],
    w_vote: float = 0.75,
) -> list[BlendedEntry]:
    """Compute blended ranking for entries present in both rank dicts.

    ``blend = w_vote * vote_rank + (1 - w_vote) * lambda_rank``

    Returns rows sorted ascending by ``(blend, entry_id)`` for
    determinism on ties; ``blend_rank`` is 1-based.
    """
    entries: list[BlendedEntry] = []
    for entry_id, vr in vote_ranks.items():
        lr = lambda_ranks[entry_id]
        blend = w_vote * vr + (1.0 - w_vote) * lr
        entries.append(
            {
                "entry_id": entry_id,
                "vote_rank": vr,
                "lambda_rank": lr,
                "blend": blend,
                "blend_rank": 0,  # filled below
            }
        )

    entries.sort(key=lambda x: (x["blend"], x["entry_id"]))

    for i, e in enumerate(entries, 1):
        e["blend_rank"] = i

    return entries


def rank_moves(
    published_order: list[str],
    blended: list[BlendedEntry],
) -> dict[str, int]:
    """Compute position changes from published order to blended ranking.

    Returns ``{entry_id: published_pos - blend_pos}`` where positive means
    the entry moved up in the blended ranking relative to its published
    position.  ``published_pos`` is 1-based.
    """
    blend_pos: dict[str, int] = {e["entry_id"]: e["blend_rank"] for e in blended}
    return {
        entry_id: (pub_pos - blend_pos[entry_id])
        for pub_pos, entry_id in enumerate(published_order, 1)
    }
