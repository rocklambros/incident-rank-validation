"""Merkle-chained post-hoc analysis register (M16 / HANDOFF v2.5 §6.11).

Provides an append-only, tamper-evident register for post-hoc analyses
that must be declared after pre-registration but before results are final.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from engine.erratum.merkle import GENESIS, chain_link, verify_chain


@dataclass(frozen=True, slots=True)
class PostHocAnalysis:
    """One post-hoc analysis entry."""

    cycle_id: str
    title: str
    description: str
    rationale: str
    added_at: str  # ISO 8601
    artifacts: tuple[str, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class PostHocRegister:
    """Append-only register of post-hoc analyses with chain integrity."""

    cycle_id: str
    analyses: tuple[PostHocAnalysis, ...]
    chain_terminal_hash: str


def _analysis_to_payload(a: PostHocAnalysis) -> dict[str, object]:
    return {
        "cycle_id": a.cycle_id,
        "title": a.title,
        "description": a.description,
        "rationale": a.rationale,
        "added_at": a.added_at,
        "artifacts": list(a.artifacts),
    }


def write_register(register: PostHocRegister, path: Path) -> None:
    """Serialize register to JSON with Merkle chain hashes."""
    entries: list[dict[str, object]] = []
    prev = GENESIS
    for a in register.analyses:
        payload = _analysis_to_payload(a)
        h = chain_link(prev, payload)
        entry: dict[str, object] = {**payload, "chain_hash": h}
        entries.append(entry)
        prev = h

    doc: dict[str, object] = {
        "cycle_id": register.cycle_id,
        "analyses": entries,
        "chain_terminal_hash": prev,
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_artifacts(raw: object) -> tuple[str, ...]:
    """Safely convert a JSON-decoded artifacts list to a tuple of strings."""
    if not isinstance(raw, list):
        return ()
    return tuple(str(x) for x in raw)


def read_and_verify_register(path: Path) -> PostHocRegister:
    """Load register from JSON and verify chain integrity."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, object]] = doc["analyses"]

    terminal = verify_chain(entries)

    if doc.get("chain_terminal_hash") != terminal:
        raise ValueError(
            f"terminal hash mismatch: file says {doc.get('chain_terminal_hash')}, "
            f"computed {terminal}"
        )

    analyses = tuple(
        PostHocAnalysis(
            cycle_id=str(e["cycle_id"]),
            title=str(e["title"]),
            description=str(e["description"]),
            rationale=str(e["rationale"]),
            added_at=str(e["added_at"]),
            artifacts=_parse_artifacts(e.get("artifacts")),
        )
        for e in entries
    )

    return PostHocRegister(
        cycle_id=str(doc["cycle_id"]),
        analyses=analyses,
        chain_terminal_hash=terminal,
    )


def append_analysis(
    register: PostHocRegister,
    analysis: PostHocAnalysis,
) -> PostHocRegister:
    """Append one analysis to a register, extending the chain."""
    new_analyses = (*register.analyses, analysis)

    # Recompute the full chain to get the new terminal hash
    prev = GENESIS
    for a in new_analyses:
        prev = chain_link(prev, _analysis_to_payload(a))

    return PostHocRegister(
        cycle_id=register.cycle_id,
        analyses=new_analyses,
        chain_terminal_hash=prev,
    )
