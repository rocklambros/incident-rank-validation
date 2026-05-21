"""Adjudication log for rubric boundary-cell decisions.

Records Rock's per-cell adjudications per HANDOFF §5.2. Boundary cells
that are genuine 50/50 calls carry ``decision="ambiguous-both-labels"``
and propagate as label uncertainty into the measurement model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.prereg.rubric import Rubric


_VALID_DECISIONS = frozenset({"ambiguous-both-labels"})
_RESOLVED_PREFIX = "resolved:"


@dataclass(frozen=True, slots=True)
class AdjudicationEntry:
    """A single boundary-cell adjudication decision."""

    entry_id_a: str
    entry_id_b: str
    decision: str  # "resolved:<winner_entry_id>" or "ambiguous-both-labels"
    rationale: str
    adjudicator: str
    date: str  # ISO 8601

    def __post_init__(self) -> None:
        if (
            self.decision not in _VALID_DECISIONS
            and not self.decision.startswith(_RESOLVED_PREFIX)
        ):
            raise ValueError(
                f"invalid decision format: {self.decision!r}. "
                f"Must be 'resolved:<entry_id>' or 'ambiguous-both-labels'."
            )
        if self.decision.startswith(_RESOLVED_PREFIX):
            winner = self.decision[len(_RESOLVED_PREFIX):]
            if winner not in {self.entry_id_a, self.entry_id_b}:
                raise ValueError(
                    f"resolved entry {winner!r} not in adjudicated pair "
                    f"({self.entry_id_a}, {self.entry_id_b})"
                )


@dataclass(frozen=True, slots=True)
class AdjudicationLog:
    """Complete adjudication log bound to a rubric hash."""

    rubric_hash: str
    entries: tuple[AdjudicationEntry, ...]

    def validate_coverage(self, rubric: Rubric) -> None:
        """Verify every boundary rule pair has an adjudication entry."""
        boundary_pairs: set[tuple[str, str]] = set()
        for entry in rubric.entries:
            for br in entry.boundary_rules:
                a, b = sorted([entry.entry_id, br.adjacent_entry_id])
                boundary_pairs.add((a, b))
        adjudicated: set[tuple[str, str]] = set()
        for ae in self.entries:
            a, b = sorted([ae.entry_id_a, ae.entry_id_b])
            adjudicated.add((a, b))
        missing = boundary_pairs - adjudicated
        if missing:
            raise ValueError(
                f"boundary pairs without adjudication: {sorted(missing)}"
            )
