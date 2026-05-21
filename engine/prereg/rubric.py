"""Frozen classification rubric per HANDOFF §5.2 Artifact 1.

The rubric is the primary pre-registration artifact: per-entry classification
rules frozen, hash-locked, and independently reviewed before any concordance
number exists.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundaryRule:
    """Pairwise boundary rule between two taxonomy entries."""

    adjacent_entry_id: str
    rule: str
    is_ambiguous: bool


def _to_serializable(obj: object) -> object:
    """Recursively convert frozen dataclasses and tuples to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_serializable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, tuple):
        return [_to_serializable(item) for item in obj]
    return obj


@dataclass(frozen=True, slots=True)
class RubricEntry:
    """Per-entry classification rubric per HANDOFF §5.2 Artifact 1."""

    entry_id: str
    canonical_name: str
    in_scope: str
    exclusions: tuple[str, ...]
    boundary_rules: tuple[BoundaryRule, ...]
    positive_indicators: tuple[str, ...]
    negative_indicators: tuple[str, ...]
    co_occurrence_pairs: tuple[tuple[str, str], ...]
    is_rollup_candidate: bool
    rolled_into: str | None


@dataclass(frozen=True, slots=True)
class Rubric:
    """Complete frozen classification rubric."""

    cycle_id: str
    version: str
    entries: tuple[RubricEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dict for serialization and hashing."""
        return _to_serializable(self)  # type: ignore[return-value]

    def compute_hash(self) -> str:
        """SHA-256 of canonical JSON representation."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate_completeness(
        self,
        expected_entry_ids: set[str],
        *,
        no_adjacency_attested: set[str] | None = None,
    ) -> None:
        """Verify all expected entries are present and required fields non-empty.

        Parameters
        ----------
        no_adjacency_attested:
            Entry IDs for which the drafter attests no genuine adjacency exists.
            These are allowed to have empty boundary_rules without raising.
            A warning is logged instead.
        """
        attested = no_adjacency_attested or set()
        actual = {e.entry_id for e in self.entries}
        missing = expected_entry_ids - actual
        extra = actual - expected_entry_ids
        if missing:
            raise ValueError(f"rubric missing entries: {sorted(missing)}")
        if extra:
            raise ValueError(f"rubric has unexpected entries: {sorted(extra)}")
        for entry in self.entries:
            if not entry.in_scope.strip():
                raise ValueError(f"{entry.entry_id}: in_scope is empty")
            if not entry.positive_indicators:
                raise ValueError(f"{entry.entry_id}: positive_indicators is empty")
            if not entry.negative_indicators:
                raise ValueError(f"{entry.entry_id}: negative_indicators is empty")
            if not entry.is_rollup_candidate and not entry.boundary_rules:
                if entry.entry_id in attested:
                    import logging
                    logging.getLogger(__name__).warning(
                        "%s: no boundary_rules (no-adjacency attested by drafter)",
                        entry.entry_id,
                    )
                else:
                    raise ValueError(
                        f"{entry.entry_id}: boundary_rules is empty for non-rollup entry "
                        f"(pass entry_id in no_adjacency_attested if no genuine adjacency exists)"
                    )

    def validate_boundary_rules(self) -> None:
        """Verify boundary rules are paired: if A->B exists, B->A must exist."""
        pairs: set[tuple[str, str]] = set()
        entry_ids = {e.entry_id for e in self.entries}
        for entry in self.entries:
            for br in entry.boundary_rules:
                if br.adjacent_entry_id not in entry_ids:
                    raise ValueError(
                        f"{entry.entry_id}: boundary rule references unknown "
                        f"entry {br.adjacent_entry_id}"
                    )
                pairs.add((entry.entry_id, br.adjacent_entry_id))
        for a, b in pairs:
            if (b, a) not in pairs:
                raise ValueError(
                    f"boundary rule {a}->{b} exists but {b}->{a} is missing"
                )

    def validate_co_occurrences(self) -> None:
        """Verify all co_occurrence_pairs reference valid entry IDs."""
        entry_ids = {e.entry_id for e in self.entries}
        for entry in self.entries:
            for pair in entry.co_occurrence_pairs:
                for eid in pair:
                    if eid not in entry_ids:
                        raise ValueError(
                            f"{entry.entry_id}: co_occurrence_pair references "
                            f"unknown entry {eid}"
                        )

    def rollup_candidates(self) -> tuple[RubricEntry, ...]:
        """Return only the rolled-up candidate entries."""
        return tuple(e for e in self.entries if e.is_rollup_candidate)

    def standalone_entries(self) -> tuple[RubricEntry, ...]:
        """Return entries that are NOT rollup candidates."""
        return tuple(e for e in self.entries if not e.is_rollup_candidate)
