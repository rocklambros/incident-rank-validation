"""Canonical data types for the incident-rank-validation engine.

Every corpus adapter emits ``IncidentRecord`` instances.  The engine never sees
a source schema — all adapter-specific details are normalised here before any
analysis takes place.

See HANDOFF §5.1 for the canonical-record contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

__all__ = [
    "IncidentRecord",
    "BiasProfile",
    "StratumSize",
    "make_stratum_size",
    "EntryDefinition",
]

# ---------------------------------------------------------------------------
# StratumSize
# ---------------------------------------------------------------------------

StratumSize = NewType("StratumSize", int)
"""Positive integer that represents EXPOSURE (the Poisson-rate denominator).

The model interprets stratum_size as the *exposure* of a stratum (e.g. person-
years of observation, or document count), not as the observed incident count.
This is why it must be strictly positive.  The constraint stratum_size >=
observed_count is enforced downstream (Task 25), not here.
"""


def make_stratum_size(n: int) -> StratumSize:
    """Validate and wrap *n* as a :class:`StratumSize`.

    Raises
    ------
    ValueError
        If *n* is not strictly positive.
    """
    if n <= 0:
        raise ValueError(f"stratum size must be positive, got {n}")
    return StratumSize(n)


# ---------------------------------------------------------------------------
# IncidentRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IncidentRecord:
    """A single normalised incident emitted by a corpus adapter.

    Fields
    ------
    id
        Unique incident identifier (adapter-scoped).
    date
        ISO 8601 date string ``YYYY-MM-DD``.
    text
        Title, description, and impact concatenated into one string.
    severity
        Normalised severity label (e.g. ``"Critical"``, ``"High"``) or
        ``None`` when unknown.  A ``"Medium"`` default in the source corpus is
        treated as an artifact of the original dataset's convention rather than
        ground truth (HANDOFF §3).
    source_class
        Coarse record category, e.g. ``"cve"``, ``"advisory"``,
        ``"harm-report"``.
    corpus_stratum
        Sub-corpus stratum membership, e.g. ``"security"``, ``"ai-harm"``.
    quality
        Curation level: ``"curated"``, ``"reviewed"``, or ``"auto"``.
    native_labels
        Original labels from the source corpus.  These are non-authoritative
        and are *never* used as a join key or ground-truth signal (HANDOFF §4).
    source_url
        Resolvable URL pointing to the original record.
    """

    id: str
    date: str
    text: str
    severity: str | None
    source_class: str
    corpus_stratum: str
    quality: str
    native_labels: tuple[str, ...]
    source_url: str


# ---------------------------------------------------------------------------
# BiasProfile
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BiasProfile:
    """Per-stratum declaration of known selection bias and contamination.

    Fields
    ------
    stratum
        The stratum identifier this profile covers.
    description
        Human-readable description of the selection bias affecting this
        stratum.
    known_blind_spots
        Entry identifiers or phenomena that are invisible to this stratum's
        sampling frame.
    contamination_description
        Description of known contamination in this stratum, e.g.
        ``"bare LLM03 default seed"``.
    quarantine_rule
        The rule applied to quarantine contaminated records, e.g.
        ``"drop bare ['LLM03'] CVE singletons"``.
    """

    stratum: str
    description: str
    known_blind_spots: tuple[str, ...]
    contamination_description: str
    quarantine_rule: str


# ---------------------------------------------------------------------------
# EntryDefinition
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EntryDefinition:
    """A taxonomy entry with its canonical name and frame-visibility flag.

    Fields
    ------
    entry_id
        Canonical entry identifier, e.g. ``"LLM01"``, ``"STR01"``.
    name
        Human-readable canonical name for this entry.
    frame_blind
        ``True`` if this entry is invisible to the corpus sampling frame
        (HANDOFF §3 F-frame).  Defaults to ``False``.
    """

    entry_id: str
    name: str
    frame_blind: bool = field(default=False)
