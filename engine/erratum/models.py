"""Erratum data model (HANDOFF §5.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Erratum:
    """A post-publication correction to a validation cycle."""

    cycle_id: str
    erratum_number: int
    title: str
    description: str
    impact: str  # "flips_measurability" | "flips_flag" | "cosmetic"
    issued_at: str  # ISO 8601
