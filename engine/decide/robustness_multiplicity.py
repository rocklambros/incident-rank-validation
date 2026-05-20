"""Robustness-spec cherry-picking + direction-consistency (M18 / HANDOFF §6.11(g))."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FlagDirection(Enum):
    """Direction of vote-vs-incident tier mismatch for a flagged entry."""

    VOTE_OVER_RANKS = "vote_over_ranks"
    VOTE_UNDER_RANKS = "vote_under_ranks"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class FlagFinding:
    """Per-entry flag: vote tier differs from incident tier with given probability."""

    entry_id: str
    direction: FlagDirection
    probability: float  # P(incident tier != vote tier)


@dataclass(frozen=True, slots=True)
class SpecResult:
    """Concordance result for one robustness spec."""

    spec_name: str
    weighted_kappa_median: float | None
    weighted_kappa_ci: tuple[float, float] | None
    flags: tuple[FlagFinding, ...]


@dataclass(frozen=True, slots=True)
class RobustnessSpread:
    """Primary spec plus robustness alternates for cherry-pick detection."""

    primary: SpecResult
    robustness: tuple[SpecResult, ...]

    @property
    def kappa_range(self) -> tuple[float, float] | None:
        """Min/max kappa across primary + all robustness specs."""
        kappas = [self.primary.weighted_kappa_median] + [
            s.weighted_kappa_median for s in self.robustness
        ]
        valid: list[float] = [k for k in kappas if k is not None]
        return None if not valid else (min(valid), max(valid))

    @property
    def spread(self) -> float | None:
        """Range of kappa values (max - min)."""
        r = self.kappa_range
        return None if r is None else r[1] - r[0]

    def is_consistent_in_direction(self) -> bool:
        """M18: do all specs agree on per-entry flag direction?"""
        all_specs = [self.primary, *list(self.robustness)]
        per_entry_dirs: dict[str, set[FlagDirection]] = {}
        for spec in all_specs:
            for f in spec.flags:
                per_entry_dirs.setdefault(f.entry_id, set()).add(f.direction)
        for dirs in per_entry_dirs.values():
            non_indet = {d for d in dirs if d != FlagDirection.INDETERMINATE}
            if len(non_indet) > 1:
                return False
        return True
