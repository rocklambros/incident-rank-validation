"""Unit tests for engine.decide.robustness_multiplicity — direction-consistency (M18)."""

from __future__ import annotations

from engine.decide.robustness_multiplicity import (
    FlagDirection,
    FlagFinding,
    RobustnessSpread,
    SpecResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flag(entry: str, direction: FlagDirection, p: float = 0.8) -> FlagFinding:
    return FlagFinding(entry_id=entry, direction=direction, probability=p)


def _spec(
    name: str,
    kappa: float | None,
    flags: tuple[FlagFinding, ...] = (),
) -> SpecResult:
    ci = (kappa - 0.1, kappa + 0.1) if kappa is not None else None
    return SpecResult(
        spec_name=name, weighted_kappa_median=kappa, weighted_kappa_ci=ci, flags=flags,
    )


# ---------------------------------------------------------------------------
# Tests: is_consistent_in_direction
# ---------------------------------------------------------------------------


class TestDirectionConsistency:
    def test_consistent_when_all_same_direction(self) -> None:
        """All specs flag the same entry in the same direction -> consistent."""
        primary = _spec("primary", 0.6, flags=(
            _flag("E1", FlagDirection.VOTE_OVER_RANKS),
        ))
        alt = _spec("alt_1", 0.55, flags=(
            _flag("E1", FlagDirection.VOTE_OVER_RANKS),
        ))
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.is_consistent_in_direction() is True

    def test_inconsistent_when_directions_conflict(self) -> None:
        """One spec says OVER, another says UNDER for same entry -> inconsistent."""
        primary = _spec("primary", 0.6, flags=(
            _flag("E1", FlagDirection.VOTE_OVER_RANKS),
        ))
        alt = _spec("alt_1", 0.55, flags=(
            _flag("E1", FlagDirection.VOTE_UNDER_RANKS),
        ))
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.is_consistent_in_direction() is False

    def test_indeterminate_does_not_conflict(self) -> None:
        """INDETERMINATE should not count as conflicting with a real direction."""
        primary = _spec("primary", 0.6, flags=(
            _flag("E1", FlagDirection.VOTE_OVER_RANKS),
        ))
        alt = _spec("alt_1", 0.55, flags=(
            _flag("E1", FlagDirection.INDETERMINATE),
        ))
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.is_consistent_in_direction() is True

    def test_no_flags_is_consistent(self) -> None:
        """No flags at all -> trivially consistent."""
        primary = _spec("primary", 0.7)
        alt = _spec("alt_1", 0.65)
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.is_consistent_in_direction() is True

    def test_different_entries_flagged_is_consistent(self) -> None:
        """Different entries flagged in different specs -> consistent (no conflict)."""
        primary = _spec("primary", 0.6, flags=(
            _flag("E1", FlagDirection.VOTE_OVER_RANKS),
        ))
        alt = _spec("alt_1", 0.55, flags=(
            _flag("E2", FlagDirection.VOTE_UNDER_RANKS),
        ))
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.is_consistent_in_direction() is True

    def test_multiple_robustness_specs(self) -> None:
        """Three specs: all agree on direction -> consistent."""
        primary = _spec("primary", 0.6, flags=(
            _flag("E1", FlagDirection.VOTE_OVER_RANKS),
            _flag("E2", FlagDirection.VOTE_UNDER_RANKS),
        ))
        alt1 = _spec("alt_1", 0.55, flags=(
            _flag("E1", FlagDirection.VOTE_OVER_RANKS),
        ))
        alt2 = _spec("alt_2", 0.58, flags=(
            _flag("E2", FlagDirection.VOTE_UNDER_RANKS),
        ))
        spread = RobustnessSpread(primary=primary, robustness=(alt1, alt2))

        assert spread.is_consistent_in_direction() is True


# ---------------------------------------------------------------------------
# Tests: kappa_range and spread
# ---------------------------------------------------------------------------


class TestKappaRange:
    def test_kappa_range(self) -> None:
        primary = _spec("primary", 0.6)
        alt = _spec("alt_1", 0.4)
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.kappa_range == (0.4, 0.6)

    def test_spread_value(self) -> None:
        primary = _spec("primary", 0.6)
        alt = _spec("alt_1", 0.4)
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.spread is not None
        assert abs(spread.spread - 0.2) < 1e-9

    def test_none_kappas_skipped(self) -> None:
        primary = _spec("primary", 0.6)
        alt = _spec("alt_1", None)
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.kappa_range == (0.6, 0.6)
        assert spread.spread == 0.0

    def test_all_none_returns_none(self) -> None:
        primary = _spec("primary", None)
        alt = _spec("alt_1", None)
        spread = RobustnessSpread(primary=primary, robustness=(alt,))

        assert spread.kappa_range is None
        assert spread.spread is None
