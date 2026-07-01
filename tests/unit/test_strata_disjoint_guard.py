"""Tests for the F8 strata-disjoint incident guard (U2-6).

Covers:
- PASSING: legitimate multi-stratum-disjoint entry (critical anti-false-positive)
- RAISING: same incident_id in two strata
- RAISING: stratum repeated in entry_strata[e]
- RAISING: empty stratum population for a multi-stratum entry
- Graceful degradation: empty labeled_incidents list (check.py absent-file path)
"""
import pytest

from engine.verify.strata_guard import (
    StrataOverlapError,
    check_strata_disjoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _labeled(rows: list[tuple[str, str, str]]) -> list[dict[str, object]]:
    """Build minimal labeled_incidents rows from (incident_id, entry_id, stratum)."""
    return [{"incident_id": iid, "entry_id": eid, "stratum": s} for iid, eid, s in rows]


# ---------------------------------------------------------------------------
# PASSING: legitimate multi-stratum-disjoint cases
# ---------------------------------------------------------------------------


def test_multi_stratum_disjoint_does_not_raise() -> None:
    """Entry spanning 2 strata with fully disjoint incidents MUST NOT raise.

    This is the critical anti-false-positive test.  9/20 real OWASP-LLM entries
    span both 'security' and 'ai-harm'.  A naive len(entry_strata[e]) > 1 gate
    would false-positive on every one of them — that design was explicitly
    rejected by the U2-6 premortem finding P.
    """
    labeled = _labeled([
        ("INC-001", "E1", "security"),
        ("INC-002", "E1", "security"),
        ("INC-003", "E1", "ai-harm"),   # E1 legitimately spans both strata
        ("INC-004", "E2", "security"),
        ("INC-005", "E2", "ai-harm"),   # E2 also spans both strata
        ("INC-006", "E3", "ai-harm"),   # E3 is single-stratum
    ])
    entry_strata: dict[str, tuple[str, ...]] = {
        "E1": ("ai-harm", "security"),
        "E2": ("ai-harm", "security"),
        "E3": ("ai-harm",),
    }
    # Must not raise — this is the legitimate multi-stratum case.
    check_strata_disjoint(labeled, entry_strata)


def test_single_stratum_cycle_does_not_raise() -> None:
    """Single-stratum cycle (common baseline): must not raise."""
    labeled = _labeled([
        ("INC-001", "E1", "default"),
        ("INC-002", "E2", "default"),
        ("INC-003", "E1", "default"),
    ])
    entry_strata: dict[str, tuple[str, ...]] = {
        "E1": ("default",),
        "E2": ("default",),
    }
    check_strata_disjoint(labeled, entry_strata)


def test_whitespace_canonicalization_does_not_produce_false_positive() -> None:
    """Leading/trailing whitespace is stripped from incident_id before comparison."""
    labeled: list[dict[str, object]] = [
        {"incident_id": "  INC-001  ", "entry_id": "E1", "stratum": "security"},
        {"incident_id": "INC-001", "entry_id": "E1", "stratum": "security"},
    ]
    entry_strata: dict[str, tuple[str, ...]] = {"E1": ("security",)}
    # Same canonical id, same stratum → no violation.
    check_strata_disjoint(labeled, entry_strata)


# ---------------------------------------------------------------------------
# RAISING: same incident_id in two strata
# ---------------------------------------------------------------------------


def test_same_incident_in_two_strata_raises() -> None:
    """Incident appearing in two different strata' populations MUST raise."""
    labeled = _labeled([
        ("INC-001", "E1", "security"),
        ("INC-001", "E1", "ai-harm"),   # INC-001 in BOTH strata — violation
        ("INC-002", "E2", "ai-harm"),
    ])
    entry_strata: dict[str, tuple[str, ...]] = {
        "E1": ("ai-harm", "security"),
        "E2": ("ai-harm",),
    }
    with pytest.raises(StrataOverlapError, match="INC-001"):
        check_strata_disjoint(labeled, entry_strata)


def test_same_incident_with_whitespace_in_two_strata_raises() -> None:
    """Canonicalization: '  INC-001  ' and 'INC-001' are the same id across strata."""
    labeled: list[dict[str, object]] = [
        {"incident_id": "  INC-001  ", "entry_id": "E1", "stratum": "security"},
        {"incident_id": "INC-001", "entry_id": "E1", "stratum": "ai-harm"},
    ]
    entry_strata: dict[str, tuple[str, ...]] = {"E1": ("ai-harm", "security")}
    with pytest.raises(StrataOverlapError, match="INC-001"):
        check_strata_disjoint(labeled, entry_strata)


# ---------------------------------------------------------------------------
# RAISING: stratum repeated in entry_strata[e]
# ---------------------------------------------------------------------------


def test_stratum_repeated_in_entry_strata_raises() -> None:
    """Stratum listed twice in entry_strata[e] MUST raise.

    A repeated stratum doubles the Σsize exposure term in the concordance
    incidence computation (concordance.py line 84).
    """
    labeled = _labeled([
        ("INC-001", "E1", "security"),
        ("INC-002", "E1", "security"),
    ])
    entry_strata: dict[str, tuple[str, ...]] = {
        "E1": ("security", "security"),  # "security" repeated — violation
    }
    with pytest.raises(StrataOverlapError, match="security"):
        check_strata_disjoint(labeled, entry_strata)


# ---------------------------------------------------------------------------
# RAISING: empty stratum population for a multi-stratum entry
# ---------------------------------------------------------------------------


def test_empty_stratum_population_for_multi_stratum_entry_raises() -> None:
    """Multi-stratum entry whose one stratum has no incidents MUST raise.

    Declaring a multi-stratum entry with an empty stratum makes the Σsize term
    malformed — the population size is 0 but the entry claims exposure there.
    """
    labeled = _labeled([
        ("INC-001", "E1", "security"),
        ("INC-002", "E1", "security"),
        # No incidents in "ai-harm" — stratum is absent from labeled_incidents.
    ])
    entry_strata: dict[str, tuple[str, ...]] = {
        "E1": ("ai-harm", "security"),  # declares "ai-harm" but it's empty
    }
    with pytest.raises(StrataOverlapError, match="ai-harm"):
        check_strata_disjoint(labeled, entry_strata)


# ---------------------------------------------------------------------------
# Graceful degradation: empty labeled_incidents (verify/check.py absent-file path)
# ---------------------------------------------------------------------------


def test_empty_labeled_incidents_single_stratum_does_not_raise() -> None:
    """Empty labeled_incidents with single-stratum entry: must not crash.

    Simulates the verify/check.py path when labeled_incidents.json is absent
    (the outer labeled_path.exists() guard causes the call to be skipped entirely;
    this test verifies the helper itself is safe with an empty list).
    """
    entry_strata: dict[str, tuple[str, ...]] = {
        "E1": ("default",),
    }
    # No incidents, no multi-stratum entry → no violation possible.
    check_strata_disjoint([], entry_strata)


def test_empty_labeled_and_empty_entry_strata_does_not_raise() -> None:
    """Completely empty inputs: must not crash."""
    check_strata_disjoint([], {})


def test_rows_with_missing_incident_id_are_skipped() -> None:
    """Rows lacking 'incident_id' are silently skipped (defensive)."""
    labeled: list[dict[str, object]] = [
        {"entry_id": "E1", "stratum": "security"},          # no incident_id key
        {"incident_id": "", "entry_id": "E1", "stratum": "security"},  # empty string
        {"incident_id": "   ", "entry_id": "E1", "stratum": "security"},  # whitespace only
        {"incident_id": "INC-001", "entry_id": "E1", "stratum": "security"},
    ]
    entry_strata: dict[str, tuple[str, ...]] = {"E1": ("security",)}
    check_strata_disjoint(labeled, entry_strata)
