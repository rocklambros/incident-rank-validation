"""Tests for the F8 strata-disjoint incident guard (U2-6).

Covers:
- PASSING: legitimate multi-stratum-disjoint entry (critical anti-false-positive)
- RAISING: same incident_id in two strata
- RAISING: stratum repeated in entry_strata[e]
- RAISING: empty stratum population for a multi-stratum entry
- Graceful degradation: empty labeled_incidents list (check.py absent-file path)
- Real-2026 regression: F8 guard does NOT raise on the committed 6,639-row
  labeled_incidents.json (9/20 entries legitimately span both strata).
- Canonicalization alignment (U2 fix #3): whitespace stratum + blank-id row that
  the pipeline accepts must NOT false-positive the guard.
"""
import json
from collections import defaultdict
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Real-2026 F8 regression: committed labeled_incidents.json must not raise
# ---------------------------------------------------------------------------


def test_real_2026_labeled_incidents_no_false_positive() -> None:
    """F8 guard MUST NOT raise on the committed 2026 labeled_incidents.json.

    This is a regression lock for the 6,639-row real corpus (9/20 entries
    legitimately span both 'security' and 'ai-harm' strata with disjoint
    incident populations).  A naive len(entry_strata[e]) > 1 guard would
    false-positive on all 9 multi-stratum entries; this test permanently
    locks in the 'no false-positive on legitimate multi-stratum entries'
    property against the actual committed data.
    """
    labeled_path = (
        Path(__file__).parent.parent.parent
        / "projects" / "owasp-llm" / "cycles" / "2026" / "classify"
        / "labeled_incidents.json"
    )
    if not labeled_path.exists():
        pytest.skip(f"Real labeled_incidents.json not found: {labeled_path}")

    labeled: list[dict[str, object]] = json.loads(labeled_path.read_text())

    # Build entry_strata identically to _build_strata in engine/verify/check.py:
    # for each entry_id, collect all strata that entry appears in (sorted tuple).
    entry_strata_sets: dict[str, set[str]] = defaultdict(set)
    for item in labeled:
        eid = str(item.get("entry_id", ""))
        stratum = str(item.get("stratum", "default"))
        entry_strata_sets[eid].add(stratum)
    entry_strata: dict[str, tuple[str, ...]] = {
        e: tuple(sorted(ss)) for e, ss in entry_strata_sets.items()
    }

    # Must NOT raise — 9/20 entries legitimately span both strata with
    # fully disjoint incident populations.
    check_strata_disjoint(labeled, entry_strata)

    # Verify the fixture has the expected multi-stratum entries so a future
    # data change that removes them is detected.
    multi_stratum_entries = [e for e, ss in entry_strata.items() if len(ss) > 1]
    assert len(multi_stratum_entries) == 9, (
        f"Expected 9 multi-stratum entries in the 2026 corpus; "
        f"found {len(multi_stratum_entries)}: {sorted(multi_stratum_entries)}"
    )


# ---------------------------------------------------------------------------
# Canonicalization alignment (U2 fix #3): whitespace stratum + blank-id row
# ---------------------------------------------------------------------------


def test_whitespace_stratum_and_blank_id_row_not_false_positive() -> None:
    """Guard must NOT false-positive on data the pipeline accepts (U2 fix #3).

    Two mismatch scenarios corrected by the fix:

    (a) Whitespace stratum: a row has stratum=' security ' (with spaces).
        _build_strata / _build_counts_from_labeled do NOT strip, so entry_strata
        uses ' security ' as the key.  Before the fix the guard stripped strata,
        so stratum_incident_sets used 'security' — assertion 4 looked up ' security '
        and found nothing → false-positive raise.  After the fix the guard also does
        NOT strip, so the key matches.

    (b) Blank incident_id as the ONLY row in a stratum: the pipeline counts this row
        toward stratum_doc_counts, making the stratum appear non-empty in the exposure
        term.  Before the fix the guard skipped blank-id rows entirely, leaving the
        stratum absent from stratum_incident_sets → assertion 4 raised on the
        multi-stratum entry.  After the fix the stratum is registered (key present),
        and assertion 4 uses key-existence rather than set-emptiness, so it passes.
    """
    labeled: list[dict[str, object]] = [
        # (a) whitespace stratum — must not be stripped by the guard
        {"incident_id": "INC-001", "entry_id": "E1", "stratum": " security "},
        {"incident_id": "INC-002", "entry_id": "E1", "stratum": " security "},
        # (b) blank incident_id as the only row in 'ai-harm' — pipeline counts it;
        #     guard must register the stratum key and pass assertion 4.
        {"incident_id": "", "entry_id": "E1", "stratum": "ai-harm"},
    ]
    # entry_strata as _build_strata would produce (no strip on stratum values):
    entry_strata: dict[str, tuple[str, ...]] = {"E1": ("ai-harm", " security ")}
    # Must NOT raise on either the whitespace stratum or the blank-id-only stratum.
    check_strata_disjoint(labeled, entry_strata)
