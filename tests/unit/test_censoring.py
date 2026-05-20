"""Tests for engine/model/censoring.py — frame-blind partitioning."""

from __future__ import annotations

import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.censoring import (
    MeasurabilityVerdict,
    partition_entries,
)
from engine.schema import EntryDefinition

# ---------------------------------------------------------------------------
# Fixtures — mirror SyntheticAdapter's E01-E06 definitions
# ---------------------------------------------------------------------------

_ENTRIES: tuple[EntryDefinition, ...] = (
    EntryDefinition(entry_id="E01", name="Entry One", frame_blind=False),
    EntryDefinition(entry_id="E02", name="Entry Two", frame_blind=False),
    EntryDefinition(entry_id="E03", name="Entry Three", frame_blind=False),
    EntryDefinition(entry_id="E04", name="Entry Four", frame_blind=False),
    EntryDefinition(entry_id="E05", name="Entry Five", frame_blind=False),
    EntryDefinition(entry_id="E06", name="Entry Six", frame_blind=True),
)

_STRATA = ("stratum_a", "stratum_b")


def _high_recall_calibration() -> Calibration:
    """All non-frame-blind entries have recall mean ~0.9 across both strata."""
    recall: dict[tuple[str, str], BetaPosterior] = {}
    precision: dict[tuple[str, str], BetaPosterior] = {}
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        for stratum in _STRATA:
            # alpha=9, beta=1 → mean=0.9
            recall[(eid, stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
            precision[(eid, stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
    return Calibration(recall=recall, precision=precision)


def _low_recall_calibration() -> Calibration:
    """All non-frame-blind entries have recall mean ~0.05 across both strata."""
    recall: dict[tuple[str, str], BetaPosterior] = {}
    precision: dict[tuple[str, str], BetaPosterior] = {}
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        for stratum in _STRATA:
            # alpha=1, beta=19 → mean=0.05
            recall[(eid, stratum)] = BetaPosterior(alpha=1.0, beta=19.0)
            precision[(eid, stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
    return Calibration(recall=recall, precision=precision)


def _mixed_recall_calibration() -> Calibration:
    """E01-E04 high recall (0.9), E05 low recall (0.05)."""
    recall: dict[tuple[str, str], BetaPosterior] = {}
    precision: dict[tuple[str, str], BetaPosterior] = {}
    for eid in ("E01", "E02", "E03", "E04"):
        for stratum in _STRATA:
            recall[(eid, stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
            precision[(eid, stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
    for stratum in _STRATA:
        recall[("E05", stratum)] = BetaPosterior(alpha=1.0, beta=19.0)
        precision[("E05", stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
    return Calibration(recall=recall, precision=precision)


# ---------------------------------------------------------------------------
# Test 1: frame-blind entries always land in frame_blind regardless of calib
# ---------------------------------------------------------------------------

def test_frame_blind_goes_to_frame_blind_partition_no_calibration() -> None:
    result = partition_entries(_ENTRIES, calibration=None)
    assert "E06" in result.frame_blind
    assert "E06" not in result.measurable
    assert "E06" not in result.classifier_blind


def test_frame_blind_goes_to_frame_blind_partition_with_calibration() -> None:
    result = partition_entries(_ENTRIES, calibration=_high_recall_calibration())
    assert "E06" in result.frame_blind
    assert "E06" not in result.measurable
    assert "E06" not in result.classifier_blind


def test_frame_blind_verdict() -> None:
    result = partition_entries(_ENTRIES, calibration=None)
    assert result.verdicts["E06"] is MeasurabilityVerdict.FRAME_BLIND_UNMEASURABLE


# ---------------------------------------------------------------------------
# Test 2: with calibration — high recall → measurable, low recall → classifier_blind
# ---------------------------------------------------------------------------

def test_high_recall_entries_are_measurable() -> None:
    result = partition_entries(_ENTRIES, calibration=_high_recall_calibration())
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert eid in result.measurable, f"{eid} should be measurable"
        assert result.verdicts[eid] is MeasurabilityVerdict.MEASURABLE


def test_low_recall_entries_are_classifier_blind() -> None:
    result = partition_entries(_ENTRIES, calibration=_low_recall_calibration())
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert eid in result.classifier_blind, f"{eid} should be classifier_blind"
        assert result.verdicts[eid] is MeasurabilityVerdict.CLASSIFIER_BLIND_BOUNDED


def test_mixed_recall_splits_correctly() -> None:
    result = partition_entries(_ENTRIES, calibration=_mixed_recall_calibration())
    for eid in ("E01", "E02", "E03", "E04"):
        assert eid in result.measurable
    assert "E05" in result.classifier_blind
    assert "E06" in result.frame_blind


# ---------------------------------------------------------------------------
# Test 3: without calibration — non-frame-blind entries default to measurable
# ---------------------------------------------------------------------------

def test_no_calibration_non_frame_blind_are_measurable() -> None:
    result = partition_entries(_ENTRIES, calibration=None)
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert eid in result.measurable, f"{eid} should default to measurable"
        assert result.verdicts[eid] is MeasurabilityVerdict.MEASURABLE
    assert len(result.classifier_blind) == 0


# ---------------------------------------------------------------------------
# Test 4: partitions are disjoint — every entry in exactly one partition
# ---------------------------------------------------------------------------

def test_partitions_are_disjoint_no_calibration() -> None:
    result = partition_entries(_ENTRIES, calibration=None)
    all_ids = [e.entry_id for e in _ENTRIES]
    combined = list(result.measurable) + list(result.classifier_blind) + list(result.frame_blind)
    assert sorted(combined) == sorted(all_ids), "every entry must appear in exactly one partition"
    # no duplicates
    assert len(combined) == len(set(combined))


def test_partitions_are_disjoint_with_calibration() -> None:
    result = partition_entries(_ENTRIES, calibration=_mixed_recall_calibration())
    all_ids = [e.entry_id for e in _ENTRIES]
    combined = list(result.measurable) + list(result.classifier_blind) + list(result.frame_blind)
    assert sorted(combined) == sorted(all_ids)
    assert len(combined) == len(set(combined))


# ---------------------------------------------------------------------------
# Test 5: verdicts dict contains all entries
# ---------------------------------------------------------------------------

def test_verdicts_contains_all_entries() -> None:
    for calibration in (None, _high_recall_calibration(), _mixed_recall_calibration()):
        result = partition_entries(_ENTRIES, calibration=calibration)
        for entry in _ENTRIES:
            assert entry.entry_id in result.verdicts, (
                f"{entry.entry_id} missing from verdicts"
            )
        assert len(result.verdicts) == len(_ENTRIES)


# ---------------------------------------------------------------------------
# Test 6: recall_floor parameter — lowering it moves entries from
#          classifier_blind to measurable
# ---------------------------------------------------------------------------

def test_recall_floor_moves_entries_from_classifier_blind_to_measurable() -> None:
    # With recall_floor=0.1 (default), E05 recall=0.05 → classifier_blind
    result_default = partition_entries(
        _ENTRIES, calibration=_mixed_recall_calibration(), recall_floor=0.1
    )
    assert "E05" in result_default.classifier_blind

    # Lower the floor below E05's recall (0.05) → E05 becomes measurable
    result_low_floor = partition_entries(
        _ENTRIES, calibration=_mixed_recall_calibration(), recall_floor=0.04
    )
    assert "E05" in result_low_floor.measurable
    assert "E05" not in result_low_floor.classifier_blind


def test_recall_floor_raising_moves_entries_to_classifier_blind() -> None:
    # With recall_floor=0.95, E01-E05 recall=0.9 < 0.95 → all classifier_blind
    result = partition_entries(
        _ENTRIES, calibration=_high_recall_calibration(), recall_floor=0.95
    )
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert eid in result.classifier_blind
    # E06 still frame_blind
    assert "E06" in result.frame_blind


# ---------------------------------------------------------------------------
# Test 7: CensoringResult is frozen (immutable)
# ---------------------------------------------------------------------------

def test_censoring_result_is_frozen() -> None:
    result = partition_entries(_ENTRIES, calibration=None)
    with pytest.raises((AttributeError, TypeError)):
        result.measurable = ()  # type: ignore[misc]


def test_censoring_result_frozen_tuple_fields() -> None:
    result = partition_entries(_ENTRIES, calibration=None)
    assert isinstance(result.measurable, tuple)
    assert isinstance(result.classifier_blind, tuple)
    assert isinstance(result.frame_blind, tuple)
