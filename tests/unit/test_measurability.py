"""Tests for engine/decide/measurability.py — measurability map."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.decide.measurability import MeasurabilityMap, build_measurability_map
from engine.model.censoring import CensoringResult, MeasurabilityVerdict

# ---------------------------------------------------------------------------
# Helpers — shared test fixtures
# ---------------------------------------------------------------------------

_STRATA = ("stratum_a", "stratum_b")


def _make_censoring(
    measurable: tuple[str, ...] = ("E01", "E02", "E03", "E04", "E05"),
    classifier_blind: tuple[str, ...] = (),
    frame_blind: tuple[str, ...] = ("E06",),
) -> CensoringResult:
    verdicts: dict[str, MeasurabilityVerdict] = {}
    for eid in measurable:
        verdicts[eid] = MeasurabilityVerdict.MEASURABLE
    for eid in classifier_blind:
        verdicts[eid] = MeasurabilityVerdict.CLASSIFIER_BLIND_BOUNDED
    for eid in frame_blind:
        verdicts[eid] = MeasurabilityVerdict.FRAME_BLIND_UNMEASURABLE
    return CensoringResult(
        measurable=measurable,
        classifier_blind=classifier_blind,
        frame_blind=frame_blind,
        verdicts=verdicts,
    )


def _high_recall_calibration(
    entry_ids: tuple[str, ...] = ("E01", "E02", "E03", "E04", "E05"),
) -> Calibration:
    """All entries have recall Beta(9,1) → mean=0.9, P(recall>0.1) ≈ 1.0."""
    recall: dict[tuple[str, str], BetaPosterior] = {}
    precision: dict[tuple[str, str], BetaPosterior] = {}
    for eid in entry_ids:
        for stratum in _STRATA:
            recall[(eid, stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
            precision[(eid, stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
    return Calibration(recall=recall, precision=precision)


def _low_recall_calibration(
    entry_ids: tuple[str, ...] = ("E01", "E02", "E03", "E04", "E05"),
) -> Calibration:
    """All entries have recall Beta(1,99) → mean≈0.01, P(recall>0.1) ≈ 0."""
    recall: dict[tuple[str, str], BetaPosterior] = {}
    precision: dict[tuple[str, str], BetaPosterior] = {}
    for eid in entry_ids:
        for stratum in _STRATA:
            recall[(eid, stratum)] = BetaPosterior(alpha=1.0, beta=99.0)
            precision[(eid, stratum)] = BetaPosterior(alpha=9.0, beta=1.0)
    return Calibration(recall=recall, precision=precision)


# ---------------------------------------------------------------------------
# Test 1: frame-blind entries get P=0.0
# ---------------------------------------------------------------------------


def test_frame_blind_entries_get_p_zero_no_calibration() -> None:
    censoring = _make_censoring(frame_blind=("E06",))
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    assert mm.recall_p_above_threshold["E06"] == 0.0


def test_frame_blind_entries_get_p_zero_with_calibration() -> None:
    censoring = _make_censoring(frame_blind=("E06",))
    mm = build_measurability_map(
        censoring, calibration=_high_recall_calibration(), measurability_minimum=3
    )
    assert mm.recall_p_above_threshold["E06"] == 0.0


# ---------------------------------------------------------------------------
# Test 2: with calibration — high-recall entry gets P close to 1.0
# ---------------------------------------------------------------------------


def test_high_recall_entry_p_close_to_one_with_calibration() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(
        censoring,
        calibration=_high_recall_calibration(),
        measurability_minimum=3,
        recall_threshold=0.1,
    )
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert mm.recall_p_above_threshold[eid] > 0.99, (
            f"{eid}: expected P>0.99, got {mm.recall_p_above_threshold[eid]}"
        )


def test_low_recall_entry_p_near_zero_with_calibration() -> None:
    censoring = _make_censoring(
        measurable=(),
        classifier_blind=("E01", "E02", "E03", "E04", "E05"),
        frame_blind=("E06",),
    )
    mm = build_measurability_map(
        censoring,
        calibration=_low_recall_calibration(),
        measurability_minimum=3,
        recall_threshold=0.1,
    )
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert mm.recall_p_above_threshold[eid] < 0.01, (
            f"{eid}: expected P<0.01, got {mm.recall_p_above_threshold[eid]}"
        )


# ---------------------------------------------------------------------------
# Test 3: without calibration — all non-frame-blind get P=1.0
# ---------------------------------------------------------------------------


def test_no_calibration_non_frame_blind_get_p_one() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert mm.recall_p_above_threshold[eid] == 1.0, (
            f"{eid}: expected P=1.0, got {mm.recall_p_above_threshold[eid]}"
        )


def test_no_calibration_classifier_blind_also_get_p_one() -> None:
    censoring = _make_censoring(
        measurable=("E01", "E02"),
        classifier_blind=("E03", "E04", "E05"),
        frame_blind=("E06",),
    )
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=2)
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert mm.recall_p_above_threshold[eid] == 1.0


# ---------------------------------------------------------------------------
# Test 4: coverage_ratio is correct fraction
# ---------------------------------------------------------------------------


def test_coverage_ratio_all_measurable() -> None:
    # 5 measurable, 0 classifier_blind, 1 frame_blind → 5/6
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    assert abs(mm.coverage_ratio - 5 / 6) < 1e-9


def test_coverage_ratio_mixed() -> None:
    # 2 measurable, 3 classifier_blind, 1 frame_blind → 2/6
    censoring = _make_censoring(
        measurable=("E01", "E02"),
        classifier_blind=("E03", "E04", "E05"),
        frame_blind=("E06",),
    )
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=2)
    assert abs(mm.coverage_ratio - 2 / 6) < 1e-9


def test_coverage_ratio_empty() -> None:
    censoring = _make_censoring(measurable=(), classifier_blind=(), frame_blind=())
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=1)
    assert mm.coverage_ratio == 0.0


# ---------------------------------------------------------------------------
# Test 5: below_prereg_minimum True when measurable count < minimum
# ---------------------------------------------------------------------------


def test_below_prereg_minimum_true_when_too_few_measurable() -> None:
    censoring = _make_censoring(measurable=("E01", "E02"), frame_blind=("E06",))
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    assert mm.below_prereg_minimum is True


# ---------------------------------------------------------------------------
# Test 6: below_prereg_minimum False when measurable count >= minimum
# ---------------------------------------------------------------------------


def test_below_prereg_minimum_false_when_enough_measurable() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=5)
    assert mm.below_prereg_minimum is False


def test_below_prereg_minimum_false_at_exact_minimum() -> None:
    censoring = _make_censoring(measurable=("E01", "E02", "E03"), frame_blind=("E06",))
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    assert mm.below_prereg_minimum is False


# ---------------------------------------------------------------------------
# Test 7: to_coverage_json produces valid JSON with sorted keys
# ---------------------------------------------------------------------------


def test_to_coverage_json_is_valid_json() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    raw = mm.to_coverage_json()
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)


def test_to_coverage_json_has_expected_keys() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    parsed = json.loads(mm.to_coverage_json())
    expected_keys = {
        "coverage_ratio",
        "measurable",
        "classifier_blind",
        "frame_blind",
        "below_prereg_minimum",
        "recall_p_above_threshold",
    }
    assert set(parsed.keys()) == expected_keys


def test_to_coverage_json_sorted_keys() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    raw = mm.to_coverage_json()
    parsed = json.loads(raw)
    # Top-level keys sorted
    assert list(parsed.keys()) == sorted(parsed.keys())
    # recall_p_above_threshold keys sorted
    p_keys = list(parsed["recall_p_above_threshold"].keys())
    assert p_keys == sorted(p_keys)


def test_to_coverage_json_ends_with_newline() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    assert mm.to_coverage_json().endswith("\n")


def test_to_coverage_json_measurable_sorted() -> None:
    censoring = _make_censoring(measurable=("E05", "E01", "E03"))
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=2)
    parsed = json.loads(mm.to_coverage_json())
    assert parsed["measurable"] == sorted(parsed["measurable"])


# ---------------------------------------------------------------------------
# Test 8: coverage.json round-trips (write + read back)
# ---------------------------------------------------------------------------


def test_write_coverage_round_trip() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(
        censoring, calibration=_high_recall_calibration(), measurability_minimum=3
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out" / "coverage.json"
        mm.write_coverage(path)
        assert path.exists()
        content = path.read_text()
        parsed = json.loads(content)
        assert parsed["coverage_ratio"] == mm.coverage_ratio
        assert parsed["measurable"] == sorted(mm.measurable)
        assert parsed["classifier_blind"] == sorted(mm.classifier_blind)
        assert parsed["frame_blind"] == sorted(mm.frame_blind)
        assert parsed["below_prereg_minimum"] == mm.below_prereg_minimum


def test_write_coverage_creates_parent_dirs() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        deep_path = Path(tmpdir) / "a" / "b" / "c" / "coverage.json"
        mm.write_coverage(deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# Test 9: MeasurabilityMap is frozen
# ---------------------------------------------------------------------------


def test_measurability_map_is_frozen() -> None:
    censoring = _make_censoring()
    mm: MeasurabilityMap = build_measurability_map(
        censoring, calibration=None, measurability_minimum=3
    )
    with pytest.raises((AttributeError, TypeError)):
        mm.coverage_ratio = 0.5  # type: ignore[misc]


def test_measurability_map_verdict_dict_preserved() -> None:
    censoring = _make_censoring()
    mm = build_measurability_map(censoring, calibration=None, measurability_minimum=3)
    for eid in ("E01", "E02", "E03", "E04", "E05"):
        assert mm.verdict[eid] is MeasurabilityVerdict.MEASURABLE
    assert mm.verdict["E06"] is MeasurabilityVerdict.FRAME_BLIND_UNMEASURABLE
