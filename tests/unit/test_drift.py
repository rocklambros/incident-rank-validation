"""Unit tests for engine.snapshot.drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from engine.snapshot.drift import (
    DriftReport,
    DriftSignoffRequired,
    _count_entries,
    detect_drift,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write *records* as JSONL to *path* and return *path*."""
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n")
    return path


def _snapshot(tmp_path: Path, name: str, records: list[dict[str, Any]]) -> Path:
    return _write_jsonl(tmp_path / name, records)


# ---------------------------------------------------------------------------
# _count_entries
# ---------------------------------------------------------------------------


def test_count_entries_basic(tmp_path: Path) -> None:
    p = _snapshot(
        tmp_path,
        "snap.jsonl",
        [
            {"owasp_llm": ["LLM01", "LLM02"]},
            {"owasp_llm": ["LLM01"]},
            {"owasp_llm": []},
            {"no_owasp": True},
        ],
    )
    counts = _count_entries(p)
    assert counts == {"LLM01": 2, "LLM02": 1}


def test_count_entries_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert _count_entries(p) == {}


def test_count_entries_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "blanks.jsonl"
    p.write_text('\n{"owasp_llm": ["LLM03"]}\n\n{"owasp_llm": ["LLM03"]}\n')
    assert _count_entries(p) == {"LLM03": 2}


# ---------------------------------------------------------------------------
# No drift → clean report
# ---------------------------------------------------------------------------


def test_no_drift_identical_snapshots(tmp_path: Path) -> None:
    records = [{"owasp_llm": ["LLM01", "LLM02"]}] * 100
    prev = _snapshot(tmp_path, "prev.jsonl", records)
    curr = _snapshot(tmp_path, "curr.jsonl", records)
    report = detect_drift(prev, curr)
    assert report.requires_signoff is False
    assert report.anomalies == ()


def test_no_drift_small_change(tmp_path: Path) -> None:
    # prev: LLM01 × 100, LLM02 × 100; curr: LLM01 × 105 (+5, 5% < 20%)
    prev_records = [{"owasp_llm": ["LLM01"]}] * 100 + [{"owasp_llm": ["LLM02"]}] * 100
    curr_records = [{"owasp_llm": ["LLM01"]}] * 105 + [{"owasp_llm": ["LLM02"]}] * 100
    prev = _snapshot(tmp_path, "prev.jsonl", prev_records)
    curr = _snapshot(tmp_path, "curr.jsonl", curr_records)
    report = detect_drift(prev, curr)
    assert report.requires_signoff is False
    assert report.anomalies == ()


# ---------------------------------------------------------------------------
# Count-delta drift
# ---------------------------------------------------------------------------


def test_drift_relative_threshold_exceeded(tmp_path: Path) -> None:
    # prev: LLM01 × 100; curr: LLM01 × 130 (+30, 30% > 20%)
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 100)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 130)
    report = detect_drift(prev, curr)
    assert report.requires_signoff is True
    assert len(report.anomalies) == 1
    a = report.anomalies[0]
    assert a.entry_id == "LLM01"
    assert a.metric == "count_delta"
    assert a.previous_value == 100.0
    assert a.current_value == 130.0


def test_drift_absolute_threshold_exceeded(tmp_path: Path) -> None:
    # prev: LLM01 × 1000; curr: LLM01 × 1060 (+60 absolute > 50, but 6% < 20%)
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 1000)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 1060)
    report = detect_drift(prev, curr)
    assert report.requires_signoff is True
    assert len(report.anomalies) == 1
    a = report.anomalies[0]
    assert a.metric == "count_delta"


def test_drift_below_both_thresholds_no_anomaly(tmp_path: Path) -> None:
    # prev: LLM01 × 1000; curr: LLM01 × 1010 (+10 absolute < 50, 1% < 20%)
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 1000)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 1010)
    report = detect_drift(prev, curr)
    assert report.requires_signoff is False


def test_drift_tighter_relative_threshold(tmp_path: Path) -> None:
    # With default 20% threshold: +10% is fine.  With 5% threshold: flagged.
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 100)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 110)

    report_loose = detect_drift(prev, curr, relative_threshold=0.20, absolute_threshold=50)
    assert report_loose.requires_signoff is False

    report_tight = detect_drift(prev, curr, relative_threshold=0.05, absolute_threshold=50)
    assert report_tight.requires_signoff is True
    assert report_tight.anomalies[0].metric == "count_delta"


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------


def test_burst_zero_to_above_threshold(tmp_path: Path) -> None:
    # prev: LLM04 absent (0); curr: LLM04 × 15 (> default burst_threshold=10)
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 50)
    curr = _snapshot(
        tmp_path,
        "curr.jsonl",
        [{"owasp_llm": ["LLM01"]}] * 50 + [{"owasp_llm": ["LLM04"]}] * 15,
    )
    report = detect_drift(prev, curr)
    assert report.requires_signoff is True
    bursts = [a for a in report.anomalies if a.metric == "burst"]
    assert len(bursts) == 1
    assert bursts[0].entry_id == "LLM04"
    assert bursts[0].previous_value == 0.0
    assert bursts[0].current_value == 15.0


def test_burst_zero_to_below_threshold_no_anomaly(tmp_path: Path) -> None:
    # prev: LLM04 absent (0); curr: LLM04 × 5 (≤ default burst_threshold=10)
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 50)
    curr = _snapshot(
        tmp_path,
        "curr.jsonl",
        [{"owasp_llm": ["LLM01"]}] * 50 + [{"owasp_llm": ["LLM04"]}] * 5,
    )
    report = detect_drift(prev, curr)
    assert report.requires_signoff is False


def test_burst_custom_threshold(tmp_path: Path) -> None:
    # burst_threshold=20: curr × 15 should NOT trigger; curr × 25 SHOULD.
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 50)

    curr_under = _snapshot(
        tmp_path,
        "curr_under.jsonl",
        [{"owasp_llm": ["LLM01"]}] * 50 + [{"owasp_llm": ["LLM04"]}] * 15,
    )
    curr_over = _snapshot(
        tmp_path,
        "curr_over.jsonl",
        [{"owasp_llm": ["LLM01"]}] * 50 + [{"owasp_llm": ["LLM04"]}] * 25,
    )

    assert detect_drift(prev, curr_under, burst_threshold=20).requires_signoff is False
    report = detect_drift(prev, curr_over, burst_threshold=20)
    assert report.requires_signoff is True
    assert report.anomalies[0].metric == "burst"


# ---------------------------------------------------------------------------
# DriftSignoffRequired
# ---------------------------------------------------------------------------


def test_drift_signoff_required_carries_report(tmp_path: Path) -> None:
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 100)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 130)
    report = detect_drift(prev, curr)
    exc = DriftSignoffRequired(report)
    assert exc.report is report
    assert "1 anomalies detected" in str(exc)
    assert "--accept-drift-signoff" in str(exc)


def test_drift_signoff_required_is_exception(tmp_path: Path) -> None:
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 100)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 130)
    report = detect_drift(prev, curr)
    with pytest.raises(DriftSignoffRequired) as exc_info:
        raise DriftSignoffRequired(report)
    assert isinstance(exc_info.value.report, DriftReport)


# ---------------------------------------------------------------------------
# DriftReport.to_json
# ---------------------------------------------------------------------------


def test_drift_report_to_json_structure(tmp_path: Path) -> None:
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 100)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 130)
    report = detect_drift(prev, curr)
    payload = json.loads(report.to_json())
    assert "previous_snapshot_hash" in payload
    assert "current_snapshot_hash" in payload
    assert isinstance(payload["anomalies"], list)
    assert payload["requires_signoff"] is True
    a = payload["anomalies"][0]
    for key in ("entry_id", "metric", "previous_value", "current_value", "description"):
        assert key in a


def test_drift_report_to_json_ends_with_newline(tmp_path: Path) -> None:
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 100)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 100)
    report = detect_drift(prev, curr)
    assert report.to_json().endswith("\n")


# ---------------------------------------------------------------------------
# Hash fields in report
# ---------------------------------------------------------------------------


def test_report_hash_fields_differ_for_different_snapshots(tmp_path: Path) -> None:
    prev = _snapshot(tmp_path, "prev.jsonl", [{"owasp_llm": ["LLM01"]}] * 10)
    curr = _snapshot(tmp_path, "curr.jsonl", [{"owasp_llm": ["LLM01"]}] * 20)
    report = detect_drift(prev, curr)
    assert report.previous_snapshot_hash != report.current_snapshot_hash
    assert len(report.previous_snapshot_hash) == 64  # SHA-256 hex


def test_report_hash_fields_equal_for_identical_content(tmp_path: Path) -> None:
    records = [{"owasp_llm": ["LLM01"]}] * 10
    prev = _snapshot(tmp_path, "prev.jsonl", records)
    curr = _snapshot(tmp_path, "curr.jsonl", records)
    report = detect_drift(prev, curr)
    assert report.previous_snapshot_hash == report.current_snapshot_hash


# ---------------------------------------------------------------------------
# Multiple anomalies
# ---------------------------------------------------------------------------


def test_multiple_anomalies_all_reported(tmp_path: Path) -> None:
    # LLM01: +30% drift; LLM04: burst from 0 to 20
    prev_records = [{"owasp_llm": ["LLM01"]}] * 100
    curr_records = [{"owasp_llm": ["LLM01"]}] * 130 + [{"owasp_llm": ["LLM04"]}] * 20
    prev = _snapshot(tmp_path, "prev.jsonl", prev_records)
    curr = _snapshot(tmp_path, "curr.jsonl", curr_records)
    report = detect_drift(prev, curr)
    assert report.requires_signoff is True
    assert len(report.anomalies) == 2
    metrics = {a.metric for a in report.anomalies}
    assert metrics == {"count_delta", "burst"}
