"""Tests for engine.calibrate.calibrate — calibration computation + diagnostic."""
from __future__ import annotations

import math

import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.calibrate.calibrate import (
    CalibrationDiagnostic,
    EntryCalibrationReport,
    compute_calibration,
)
from engine.calibrate.tally import PrecisionTally, RecallTally, TallyResult


def _tally(
    precision: dict[tuple[str, str], tuple[int, int, int]] | None = None,
    recall: dict[tuple[str, str], tuple[int, int, int]] | None = None,
) -> TallyResult:
    pc = {
        k: PrecisionTally(true_positives=v[0], false_positives=v[1], total=v[2])
        for k, v in (precision or {}).items()
    }
    rc = {
        k: RecallTally(true_positives=v[0], false_negatives=v[1], total_in_sample=v[2])
        for k, v in (recall or {}).items()
    }
    total = sum(t.total for t in pc.values()) + sum(
        t.total_in_sample for t in rc.values()
    )
    return TallyResult(
        precision_counts=pc,
        recall_counts=rc,
        rollup_counts={},
        total_coded=total,
        amendments_applied=0,
    )


class TestComputeCalibration:
    def test_precision_posterior(self) -> None:
        tally = _tally(
            precision={("LLM01", "security"): (35, 5, 40)},
        )
        cal, diag = compute_calibration(
            tally, all_entry_ids=["LLM01"], strata=["security"],
            frame_blind_ids=set(),
        )
        bp = cal.precision[("LLM01", "security")]
        assert bp.alpha == 36.0  # 35 + 1
        assert bp.beta == 6.0  # 5 + 1

    def test_recall_posterior(self) -> None:
        tally = _tally(
            recall={("LLM01", "security"): (30, 70, 100)},
        )
        cal, diag = compute_calibration(
            tally, all_entry_ids=["LLM01"], strata=["security"],
            frame_blind_ids=set(),
        )
        bp = cal.recall[("LLM01", "security")]
        assert bp.alpha == 31.0
        assert bp.beta == 71.0

    def test_diagnostic_adequate(self) -> None:
        tally = _tally(
            precision={("LLM01", "security"): (35, 5, 40)},
            recall={("LLM01", "security"): (30, 10, 40)},
        )
        cal, diag = compute_calibration(
            tally, all_entry_ids=["LLM01"], strata=["security"],
            frame_blind_ids=set(),
        )
        report = diag.entry_reports["LLM01"]
        assert report.has_precision_data is True
        assert report.has_recall_data is True
        assert report.flag == "adequate"

    def test_diagnostic_no_data_frame_blind(self) -> None:
        tally = _tally()
        cal, diag = compute_calibration(
            tally, all_entry_ids=["LLM04"], strata=["security"],
            frame_blind_ids={"LLM04"},
        )
        report = diag.entry_reports["LLM04"]
        assert report.flag == "no-data"
        assert "frame-blind" in report.reason

    def test_diagnostic_recall_only(self) -> None:
        tally = _tally(
            recall={("NEW-PMP", "security"): (2, 98, 100)},
        )
        cal, diag = compute_calibration(
            tally, all_entry_ids=["NEW-PMP"], strata=["security"],
            frame_blind_ids=set(),
        )
        report = diag.entry_reports["NEW-PMP"]
        assert report.has_precision_data is False
        assert report.has_recall_data is True
        assert "recall-frame-only" in report.reason
