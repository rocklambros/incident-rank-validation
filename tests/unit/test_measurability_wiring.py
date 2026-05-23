"""Tests for measurability verdict wiring from diagnostic.json.

Verifies that decide_real and report_cmd read frame-blind flags from
calibration/diagnostic.json instead of hardcoding all entries as measurable.
"""
from __future__ import annotations

import json
from pathlib import Path


class TestLoadMeasurabilityVerdicts:
    def test_no_data_entries_are_frame_blind(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        diag = {
            "entry_reports": {
                "LLM01": {"flag": "adequate"},
                "LLM04": {"flag": "no-data"},
                "LLM08": {"flag": "no-data"},
                "LLM05": {"flag": "wide"},
            }
        }
        (cal_dir / "diagnostic.json").write_text(json.dumps(diag))

        from engine.cli.pipeline import _load_measurability_verdicts

        entry_ids = ("LLM01", "LLM04", "LLM05", "LLM08")
        verdicts = _load_measurability_verdicts(cal_dir, entry_ids)

        assert verdicts["LLM04"] == "frame_blind_unmeasurable"
        assert verdicts["LLM08"] == "frame_blind_unmeasurable"
        assert verdicts["LLM01"] == "measurable"
        assert verdicts["LLM05"] == "measurable"

    def test_measurable_count_excludes_frame_blind(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        diag = {
            "entry_reports": {
                "E1": {"flag": "adequate"},
                "E2": {"flag": "no-data"},
                "E3": {"flag": "wide"},
                "E4": {"flag": "no-data"},
                "E5": {"flag": "adequate"},
            }
        }
        (cal_dir / "diagnostic.json").write_text(json.dumps(diag))

        from engine.cli.pipeline import _load_measurability_verdicts

        entry_ids = ("E1", "E2", "E3", "E4", "E5")
        verdicts = _load_measurability_verdicts(cal_dir, entry_ids)

        measurable = [e for e, v in verdicts.items() if v != "frame_blind_unmeasurable"]
        frame_blind = [e for e, v in verdicts.items() if v == "frame_blind_unmeasurable"]
        assert len(measurable) == 3
        assert len(frame_blind) == 2
        assert set(frame_blind) == {"E2", "E4"}

    def test_missing_diagnostic_defaults_all_measurable(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()

        from engine.cli.pipeline import _load_measurability_verdicts

        entry_ids = ("E1", "E2")
        verdicts = _load_measurability_verdicts(cal_dir, entry_ids)

        assert all(v == "measurable" for v in verdicts.values())

    def test_entry_missing_from_diagnostic_defaults_measurable(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        diag = {
            "entry_reports": {
                "E1": {"flag": "adequate"},
            }
        }
        (cal_dir / "diagnostic.json").write_text(json.dumps(diag))

        from engine.cli.pipeline import _load_measurability_verdicts

        entry_ids = ("E1", "E2")
        verdicts = _load_measurability_verdicts(cal_dir, entry_ids)

        assert verdicts["E1"] == "measurable"
        assert verdicts["E2"] == "measurable"

    def test_real_diagnostic_20_entries_gives_17_measurable(self, tmp_path: Path) -> None:
        """Mirrors the real 2026 cycle: 3 no-data entries → 17 measurable."""
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        diag = {
            "entry_reports": {
                "LLM01": {"flag": "adequate"},
                "LLM02": {"flag": "wide"},
                "LLM03": {"flag": "wide"},
                "LLM04": {"flag": "no-data"},
                "LLM05": {"flag": "adequate"},
                "LLM06": {"flag": "wide"},
                "LLM07": {"flag": "wide"},
                "LLM08": {"flag": "no-data"},
                "LLM09": {"flag": "adequate"},
                "LLM10": {"flag": "no-data"},
                "NEW-PMP": {"flag": "wide"},
                "NEW-MTIE": {"flag": "wide"},
                "NEW-MA": {"flag": "wide"},
                "NEW-ITSCD": {"flag": "wide"},
                "NEW-WLA": {"flag": "wide"},
                "NEW-MSDA": {"flag": "wide"},
                "ROLL-CMSB": {"flag": "wide"},
                "ROLL-LAPTF": {"flag": "wide"},
                "ROLL-SICG": {"flag": "wide"},
                "ROLL-CFAS": {"flag": "wide"},
            }
        }
        (cal_dir / "diagnostic.json").write_text(json.dumps(diag))

        from engine.cli.pipeline import _load_measurability_verdicts

        entry_ids = tuple(diag["entry_reports"].keys())
        verdicts = _load_measurability_verdicts(cal_dir, entry_ids)

        measurable = [e for e, v in verdicts.items() if v != "frame_blind_unmeasurable"]
        frame_blind = [e for e, v in verdicts.items() if v == "frame_blind_unmeasurable"]
        assert len(measurable) == 17
        assert len(frame_blind) == 3
        assert set(frame_blind) == {"LLM04", "LLM08", "LLM10"}
