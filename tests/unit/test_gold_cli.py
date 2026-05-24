"""Tests for --gold-calibration flag on cal-tally."""
from __future__ import annotations

from click.testing import CliRunner

from engine.cli.calibration import cal_tally


class TestGoldCalibrationFlag:
    def test_flag_accepted(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cal_tally, ["--help"])
        assert "--gold-calibration" in result.output

    def test_flag_optional(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cal_tally, ["--help"])
        assert result.exit_code == 0
