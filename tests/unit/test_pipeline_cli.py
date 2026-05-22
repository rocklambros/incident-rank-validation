# tests/unit/test_pipeline_cli.py
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from engine.cli.main import cli


class TestClassifyRealCLI:
    def test_classify_real_requires_manifest(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["classify-real", "--cycle", str(tmp_path)])
        assert result.exit_code != 0
        assert "prereg" in result.output.lower() or "manifest" in result.output.lower()

    def test_classify_real_requires_calibration(self, tmp_path: Path) -> None:
        """R3: calibration posteriors must exist before classify-real."""
        prereg = tmp_path / "prereg"
        prereg.mkdir()
        (prereg / "manifest.json").write_text("{}")
        (prereg / "manifest.lock").write_text("{}")
        (prereg / "rubric.json").write_text("{}")
        (tmp_path / "corpora").mkdir()
        runner = CliRunner()
        result = runner.invoke(cli, ["classify-real", "--cycle", str(tmp_path)])
        assert result.exit_code != 0
        assert "calibration" in result.output.lower() or "posteriors" in result.output.lower()

    def test_classify_real_requires_rubric(self, tmp_path: Path) -> None:
        prereg = tmp_path / "prereg"
        prereg.mkdir()
        (prereg / "manifest.json").write_text("{}")
        (prereg / "manifest.lock").write_text("{}")
        runner = CliRunner()
        result = runner.invoke(cli, ["classify-real", "--cycle", str(tmp_path)])
        assert result.exit_code != 0
