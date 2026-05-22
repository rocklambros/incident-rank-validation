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


class TestInferRealCLI:
    def test_infer_real_requires_lock(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["infer-real", "--cycle", str(tmp_path)])
        assert result.exit_code != 0
        assert "lock" in result.output.lower()

    def test_infer_real_rejects_vote_data(self, tmp_path: Path) -> None:
        prereg = tmp_path / "prereg"
        prereg.mkdir()
        (prereg / "manifest.lock").write_text("{}")
        classify_dir = tmp_path / "classify"
        classify_dir.mkdir()
        (classify_dir / "labeled_incidents.json").write_text("[]")
        vote_dir = tmp_path / "vote"
        vote_dir.mkdir()
        (vote_dir / "results.json").write_text("{}")
        runner = CliRunner()
        result = runner.invoke(cli, ["infer-real", "--cycle", str(tmp_path)])
        assert result.exit_code != 0
        assert "vote" in result.output.lower()

    def test_infer_real_requires_classify_output(self, tmp_path: Path) -> None:
        prereg = tmp_path / "prereg"
        prereg.mkdir()
        (prereg / "manifest.lock").write_text("{}")
        runner = CliRunner()
        result = runner.invoke(cli, ["infer-real", "--cycle", str(tmp_path)])
        assert result.exit_code != 0
        assert "classify" in result.output.lower() or "labeled" in result.output.lower()


class TestDecideRealCLI:
    def test_decide_real_requires_lock(self, tmp_path: Path) -> None:
        vote_file = tmp_path / "vote.xlsx"
        vote_file.write_bytes(b"")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "decide-real", "--cycle", str(tmp_path),
            "--vote-xlsx", str(vote_file),
        ])
        assert result.exit_code != 0
        assert "lock" in result.output.lower()

    def test_decide_real_requires_infer(self, tmp_path: Path) -> None:
        prereg = tmp_path / "prereg"
        prereg.mkdir()
        (prereg / "manifest.lock").write_text("{}")
        vote_file = tmp_path / "vote.xlsx"
        vote_file.write_bytes(b"")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "decide-real", "--cycle", str(tmp_path),
            "--vote-xlsx", str(vote_file),
        ])
        assert result.exit_code != 0
        assert "infer" in result.output.lower()
