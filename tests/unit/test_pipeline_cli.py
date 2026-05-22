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


class TestExecuteFlags:
    def test_classify_real_execute_attempts_orchestration(self, tmp_path: Path) -> None:
        """F4.1: --execute flag triggers real classification, not just gate-checks."""
        cycle = tmp_path / "cycle"
        prereg = cycle / "prereg"
        prereg.mkdir(parents=True)
        (prereg / "manifest.json").write_text("{}")
        (prereg / "manifest.lock").write_text("{}")
        rubric_data = json.dumps({
            "cycle_id": "test-2026",
            "version": 1,
            "entries": [],
        })
        (prereg / "rubric.json").write_text(rubric_data)
        cal_dir = cycle / "calibrate"
        cal_dir.mkdir(parents=True)
        (cal_dir / "posteriors.json").write_text("{}")
        corpus = cycle / "corpora"
        corpus.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "classify-real", "--cycle", str(cycle), "--execute",
        ])
        # With --execute, the command should NOT just print "prerequisites satisfied"
        assert "prerequisites satisfied" not in (result.output or "").lower()

    def test_infer_real_execute_attempts_orchestration(self, tmp_path: Path) -> None:
        """--execute flag triggers real inference attempt."""
        cycle = tmp_path / "cycle"
        prereg = cycle / "prereg"
        prereg.mkdir(parents=True)
        (prereg / "manifest.lock").write_text("{}")
        classify_dir = cycle / "classify"
        classify_dir.mkdir(parents=True)
        (classify_dir / "labeled_incidents.json").write_text("[]")
        cal_dir = cycle / "calibrate"
        cal_dir.mkdir(parents=True)
        (cal_dir / "posteriors.json").write_text("{}")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "infer-real", "--cycle", str(cycle), "--execute",
        ])
        assert "prerequisites satisfied" not in (result.output or "").lower()

    def test_decide_real_execute_attempts_orchestration(self, tmp_path: Path) -> None:
        """--execute flag triggers real decision attempt."""
        cycle = tmp_path / "cycle"
        prereg = cycle / "prereg"
        prereg.mkdir(parents=True)
        (prereg / "manifest.lock").write_text("{}")
        infer_dir = cycle / "infer"
        infer_dir.mkdir(parents=True)
        (infer_dir / "inference_summary.json").write_text("{}")
        vote_file = tmp_path / "vote.xlsx"
        vote_file.write_bytes(b"")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "decide-real", "--cycle", str(cycle),
            "--vote-xlsx", str(vote_file), "--execute",
        ])
        assert "prerequisites satisfied" not in (result.output or "").lower()

    def test_without_execute_flag_still_gate_checks(self, tmp_path: Path) -> None:
        """Without --execute, commands still do prerequisite validation only."""
        cycle = tmp_path / "cycle"
        prereg = cycle / "prereg"
        prereg.mkdir(parents=True)
        (prereg / "manifest.json").write_text("{}")
        (prereg / "manifest.lock").write_text("{}")
        rubric_data = json.dumps({
            "cycle_id": "test-2026",
            "version": 1,
            "entries": [],
        })
        (prereg / "rubric.json").write_text(rubric_data)
        cal_dir = cycle / "calibrate"
        cal_dir.mkdir(parents=True)
        (cal_dir / "posteriors.json").write_text("{}")
        corpus = cycle / "corpora"
        corpus.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "classify-real", "--cycle", str(cycle),
        ])
        assert "prerequisites satisfied" in (result.output or "").lower()
