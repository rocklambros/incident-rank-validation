"""Unit tests for engine.cli.main."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from engine.cli.main import cli


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Command existence
# ---------------------------------------------------------------------------


class TestCommandsExist:
    def test_infer_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["infer", "--help"])
        assert result.exit_code == 0
        assert "prereg lock" in result.output

    def test_decide_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["decide", "--help"])
        assert result.exit_code == 0

    def test_run_synthetic_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run-synthetic", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# infer phase gate
# ---------------------------------------------------------------------------


class TestInferPhaseGate:
    def test_rejects_missing_prereg_lock(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["infer", "--cycle", str(tmp_path), "--corpus-mode", "synthetic"],
        )
        assert result.exit_code != 0
        assert "prereg lock not found" in result.output

    def test_rejects_vote_data_during_infer(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "prereg"
        lock_dir.mkdir()
        (lock_dir / "prereg.lock.json").write_text("{}")
        vote_dir = tmp_path / "vote"
        vote_dir.mkdir()
        (vote_dir / "votes.json").write_text("{}")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["infer", "--cycle", str(tmp_path), "--corpus-mode", "synthetic"],
        )
        assert result.exit_code != 0
        assert "Vote data found" in result.output

    def test_passes_with_lock_no_drift(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "prereg"
        lock_dir.mkdir()
        (lock_dir / "prereg.lock.json").write_text("{}")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["infer", "--cycle", str(tmp_path), "--corpus-mode", "synthetic"],
        )
        assert result.exit_code == 0
        assert "prerequisites satisfied" in result.output


# ---------------------------------------------------------------------------
# Drift signoff M13
# ---------------------------------------------------------------------------


def _setup_drift_cycle(tmp_path: Path) -> Path:
    """Create a cycle directory with prereg lock and drifting snapshots."""
    lock_dir = tmp_path / "prereg"
    lock_dir.mkdir()
    (lock_dir / "prereg.lock.json").write_text("{}")

    corpora = tmp_path / "corpora"
    corpora.mkdir()

    # Previous: LLM01 has 100 entries
    prev_records: list[dict[str, Any]] = [
        {"id": f"CVE-prev-{i}", "owasp_llm": ["LLM01"]} for i in range(100)
    ]
    _write_jsonl(corpora / "snapshot.previous.jsonl", prev_records)

    # Current: LLM01 has 200 entries (>20% and >50 absolute drift)
    curr_records: list[dict[str, Any]] = [
        {"id": f"CVE-curr-{i}", "owasp_llm": ["LLM01"]} for i in range(200)
    ]
    _write_jsonl(corpora / "snapshot.jsonl", curr_records)

    return tmp_path


class TestDriftSignoff:
    def test_short_signoff_rejected(self, tmp_path: Path) -> None:
        cycle = _setup_drift_cycle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "infer",
                "--cycle", str(cycle),
                "--corpus-mode", "synthetic",
                "--accept-drift-signoff", "too short",
            ],
        )
        assert result.exit_code != 0
        assert "Drift signoff required" in result.output

    def test_no_signoff_rejected(self, tmp_path: Path) -> None:
        cycle = _setup_drift_cycle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "infer",
                "--cycle", str(cycle),
                "--corpus-mode", "synthetic",
            ],
        )
        assert result.exit_code != 0
        assert "Drift signoff required" in result.output

    def test_signoff_ge_30_chars_accepted(self, tmp_path: Path) -> None:
        cycle = _setup_drift_cycle(tmp_path)
        reason = "Accepted: known upstream data refresh event on 2026-01-15"
        assert len(reason) >= 30
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "infer",
                "--cycle", str(cycle),
                "--corpus-mode", "synthetic",
                "--accept-drift-signoff", reason,
            ],
        )
        assert result.exit_code == 0
        assert "prerequisites satisfied" in result.output

    def test_signoff_file_persisted(self, tmp_path: Path) -> None:
        cycle = _setup_drift_cycle(tmp_path)
        reason = "Accepted: known upstream data refresh event on 2026-01-15"
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "infer",
                "--cycle", str(cycle),
                "--corpus-mode", "synthetic",
                "--accept-drift-signoff", reason,
            ],
        )
        signoff_dir = cycle / "drift_signoffs"
        assert signoff_dir.exists()
        signoff_files = list(signoff_dir.glob("*.txt"))
        assert len(signoff_files) == 1
        content = signoff_files[0].read_text()
        assert reason in content
        assert "Drift signoff accepted" in content


# ---------------------------------------------------------------------------
# decide phase gate
# ---------------------------------------------------------------------------


class TestDecidePhaseGate:
    def test_rejects_missing_prereg_lock(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["decide", "--cycle", str(tmp_path), "--corpus-mode", "synthetic"],
        )
        assert result.exit_code != 0
        assert "prereg lock not found" in result.output

    def test_passes_with_lock(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "prereg"
        lock_dir.mkdir()
        (lock_dir / "prereg.lock.json").write_text("{}")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["decide", "--cycle", str(tmp_path), "--corpus-mode", "synthetic"],
        )
        assert result.exit_code == 0
        assert "prerequisites satisfied" in result.output
