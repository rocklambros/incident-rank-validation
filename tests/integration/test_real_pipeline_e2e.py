"""Integration test: pipeline CLI commands on synthetic data.

Verifies the full command sequence and phase gates work end-to-end
without requiring real corpus data, RunPod, or XLSX vote files.
"""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from engine.cli.main import cli


def _setup_cycle(tmp_path: Path) -> Path:
    """Create minimal cycle directory with prereg artifacts."""
    cycle = tmp_path / "projects" / "test" / "cycles" / "2026"
    prereg = cycle / "prereg"
    prereg.mkdir(parents=True)

    manifest = {
        "engine_version": "1.0.0",
        "cycle_id": "2026",
        "primary_spec": "negative_binomial_per_stratum",
        "robustness_specs": ["poisson_flat"],
        "flag_threshold_tau": 0.8,
        "measurability_minimum": 4,
    }
    (prereg / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (prereg / "manifest.lock").write_text(json.dumps({"hash": "test"}))
    (prereg / "rubric.json").write_text(json.dumps({"entries": []}))

    return cycle


class TestPipelineGates:
    def test_classify_real_blocked_without_calibration(self, tmp_path: Path) -> None:
        cycle = _setup_cycle(tmp_path)
        corpus = cycle / "corpora" / "genai_agentic" / "abc123"
        corpus.mkdir(parents=True)
        (corpus / "incidents.json").write_text("[]")
        runner = CliRunner()
        result = runner.invoke(cli, ["classify-real", "--cycle", str(cycle)])
        assert result.exit_code != 0
        assert "calibration" in result.output.lower() or "posteriors" in result.output.lower()

    def test_infer_blocked_without_classify(self, tmp_path: Path) -> None:
        cycle = _setup_cycle(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["infer-real", "--cycle", str(cycle)])
        assert result.exit_code != 0

    def test_decide_blocked_without_infer(self, tmp_path: Path) -> None:
        cycle = _setup_cycle(tmp_path)
        vote_file = tmp_path / "vote.xlsx"
        vote_file.write_bytes(b"")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "decide-real", "--cycle", str(cycle),
            "--vote-xlsx", str(vote_file),
        ])
        assert result.exit_code != 0
