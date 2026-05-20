"""Tests for the synthetic end-to-end pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from engine.adapters.synthetic import SyntheticAdapter
from engine.cli.synthetic import (
    _stratum_size_sanity_check,
    execute_synthetic_pipeline,
)
from engine.schema import StratumSize, make_stratum_size


def _setup_cycle_dir(tmp_path: Path) -> Path:
    """Create a minimal cycle directory mirroring projects/synthetic/cycles/2026."""
    project_root = tmp_path / "projects" / "synthetic"
    cycle = project_root / "cycles" / "2026"
    cycle.mkdir(parents=True)

    # Copy taxonomy from the real project
    real_taxonomy = (
        Path(__file__).resolve().parents[2]
        / "projects"
        / "synthetic"
        / "cycles"
        / "2026"
        / "taxonomy"
        / "taxonomy.json"
    )
    tax_dir = cycle / "taxonomy"
    tax_dir.mkdir()
    shutil.copy(real_taxonomy, tax_dir / "taxonomy.json")

    # Copy project.toml
    real_toml = (
        Path(__file__).resolve().parents[2]
        / "projects"
        / "synthetic"
        / "project.toml"
    )
    shutil.copy(real_toml, project_root / "project.toml")

    return cycle


# ---------------------------------------------------------------------------
# M3: stratum-size sanity check
# ---------------------------------------------------------------------------


class TestM3StratumSizeSanity:
    def test_rejects_undersized_stratum(self) -> None:
        """stratum_size below observed count must raise ValueError."""
        adapter = SyntheticAdapter(seed=42)
        incidents = tuple(adapter.iter_incidents())

        class UndersizedAdapter(SyntheticAdapter):
            def stratum_sizes(self) -> dict[str, StratumSize]:
                return {
                    "stratum_a": make_stratum_size(10),
                    "stratum_b": make_stratum_size(300),
                }

        bad_adapter = UndersizedAdapter(seed=42)
        with pytest.raises(ValueError, match="stratum_size.*must be >= observed count"):
            _stratum_size_sanity_check(bad_adapter, incidents)

    def test_correct_sizes_pass(self) -> None:
        """Default SyntheticAdapter sizes should not raise."""
        adapter = SyntheticAdapter(seed=42)
        incidents = tuple(adapter.iter_incidents())
        _stratum_size_sanity_check(adapter, incidents)


# ---------------------------------------------------------------------------
# End-to-end pipeline (non-slow: verifies outputs exist)
# ---------------------------------------------------------------------------


class TestSyntheticPipelineOutputs:
    """Verify the pipeline writes expected artifacts.

    NUTS may fail diagnostics with Beta(1,1) calibration and small
    synthetic counts.  The pipeline must write coverage.json regardless.
    """

    @pytest.fixture()
    def cycle_dir(self, tmp_path: Path) -> Path:
        return _setup_cycle_dir(tmp_path)

    @pytest.mark.slow()
    def test_pipeline_writes_coverage_json(self, cycle_dir: Path) -> None:
        execute_synthetic_pipeline(cycle_dir, corpus_mode="synthetic")
        coverage_path = cycle_dir / "results" / "coverage.json"
        assert coverage_path.exists(), "coverage.json must be written"
        data = json.loads(coverage_path.read_text())
        assert "coverage_ratio" in data
        assert "measurable" in data
        assert "frame_blind" in data

    @pytest.mark.slow()
    def test_pipeline_writes_report(self, cycle_dir: Path) -> None:
        execute_synthetic_pipeline(cycle_dir, corpus_mode="synthetic")
        report_path = cycle_dir / "results" / "report.md"
        assert report_path.exists(), "report.md must be written"
        text = report_path.read_text()
        assert "Cycle Report" in text
        assert "NON-PUBLISHABLE" in text

    @pytest.mark.slow()
    def test_pipeline_writes_summary_json(self, cycle_dir: Path) -> None:
        execute_synthetic_pipeline(cycle_dir, corpus_mode="synthetic")
        summary_path = cycle_dir / "results" / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["corpus_mode"] == "synthetic"
        assert data["non_publishable"] is True

    @pytest.mark.slow()
    def test_pipeline_writes_prereg_lock(self, cycle_dir: Path) -> None:
        execute_synthetic_pipeline(cycle_dir, corpus_mode="synthetic")
        lock_path = cycle_dir / "prereg" / "prereg.lock.json"
        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert "manifest_hash" in data

    @pytest.mark.slow()
    def test_coverage_json_structure(self, cycle_dir: Path) -> None:
        execute_synthetic_pipeline(cycle_dir, corpus_mode="synthetic")
        data = json.loads((cycle_dir / "results" / "coverage.json").read_text())
        assert isinstance(data["coverage_ratio"], float)
        assert isinstance(data["measurable"], list)
        assert isinstance(data["frame_blind"], list)
        assert "E06" in data["frame_blind"]
        assert data["coverage_ratio"] > 0
