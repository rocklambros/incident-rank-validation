"""Integration test: pipeline CLI commands on synthetic data.

Verifies the full command sequence and phase gates work end-to-end
without requiring real corpus data, RunPod, or XLSX vote files.
"""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from engine.cli.main import cli


def _write_valid_lock(prereg: Path) -> None:
    """Write manifest.lock that passes the R5 _verify_manifest_lock guard."""
    from engine.cli.pipeline_executor import _load_manifest
    from engine.prereg.lock import write_lock

    manifest = _load_manifest(prereg / "manifest.json")
    write_lock(manifest, prereg / "manifest.lock")


_FULL_MANIFEST: dict[str, object] = {
    "engine_version": "1.0.0",
    "engine_version_range_min": "0.0.0",
    "engine_version_range_max": "999.0.0",
    "cycle_id": "2026",
    "taxonomy_hash": "abc",
    "snapshot_hash": "def",
    "primary_spec": "negative_binomial_per_stratum",
    "robustness_specs": ["poisson_flat"],
    "flag_threshold_tau": 0.8,
    "statistic": "weighted_cohens_kappa",
    "measurability_minimum": 4,
    "prior_scale": 0.5,
    "concentration_shape": 5.0,
    "concentration_rate": 0.1,
    "ess_fraction": 0.4,
    "meaningful_kappa_n": 4,
    "prng_seed": 42,
    "confidence_threshold": 0.3,
    "rubric_drafting_attestation": None,
    "rubric_reviewer": None,
    "statistical_reviewer": None,
    "classifier_rule_hash": None,
    "rubric_hash": None,
    "post_hoc_register_path": None,
}


def _setup_cycle(tmp_path: Path) -> Path:
    """Create minimal cycle directory with prereg artifacts."""
    cycle = tmp_path / "projects" / "test" / "cycles" / "2026"
    prereg = cycle / "prereg"
    prereg.mkdir(parents=True)

    (prereg / "manifest.json").write_text(json.dumps(_FULL_MANIFEST, indent=2))
    _write_valid_lock(prereg)
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
