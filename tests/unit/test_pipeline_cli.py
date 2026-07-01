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
        cal_dir = cycle / "calibration"
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
        cal_dir = cycle / "calibration"
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
        cal_dir = cycle / "calibration"
        cal_dir.mkdir(parents=True)
        (cal_dir / "posteriors.json").write_text("{}")
        corpus = cycle / "corpora"
        corpus.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "classify-real", "--cycle", str(cycle),
        ])
        assert "prerequisites satisfied" in (result.output or "").lower()


def test_corroborate_requires_cycle(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["corroborate", "--cycle", str(tmp_path), "--corpus-b-dir", str(tmp_path)],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# U2-8: report_cmd robustness-spread gate (grandfathered + content-validated)
# ---------------------------------------------------------------------------

_MINIMAL_MANIFEST_FIELDS: dict[str, object] = {
    "engine_version": "0.0.0",
    "engine_version_range_min": "0.0.0",
    "engine_version_range_max": "999.0.0",
    "cycle_id": "test-u2-8",
    "taxonomy_hash": "abc",
    "snapshot_hash": "def",
    "primary_spec": "poisson_flat",
    "robustness_specs": ["poisson_flat"],
    "flag_threshold_tau": 0.8,
    "statistic": "weighted_cohens_kappa",
    "measurability_minimum": 2,
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

_MINIMAL_CONCORDANCE: dict[str, object] = {
    "weighted_kappa_median": 0.5,
    "weighted_kappa_ci": [0.3, 0.7],
    "measurable_count": 5,
    "total_count": 10,
    "coverage_ratio": 0.5,
    "flags": [],
}


def _make_report_cycle(
    tmp_path: Path,
    *,
    schema_version: int = 1,
    robustness_specs: list[str] | None = None,
    spread_json: dict[str, object] | None = None,
) -> Path:
    """Build a minimal cycle directory that can reach the robustness-spread gate."""
    cycle = tmp_path / "cycle"
    prereg = cycle / "prereg"
    prereg.mkdir(parents=True)
    results_dir = cycle / "results"
    results_dir.mkdir(parents=True)

    manifest: dict[str, object] = dict(_MINIMAL_MANIFEST_FIELDS)
    if robustness_specs is not None:
        manifest["robustness_specs"] = robustness_specs
    if schema_version != 1:
        manifest["schema_version"] = schema_version

    (prereg / "manifest.json").write_text(json.dumps(manifest))
    (results_dir / "concordance.json").write_text(json.dumps(_MINIMAL_CONCORDANCE))

    if spread_json is not None:
        (results_dir / "robustness_spread.json").write_text(json.dumps(spread_json))

    return cycle


def test_report_grandfathers_locked_v1v2_cycle(tmp_path: Path) -> None:
    """schema_version < 3 cycle with declared robustness_specs but no spread → report SUCCEEDS.

    Locked v1/v2 cycles (e.g. the committed 2026 cycle) declare robustness_specs but
    never wrote a robustness_spread.json.  The grandfather clause must let them regen
    without raising (U2-8).
    """
    cycle = _make_report_cycle(
        tmp_path,
        schema_version=1,
        robustness_specs=["poisson_flat"],
        spread_json=None,  # no spread file
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--cycle", str(cycle)])
    assert result.exit_code == 0, (
        f"report should succeed for v1 cycle without spread; "
        f"exit={result.exit_code}, output={result.output!r}"
    )
    assert (cycle / "results" / "report.md").exists()


def test_report_raises_when_v3_declares_but_spread_missing(tmp_path: Path) -> None:
    """schema_version >= 3, declared specs, no spread → ClickException naming decide phase."""
    cycle = _make_report_cycle(
        tmp_path,
        schema_version=3,
        robustness_specs=["poisson_flat"],
        spread_json=None,  # no spread file
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--cycle", str(cycle)])
    assert result.exit_code != 0
    output = result.output.lower()
    assert "decide" in output, (
        f"error message should name decide phase; got: {result.output!r}"
    )


def test_report_refuses_null_kappa_decoy() -> None:
    """Spread present, all spec names present, but one has null kappa → raises naming the spec."""
    import pytest

    from engine.cli.pipeline import assert_robustness_complete
    from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult

    class M:
        robustness_specs = ("poisson_flat",)
        schema_version = 3

    spread = RobustnessSpread(
        primary=SpecResult(
            spec_name="negative_binomial_per_stratum",
            weighted_kappa_median=0.65,
            weighted_kappa_ci=(0.5, 0.8),
            flags=(),
        ),
        robustness=(
            SpecResult(
                spec_name="poisson_flat",
                weighted_kappa_median=None,  # decoy: name present but kappa is null
                weighted_kappa_ci=None,
                flags=(),
            ),
        ),
    )
    with pytest.raises(ValueError, match="poisson_flat"):
        assert_robustness_complete(M(), spread)


def test_report_refuses_nonfinite_kappa_decoy() -> None:
    """Spread present with NaN kappa → raises naming the spec (non-finite guard)."""
    import math

    import pytest

    from engine.cli.pipeline import assert_robustness_complete
    from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult

    class M:
        robustness_specs = ("poisson_flat",)
        schema_version = 3

    spread = RobustnessSpread(
        primary=SpecResult(
            spec_name="negative_binomial_per_stratum",
            weighted_kappa_median=0.65,
            weighted_kappa_ci=(0.5, 0.8),
            flags=(),
        ),
        robustness=(
            SpecResult(
                spec_name="poisson_flat",
                weighted_kappa_median=math.nan,
                weighted_kappa_ci=None,
                flags=(),
            ),
        ),
    )
    with pytest.raises(ValueError, match="poisson_flat"):
        assert_robustness_complete(M(), spread)


def test_report_accepts_valid_spread() -> None:
    """Spread present with finite kappa for each declared spec → no raise."""
    from engine.cli.pipeline import assert_robustness_complete
    from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult

    class M:
        robustness_specs = ("poisson_flat",)
        schema_version = 3

    spread = RobustnessSpread(
        primary=SpecResult(
            spec_name="negative_binomial_per_stratum",
            weighted_kappa_median=0.65,
            weighted_kappa_ci=(0.5, 0.8),
            flags=(),
        ),
        robustness=(
            SpecResult(
                spec_name="poisson_flat",
                weighted_kappa_median=0.60,
                weighted_kappa_ci=(0.45, 0.75),
                flags=(),
            ),
        ),
    )
    assert_robustness_complete(M(), spread)  # no raise
