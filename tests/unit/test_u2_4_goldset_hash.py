"""U2-4 tests: goldset_hash threading (manifest-authoritative) + sentinel normalization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from engine.calibrate.gold_schema import GoldCalibration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_gold(provenance_hash: str = "fakehash123") -> GoldCalibration:
    return GoldCalibration(
        recall_labels=[],
        precision_labels=[],
        provenance_hash=provenance_hash,
        rubric_hash="rub",
        adjudicator_id="tester",
        session_count=1,
    )


def _make_minimal_cycle(root: Path, *, goldset_hash_in_manifest: str | None = None) -> Path:
    """Create the smallest cycle directory that repro_bundle_cmd accepts."""
    cycle = root / "cycle"
    prereg = cycle / "prereg"
    prereg.mkdir(parents=True)
    manifest_data: dict[str, object] = {"cycle_id": "test-cycle"}
    if goldset_hash_in_manifest is not None:
        manifest_data["goldset_hash"] = goldset_hash_in_manifest
    (prereg / "manifest.json").write_text(json.dumps(manifest_data))
    return cycle


# ---------------------------------------------------------------------------
# [G] execute_infer_phase writes infer/goldset_hash.txt when gold is present
# ---------------------------------------------------------------------------


def test_infer_writes_goldset_hash_when_gold_present(tmp_path: Path) -> None:
    """execute_infer_phase must write infer/goldset_hash.txt from _gold.provenance_hash."""
    from tests.unit.test_prereg import _make_manifest

    # Minimal cycle structure
    cycle = tmp_path / "cycle"
    (cycle / "calibration").mkdir(parents=True)
    # adjudicated_goldset.jsonl triggers _has_gold_files
    (cycle / "calibration" / "adjudicated_goldset.jsonl").write_text("")
    (cycle / "calibration" / "posteriors.json").write_text("{}")
    (cycle / "classify").mkdir()
    (cycle / "classify" / "labeled_incidents.json").write_text("[]")
    (cycle / "prereg").mkdir()
    (cycle / "prereg" / "manifest.json").write_text("{}")

    manifest = _make_manifest(robustness_specs=())
    # Write manifest.lock so the R5 _verify_manifest_lock guard passes.
    # Must match the patched _load_manifest return value (same manifest object).
    from engine.prereg.lock import write_lock
    write_lock(manifest, cycle / "prereg" / "manifest.lock")
    fake_gold = _fake_gold("deadbeef42")
    fake_counts = (np.array([1.0]), np.array([5.0]), ["e1"], [MagicMock()])

    with (
        patch("jax.default_backend", return_value="cpu"),
        patch("engine.cli.pipeline_executor._load_manifest", return_value=manifest),
        patch("engine.cli.pipeline_executor._load_calibration", return_value=MagicMock()),
        patch(
            "engine.cli.pipeline_executor._build_counts_from_labeled",
            return_value=fake_counts,
        ),
        patch("engine.calibrate.coverage.verify_labeled_completeness"),
        patch("engine.calibrate.gold_loader.load_classifier_labels", return_value=[]),
        patch("engine.calibrate.gold_loader.load_gold_calibration", return_value=fake_gold),
        patch(
            "engine.calibrate.confusion.build_overlap_from_confusion",
            return_value=MagicMock(),
        ),
        patch("engine.model.inference.run_inference", return_value=MagicMock()),
        patch("engine.cli.pipeline_executor.write_infer_artifacts"),
    ):
        from engine.cli.pipeline_executor import execute_infer_phase

        execute_infer_phase(cycle)

    hash_file = cycle / "infer" / "goldset_hash.txt"
    assert hash_file.exists(), "infer/goldset_hash.txt was not written"
    assert hash_file.read_text().strip() == "deadbeef42"


# ---------------------------------------------------------------------------
# [M] repro_bundle_cmd: manifest-authoritative goldset_hash resolution
# ---------------------------------------------------------------------------


def _run_repro_bundle(cycle: Path, tmp_path: Path) -> tuple[int, str]:
    """Invoke repro_bundle_cmd via Click's CliRunner; return (exit_code, output)."""
    from click.testing import CliRunner

    from engine.cli.pipeline import repro_bundle_cmd

    runner = CliRunner()
    output_path = tmp_path / "out.tar.gz"
    result = runner.invoke(
        repro_bundle_cmd,
        ["--cycle", str(cycle), "--output", str(output_path)],
        catch_exceptions=False,
    )
    return result.exit_code, result.output


def test_repro_bundle_goldset_hash_from_file_only(tmp_path: Path) -> None:
    """When only infer/goldset_hash.txt present, use it (no manifest goldset_hash)."""
    cycle = _make_minimal_cycle(tmp_path)
    (cycle / "infer").mkdir(parents=True)
    (cycle / "infer" / "goldset_hash.txt").write_text("filehash123\n")

    exit_code, _ = _run_repro_bundle(cycle, tmp_path)
    assert exit_code == 0

    bundle = json.loads((cycle / "results" / "reproduction_bundle.json").read_text())
    assert bundle["goldset_hash"] == "filehash123"


def test_repro_bundle_goldset_hash_from_manifest_only(tmp_path: Path) -> None:
    """When only manifest.goldset_hash present (no file), use it."""
    cycle = _make_minimal_cycle(tmp_path, goldset_hash_in_manifest="manhash456")

    exit_code, _ = _run_repro_bundle(cycle, tmp_path)
    assert exit_code == 0

    bundle = json.loads((cycle / "results" / "reproduction_bundle.json").read_text())
    assert bundle["goldset_hash"] == "manhash456"


def test_repro_bundle_prefers_manifest_authoritative(tmp_path: Path) -> None:
    """When file == manifest, assert passes and the hash is recorded."""
    cycle = _make_minimal_cycle(tmp_path, goldset_hash_in_manifest="matchhash789")
    (cycle / "infer").mkdir(parents=True)
    (cycle / "infer" / "goldset_hash.txt").write_text("matchhash789\n")

    exit_code, _ = _run_repro_bundle(cycle, tmp_path)
    assert exit_code == 0

    bundle = json.loads((cycle / "results" / "reproduction_bundle.json").read_text())
    assert bundle["goldset_hash"] == "matchhash789"


def test_repro_bundle_raises_on_goldset_hash_mismatch(tmp_path: Path) -> None:
    """When file != manifest goldset_hash, repro_bundle_cmd must raise (provenance break)."""
    cycle = _make_minimal_cycle(tmp_path, goldset_hash_in_manifest="manifest-hash")
    (cycle / "infer").mkdir(parents=True)
    (cycle / "infer" / "goldset_hash.txt").write_text("DIFFERENT-hash\n")

    from click.testing import CliRunner

    from engine.cli.pipeline import repro_bundle_cmd

    runner = CliRunner()
    output_path = tmp_path / "out.tar.gz"
    result = runner.invoke(
        repro_bundle_cmd,
        ["--cycle", str(cycle), "--output", str(output_path)],
        catch_exceptions=True,
    )
    # Should fail (ClickException → exit_code 1) with provenance-break message
    assert result.exit_code != 0
    assert "provenance break" in (result.output or "").lower() or result.exit_code == 1


def test_repro_bundle_no_goldset_hash_sources_defaults_empty(tmp_path: Path) -> None:
    """When neither file nor manifest goldset_hash exists, bundle records ''."""
    cycle = _make_minimal_cycle(tmp_path)  # no goldset_hash in manifest, no file

    exit_code, _ = _run_repro_bundle(cycle, tmp_path)
    assert exit_code == 0

    bundle = json.loads((cycle / "results" / "reproduction_bundle.json").read_text())
    assert bundle["goldset_hash"] == ""
