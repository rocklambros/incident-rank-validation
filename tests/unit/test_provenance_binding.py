from pathlib import Path

from engine.repro.bundle import ReproductionBundle


def test_bundle_records_goldset_hash(tmp_path: Path):
    b = ReproductionBundle(
        cycle_id="2026-rarr", engine_version="1.3.0",
        snapshot_hash="snap", manifest_hash="man", lockfile_hash="lf",
        goldset_hash="gold-sha", provenance={"calibration_hash": "c"},
    )
    p = tmp_path / "bundle.json"
    b.write(p)
    rt = ReproductionBundle.read(p)
    assert rt.goldset_hash == "gold-sha"
    assert rt.snapshot_hash == "snap"
