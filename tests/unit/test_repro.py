"""Unit tests for engine.repro.bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.repro.bundle import ReproductionBundle


def _make_bundle() -> ReproductionBundle:
    return ReproductionBundle(
        cycle_id="cycle-001",
        engine_version="0.1.0",
        snapshot_hash="sha256:abc123",
        manifest_hash="sha256:def456",
        lockfile_hash="sha256:ghi789",
        goldset_hash="none",
        provenance={"adapter": "synthetic", "timestamp": "2026-01-01T00:00:00Z"},
    )


class TestReproductionBundle:
    def test_to_json_roundtrip(self) -> None:
        bundle = _make_bundle()
        j = bundle.to_json()
        parsed = json.loads(j)
        assert parsed["cycle_id"] == "cycle-001"
        assert parsed["engine_version"] == "0.1.0"
        assert parsed["snapshot_hash"] == "sha256:abc123"
        assert parsed["provenance"]["adapter"] == "synthetic"

    def test_json_is_sorted(self) -> None:
        bundle = _make_bundle()
        j = bundle.to_json()
        parsed = json.loads(j)
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        bundle = _make_bundle()
        p = tmp_path / "sub" / "repro.json"
        bundle.write(p)
        assert p.exists()
        restored = ReproductionBundle.read(p)
        assert restored.cycle_id == bundle.cycle_id
        assert restored.engine_version == bundle.engine_version
        assert restored.snapshot_hash == bundle.snapshot_hash
        assert restored.manifest_hash == bundle.manifest_hash
        assert restored.lockfile_hash == bundle.lockfile_hash
        assert restored.provenance == bundle.provenance

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        bundle = _make_bundle()
        p = tmp_path / "a" / "b" / "c" / "repro.json"
        bundle.write(p)
        assert p.exists()

    def test_json_ends_with_newline(self) -> None:
        bundle = _make_bundle()
        assert bundle.to_json().endswith("\n")

    def test_extended_bundle_round_trip(self, tmp_path: Path) -> None:
        b = ReproductionBundle(
            cycle_id="2026",
            engine_version="1.0.0",
            snapshot_hash="snap123",
            manifest_hash="man456",
            lockfile_hash="lock789",
            goldset_hash="none",
            provenance={
                "stage2_manifest_hash": "s2hash",
                "calibration_hash": "calhash",
                "vote_data_hash": "votehash",
            },
        )
        path = tmp_path / "bundle.json"
        b.write(path)
        loaded = ReproductionBundle.read(path)
        assert loaded.provenance["stage2_manifest_hash"] == "s2hash"
        assert loaded.provenance["calibration_hash"] == "calhash"
        assert loaded.provenance["vote_data_hash"] == "votehash"


# ---------------------------------------------------------------------------
# U2-4 tests
# ---------------------------------------------------------------------------

_COMMITTED_2026_BUNDLE = (
    Path(__file__).parent.parent.parent
    / "projects/owasp-llm/cycles/2026/results/reproduction_bundle.json"
)


class TestU24GoldsetHashSentinel:
    def test_write_bundle_goldset_hash_default_empty(self, tmp_path: Path) -> None:
        """write_reproduction_bundle default goldset_hash must be '' not 'none'."""
        from engine.cli.pipeline_executor import write_reproduction_bundle

        write_reproduction_bundle(
            out_dir=tmp_path / "out",
            cycle_id="test",
            engine_version="0.0.0",
            snapshot_hash="snap",
            manifest_hash="man",
            lockfile_hash="lock",
        )
        data = json.loads((tmp_path / "out" / "repro_bundle.json").read_text())
        assert data["goldset_hash"] == "", (
            "write_reproduction_bundle default must be '' (not 'none')"
        )

    def test_legacy_bundle_read_handles_missing_goldset_key(self) -> None:
        """ReproductionBundle.read() must handle legacy bundles that lack goldset_hash."""
        if not _COMMITTED_2026_BUNDLE.exists():
            pytest.skip("committed 2026 bundle not present")
        bundle = ReproductionBundle.read(_COMMITTED_2026_BUNDLE)
        # Legacy bundle has no goldset_hash key → defaulted to "none"
        assert bundle.goldset_hash == "none"
        assert bundle.cycle_id == "2026"

    def test_2026_bundle_bytes_unchanged(self) -> None:
        """U2 must NOT rewrite the committed 2026 reproduction bundle."""
        if not _COMMITTED_2026_BUNDLE.exists():
            pytest.skip("committed 2026 bundle not present")
        raw = _COMMITTED_2026_BUNDLE.read_text()
        data = json.loads(raw)
        # The 2026 bundle has no goldset_hash key — U2 must leave it absent.
        assert "goldset_hash" not in data, (
            "U2 must not add goldset_hash to the committed 2026 bundle bytes"
        )
        # snapshot_hash sentinel must be "none" — unchanged by U2
        assert data["snapshot_hash"] == "none"
        assert data["lockfile_hash"] != ""
