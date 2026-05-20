"""Unit tests for engine.repro.bundle."""

from __future__ import annotations

import json
from pathlib import Path

from engine.repro.bundle import ReproductionBundle


def _make_bundle() -> ReproductionBundle:
    return ReproductionBundle(
        cycle_id="cycle-001",
        engine_version="0.1.0",
        snapshot_hash="sha256:abc123",
        manifest_hash="sha256:def456",
        lockfile_hash="sha256:ghi789",
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
