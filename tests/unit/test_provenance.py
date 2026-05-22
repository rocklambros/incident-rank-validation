"""Tests for engine.calibrate.provenance — stage provenance chain."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.calibrate.provenance import (
    StageProvenance,
    read_provenance,
    verify_input_hashes,
    write_provenance,
)


class TestStageProvenance:
    def test_frozen(self) -> None:
        sp = StageProvenance(
            stage_name="classify",
            manifest_lock_hash="abc",
            input_hashes={"rubric": "def"},
            output_hash="ghi",
            timestamp="2026-06-01T10:00:00-06:00",
            engine_version="0.3.0",
        )
        with pytest.raises(AttributeError):
            sp.stage_name = "mutated"  # type: ignore[misc]

    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        sp = StageProvenance(
            stage_name="classify",
            manifest_lock_hash="lock123",
            input_hashes={"rubric": "rub456", "manifest": "man789"},
            output_hash="out000",
            timestamp="2026-06-01T10:00:00-06:00",
            engine_version="0.3.0",
        )
        path = tmp_path / "classify_provenance.json"
        write_provenance(sp, path)
        loaded = read_provenance(path)
        assert loaded.stage_name == "classify"
        assert loaded.manifest_lock_hash == "lock123"
        assert loaded.input_hashes == {"rubric": "rub456", "manifest": "man789"}
        assert loaded.output_hash == "out000"

    def test_verify_input_hashes_passes(self, tmp_path: Path) -> None:
        prev = StageProvenance(
            stage_name="classify",
            manifest_lock_hash="lock",
            input_hashes={},
            output_hash="classify_out",
            timestamp="2026-06-01T10:00:00-06:00",
            engine_version="0.3.0",
        )
        prev_path = tmp_path / "classify_provenance.json"
        write_provenance(prev, prev_path)
        verify_input_hashes(
            expected={"classify": "classify_out"},
            provenance_dir=tmp_path,
        )

    def test_verify_input_hashes_raises_on_mismatch(self, tmp_path: Path) -> None:
        prev = StageProvenance(
            stage_name="classify",
            manifest_lock_hash="lock",
            input_hashes={},
            output_hash="classify_out",
            timestamp="2026-06-01T10:00:00-06:00",
            engine_version="0.3.0",
        )
        prev_path = tmp_path / "classify_provenance.json"
        write_provenance(prev, prev_path)
        with pytest.raises(ValueError, match="provenance mismatch"):
            verify_input_hashes(
                expected={"classify": "WRONG_HASH"},
                provenance_dir=tmp_path,
            )

    def test_verify_input_hashes_raises_on_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="provenance file not found"):
            verify_input_hashes(
                expected={"classify": "any"},
                provenance_dir=tmp_path,
            )
