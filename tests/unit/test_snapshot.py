"""Unit tests for engine.snapshot.hashing and engine.snapshot.provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.snapshot.hashing import snapshot_hash, verify_snapshot_hash
from engine.snapshot.provenance import SnapshotProvenance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTENT = b"hello snapshot\n"
_EXPECTED_HEX = hashlib.sha256(_CONTENT).hexdigest()

_PROVENANCE_KWARGS: dict[str, str] = {
    "source_repo": "genai_agentic_incidents",
    "source_commit_sha": "abc123def456",
    "pull_date": "2026-05-20",
    "adapter_name": "synthetic",
    "adapter_version": "0.1.0",
    "snapshot_hash": _EXPECTED_HEX,
}


def _write_snapshot(tmp_path: Path, content: bytes = _CONTENT) -> Path:
    p = tmp_path / "snapshot.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# snapshot_hash
# ---------------------------------------------------------------------------


def test_snapshot_hash_deterministic(tmp_path: Path) -> None:
    """Same content always produces the same SHA-256 hex digest."""
    p = _write_snapshot(tmp_path)
    assert snapshot_hash(p) == _EXPECTED_HEX


def test_snapshot_hash_changes_with_content(tmp_path: Path) -> None:
    """Different content produces a different hash."""
    p1 = _write_snapshot(tmp_path / "dir_a", b"aaa")
    p2 = _write_snapshot(tmp_path / "dir_b", b"bbb")
    assert snapshot_hash(p1) != snapshot_hash(p2)


# ---------------------------------------------------------------------------
# verify_snapshot_hash
# ---------------------------------------------------------------------------


def test_verify_snapshot_hash_passes(tmp_path: Path) -> None:
    """verify_snapshot_hash does not raise when the hash is correct."""
    p = _write_snapshot(tmp_path)
    verify_snapshot_hash(p, _EXPECTED_HEX)  # must not raise


def test_verify_snapshot_hash_raises_on_mismatch(tmp_path: Path) -> None:
    """verify_snapshot_hash raises ValueError when hashes differ."""
    p = _write_snapshot(tmp_path)
    wrong_hash = "0" * 64
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        verify_snapshot_hash(p, wrong_hash)


# ---------------------------------------------------------------------------
# SnapshotProvenance — JSON round-trip
# ---------------------------------------------------------------------------


def test_provenance_json_roundtrip() -> None:
    """SnapshotProvenance survives a to_json / from_json cycle unchanged."""
    original = SnapshotProvenance(**_PROVENANCE_KWARGS)
    restored = SnapshotProvenance.from_json(original.to_json())
    assert restored == original


def test_provenance_json_sorted_keys() -> None:
    """to_json always emits keys in sorted order (deterministic)."""
    prov = SnapshotProvenance(**_PROVENANCE_KWARGS)
    data = json.loads(prov.to_json())
    keys = list(data.keys())
    assert keys == sorted(keys)


def test_provenance_json_deterministic() -> None:
    """Two identical SnapshotProvenance objects produce identical JSON."""
    p1 = SnapshotProvenance(**_PROVENANCE_KWARGS)
    p2 = SnapshotProvenance(**_PROVENANCE_KWARGS)
    assert p1.to_json() == p2.to_json()


# ---------------------------------------------------------------------------
# SnapshotProvenance — disk round-trip
# ---------------------------------------------------------------------------


def test_provenance_write_read_roundtrip(tmp_path: Path) -> None:
    """SnapshotProvenance.write / .read survives a full disk round-trip."""
    prov = SnapshotProvenance(**_PROVENANCE_KWARGS)
    dest = tmp_path / "run" / "provenance.json"
    prov.write(dest)
    assert dest.exists()
    restored = SnapshotProvenance.read(dest)
    assert restored == prov


def test_provenance_write_creates_parent_dirs(tmp_path: Path) -> None:
    """write() creates missing parent directories."""
    prov = SnapshotProvenance(**_PROVENANCE_KWARGS)
    dest = tmp_path / "a" / "b" / "c" / "provenance.json"
    prov.write(dest)
    assert dest.exists()
