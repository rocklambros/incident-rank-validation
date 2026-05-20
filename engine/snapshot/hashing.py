"""Content-addressed snapshot hashing.

Each cycle vendors a content-hashed snapshot.  See HANDOFF §5.1, §6 control 9.
A cycle refuses to run if the gold-set snapshot hash and the cycle snapshot hash differ.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def snapshot_hash(snapshot_path: Path) -> str:
    """Compute SHA-256 of a snapshot file's content.

    The hash is the content address — a cycle refuses to run
    if the gold-set snapshot hash and the cycle snapshot hash differ.
    """
    h = hashlib.sha256()
    h.update(snapshot_path.read_bytes())
    return h.hexdigest()


def verify_snapshot_hash(snapshot_path: Path, expected_hash: str) -> None:
    """Raise ValueError if snapshot content doesn't match expected hash."""
    actual = snapshot_hash(snapshot_path)
    if actual != expected_hash:
        raise ValueError(
            f"snapshot hash mismatch: expected {expected_hash}, got {actual}"
        )
