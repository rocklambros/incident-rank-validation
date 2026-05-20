"""Hash-locking the pre-registration manifest.

The lock is computed before any concordance number exists (HANDOFF §6
control 1) and verified at infer and decide phases.  If any manifest
field changes after locking, the hash will no longer match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from engine.prereg.manifest import PreregManifest


def compute_lock_hash(manifest: PreregManifest) -> str:
    """Compute SHA-256 of the canonical JSON representation of *manifest*."""
    canonical = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_lock(manifest: PreregManifest, lock_path: Path) -> str:
    """Write the manifest lock file and return the hash."""
    h = compute_lock_hash(manifest)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"manifest_hash": h}, sort_keys=True, indent=2) + "\n"
    )
    return h


def verify_lock(manifest: PreregManifest, lock_path: Path) -> None:
    """Raise ValueError if lock doesn't match current manifest."""
    stored = json.loads(lock_path.read_text())
    actual = compute_lock_hash(manifest)
    if stored["manifest_hash"] != actual:
        raise ValueError(
            f"lock hash mismatch: stored={stored['manifest_hash']}, actual={actual}"
        )
