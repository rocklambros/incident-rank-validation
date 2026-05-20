"""Derive signoff timing from git history (M8).

HANDOFF v2.5 §6 control 11(e): signoff to precede the first infer run.
Self-declared signed_at strings are tamperable; git-derived timestamps are not
(without rewriting history, which verify_committed would catch).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitTimestampError(RuntimeError):
    """Raised when a git commit timestamp cannot be determined."""


def attestation_signed_at(attestation_path: Path, repo_root: Path) -> str:
    """Return ISO 8601 timestamp of the commit that introduced this file."""
    rel = attestation_path.relative_to(repo_root)
    res = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", str(rel)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        raise GitTimestampError(
            f"could not determine git commit timestamp for {rel}; "
            "file may not be committed yet"
        )
    return res.stdout.strip()
