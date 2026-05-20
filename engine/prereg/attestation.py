"""Git working-tree attestation checks.

Verifies that a file is committed to git and the working tree copy
matches HEAD.  Used to ensure that pre-registered artefacts cannot be
silently modified after locking.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class AttestationError(RuntimeError):
    """Raised when a file fails a git attestation check."""


def verify_committed(file_path: Path, repo_root: Path) -> None:
    """Raise AttestationError if file is not committed or differs from HEAD."""
    rel = file_path.relative_to(repo_root)

    # Check if file is tracked
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AttestationError(f"{rel} is not tracked by git")

    # Check if working tree matches HEAD
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(rel)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AttestationError(f"{rel} has uncommitted changes")
