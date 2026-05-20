"""Reviewer sign-off record with git-derived timestamp verification (M8).

Each ReviewerSignoff captures who reviewed, what they reviewed (attestation
file + hash), and when — with the timestamp verified against git history
to prevent backdating.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReviewerSignoff:
    """A reviewer's signed attestation, hash-locked and timestamp-verified."""

    reviewer_name: str
    reviewer_affiliation: str
    attestation_relative_path: str  # e.g., "docs/REVIEWERS/reviewer-name-rubric.txt"
    attestation_sha256: str
    signed_at: str  # ISO 8601 — must match git commit timestamp (M8)
    viewed_results_before_signoff: bool

    def verify(self, repo_root: Path) -> None:
        """Verify attestation file exists, hash matches, and signed_at matches git (M8)."""
        p = repo_root / self.attestation_relative_path
        if not p.exists():
            raise FileNotFoundError(f"attestation missing: {p}")
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_hash != self.attestation_sha256:
            raise ValueError(
                f"attestation hash mismatch: file={actual_hash} manifest={self.attestation_sha256}"
            )
        # M8: signed_at must match git commit timestamp
        from engine.prereg.git_timestamp import attestation_signed_at

        git_ts = attestation_signed_at(p, repo_root)
        if self.signed_at != git_ts:
            raise ValueError(
                f"signed_at mismatch: manifest claims {self.signed_at} "
                f"but git records {git_ts}"
            )
