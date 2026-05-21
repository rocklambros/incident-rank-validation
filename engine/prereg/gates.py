"""Pre-classify gate checks and publishability verification.

These functions enforce HANDOFF §6 controls at the module level. The CLI
commands (engine/cli/rubric.py) call these; the functions are also
independently testable.
"""
from __future__ import annotations

from pathlib import Path

from engine.prereg.manifest import PreregManifest


def _normalize_name(name: str) -> str:
    """Collapse whitespace and lowercase for name comparison."""
    return " ".join(name.lower().split())


def require_rubric_attestation(manifest: PreregManifest) -> None:
    """Raise if rubric drafting attestation is missing (HANDOFF §6 control 11(d))."""
    if manifest.rubric_drafting_attestation is None:
        raise ValueError(
            "rubric drafting attestation required before classify — "
            "populate rubric_attestation.json first"
        )


def require_rubric_hash(manifest: PreregManifest) -> None:
    """Raise if rubric hash is missing from the manifest."""
    if manifest.rubric_hash is None:
        raise ValueError(
            "rubric hash required before classify — "
            "freeze the rubric first via `freeze-rubric`"
        )


def require_rubric_hash_match(
    manifest: PreregManifest, rubric_path: Path
) -> None:
    """Raise if manifest.rubric_hash does not match the actual rubric file.

    This closes the integrity gap where the rubric file could be modified
    after freeze without detection.  Called at classify time.
    """
    from engine.prereg.rubric_io import read_rubric

    if manifest.rubric_hash is None:
        raise ValueError("rubric hash is None in manifest")
    rubric = read_rubric(rubric_path)
    actual_hash = rubric.compute_hash()
    if manifest.rubric_hash != actual_hash:
        raise ValueError(
            f"rubric hash mismatch: manifest={manifest.rubric_hash}, "
            f"file={actual_hash}. Was rubric.json modified after freeze?"
        )


def is_publishable(manifest: PreregManifest, *, ranking_author: str) -> bool:
    """Check publication readiness including reviewer independence.

    Combines the manifest's mechanical ``non_publishable`` derivation with
    the discipline-based reviewer-independence check (HANDOFF §4 Crosswalk
    authorship + REVIEWERS.md PRE-PUBLISH CHECKLIST).

    Name comparison is normalized (case-insensitive, whitespace-collapsed)
    to prevent accidental bypass via formatting differences.
    """
    if manifest.non_publishable:
        return False
    author_norm = _normalize_name(ranking_author)
    if (
        manifest.rubric_reviewer is not None
        and _normalize_name(manifest.rubric_reviewer.reviewer_name) == author_norm
    ):
        return False
    return not (
        manifest.statistical_reviewer is not None
        and _normalize_name(manifest.statistical_reviewer.reviewer_name)
        == author_norm
    )
