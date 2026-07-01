#!/usr/bin/env python3
"""Scaffold the 2026-rarr cycle directory with byte-identical copies of 2026 inputs.

Idempotent: if dst already exists with matching hash → skip.
Write-safe: never overwrites a file with differing content (raises ValueError).
Path-containment: every destination must be under RARR_ROOT or raises ValueError (R4).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

# R4: path-containment guard — every write must stay inside this root.
RARR_ROOT = Path("projects/owasp-llm/cycles/2026-rarr").resolve()
C2026 = Path("projects/owasp-llm/cycles/2026")
CRARR = Path("projects/owasp-llm/cycles/2026-rarr")

SNAP = "24806f1a4f0917f85f7509d6cb2a34b12e56eb902714b37bc2b03a2cf1a246bb"

# Subdirectories to create under CRARR
SUBDIRS = [
    "prereg",
    "corpora",
    f"corpora/genai_agentic/{SNAP}",
    "calibration",
    "classify",
    "infer",
    "results",
    "vote",
    "polling",
    "taxonomy",
]

# Files to copy byte-identical from 2026
COPY_RELS = [
    "prereg/rubric.json",
    "prereg/rubric_attestation.json",
    "taxonomy/taxonomy.json",
    "calibration/posteriors.json",
    "calibration/adjudicated_goldset.jsonl",
    f"corpora/genai_agentic/{SNAP}/incidents.json",
    f"corpora/genai_agentic/{SNAP}/incidents.jsonl",
    f"corpora/genai_agentic/{SNAP}/provenance.json",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _guard(dst: Path) -> None:
    """R4: raise if dst is outside RARR_ROOT."""
    resolved = dst.resolve()
    if not resolved.is_relative_to(RARR_ROOT):
        raise ValueError(
            f"Path-containment violation: {dst!r} resolves to {resolved!r}, "
            f"which is outside RARR_ROOT={RARR_ROOT!r}"
        )


def _safe_copy(src: Path, dst: Path) -> str:
    """Copy src → dst with path-containment + idempotency guards.

    Returns 'skipped', 'copied', or raises ValueError on conflict.
    """
    _guard(dst)
    if dst.exists():
        if _sha(dst) == _sha(src):
            return "skipped"
        raise ValueError(
            f"Hash-before-overwrite: {dst} exists with DIFFERING content vs {src}. "
            "Remove manually if intentional."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def main() -> None:
    # Create all subdirectories
    for sub in SUBDIRS:
        d = CRARR / sub
        _guard(d)
        d.mkdir(parents=True, exist_ok=True)

    # Copy byte-identical inputs
    for rel in COPY_RELS:
        src = C2026 / rel
        dst = CRARR / rel
        status = _safe_copy(src, dst)
        print(f"  {status:7s}  {rel}")

    print(f"\nScaffold complete: {CRARR}")


if __name__ == "__main__":
    main()
