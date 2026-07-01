#!/usr/bin/env python3
"""Author and lock the RARR pre-registration manifest (schema-4, science-identical to 2026).

R4 path-containment: refuses to write outside cycles/2026-rarr/.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.prereg.lock import write_lock
from engine.prereg.rarr_lock import build_rarr_manifest, compute_goldset_hash, load_manifest

# R4: path-containment guard — every write must stay inside this root.
RARR_ROOT = Path("projects/owasp-llm/cycles/2026-rarr").resolve()

C2026 = Path("projects/owasp-llm/cycles/2026")
CRARR = Path("projects/owasp-llm/cycles/2026-rarr")

_MANIFEST_2026 = C2026 / "prereg/manifest.json"
_GOLDSET_RARR = CRARR / "calibration/adjudicated_goldset.jsonl"
_MANIFEST_OUT = CRARR / "prereg/manifest.json"
_LOCK_OUT = CRARR / "prereg/manifest.lock"


def _check_in_rarr_root(dst: Path) -> None:
    """R4: raise ValueError if dst is outside RARR_ROOT."""
    resolved = dst.resolve()
    if not resolved.is_relative_to(RARR_ROOT):
        raise ValueError(
            f"Path-containment violation: {dst!r} resolves to {resolved!r}, "
            f"which is outside RARR_ROOT={RARR_ROOT!r}"
        )


def main() -> None:
    # Load 2026 base manifest (schema-1)
    base = load_manifest(_MANIFEST_2026)

    # Compute goldset hash from the RARR copy
    goldset_hash = compute_goldset_hash(_GOLDSET_RARR)

    # Build the RARR manifest (schema-4, science-identical to 2026)
    rarr = build_rarr_manifest(base, goldset_hash)

    # Write manifest.json — pretty-printed with 2-space indent, matching 2026 style
    _check_in_rarr_root(_MANIFEST_OUT)
    _MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_OUT.write_text(json.dumps(rarr.to_dict(), indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {_MANIFEST_OUT}")

    # Write manifest.lock
    _check_in_rarr_root(_LOCK_OUT)
    h = write_lock(rarr, _LOCK_OUT)
    print(f"Wrote: {_LOCK_OUT}")
    print(f"Manifest hash: {h}")
    print(f"Goldset hash:  {goldset_hash}")


if __name__ == "__main__":
    main()
