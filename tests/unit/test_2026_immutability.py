"""CI guard: enforce byte-immutability of the 2026 reference cycle.

The 2026 cycle files are used as the baseline for the RARR science-identity
invariant.  Any drift in these files invalidates the lock and must be treated as
a protocol violation requiring full re-lock.

SHA256 values were computed at implementation time (2026-07-01) and are inlined
as authoritative constants.  If any assertion fails with "re-lock required", the
2026 file has changed — do NOT update the hash; investigate the change.
"""

import hashlib
from pathlib import Path

C2026 = Path("projects/owasp-llm/cycles/2026")
SNAP = "24806f1a4f0917f85f7509d6cb2a34b12e56eb902714b37bc2b03a2cf1a246bb"

# Authoritative SHA256 values — computed 2026-07-01, never change.
_EXPECTED: dict[str, str] = {
    "prereg/manifest.json": (
        "125c9d8b742f4a773011ab027358eaf7a2d9d1a059ba30ab017824e1a8ed1e09"
    ),
    "prereg/manifest.lock": (
        "27936a1f376342c55ddfbbb4ace5c49d9c38f092e23116c4279b8d0941117987"
    ),
    "calibration/adjudicated_goldset.jsonl": (
        "f3091eb0a98ce4198e012efc029cb568d90e8c4751fe1803d8d05295abbc418f"
    ),
    f"corpora/genai_agentic/{SNAP}/incidents.json": (
        "24806f1a4f0917f85f7509d6cb2a34b12e56eb902714b37bc2b03a2cf1a246bb"
    ),
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_2026_reference_files_are_byte_immutable() -> None:
    """Fail with 're-lock required' if any 2026 reference file has been modified."""
    for rel, expected_hash in _EXPECTED.items():
        path = C2026 / rel
        assert path.exists(), f"2026 reference file missing: {rel} — re-lock required"
        actual = _sha256(path)
        assert actual == expected_hash, (
            f"2026/{rel} hash changed — re-lock required\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {actual}"
        )
