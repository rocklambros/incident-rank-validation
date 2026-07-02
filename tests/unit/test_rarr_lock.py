"""T3: RARR manifest builder, loader, and lock-verification tests.

R6 note: test_builder_sets_governance_and_identity also proves the positive __post_init__
activation path — recall_min_denominator=8 + schema_version=4 triggers the F6 guard (active
but NOT raising because schema_version >= 3), and the three power fields trigger the D6 guard
(active but NOT raising because schema_version >= 4). No ValueError = guards activated cleanly.
"""
import hashlib
from pathlib import Path

import pytest

from engine.prereg.lock import compute_lock_hash, verify_lock
from engine.prereg.rarr_lock import (
    SCIENTIFIC_FIELDS,
    build_rarr_manifest,
    compute_goldset_hash,
    load_manifest,
)

C2026 = Path("projects/owasp-llm/cycles/2026")
CRARR = Path("projects/owasp-llm/cycles/2026-rarr")

_load = load_manifest  # verified loader (field-filter); PreregManifest has NO from_dict


def test_both_schemas_load_and_locks_verify() -> None:
    # R1: prove the verified loader round-trips BOTH schema tiers and each lock verifies.
    m2026 = load_manifest(C2026 / "prereg/manifest.json")
    assert m2026.schema_version == 1
    verify_lock(m2026, C2026 / "prereg/manifest.lock")  # raises on mismatch
    mrarr = load_manifest(CRARR / "prereg/manifest.json")
    assert mrarr.schema_version == 4
    verify_lock(mrarr, CRARR / "prereg/manifest.lock")


def test_goldset_hash_matches_reused_2026_goldset() -> None:
    gs = CRARR / "calibration/adjudicated_goldset.jsonl"
    expected = hashlib.sha256(gs.read_bytes()).hexdigest()
    assert compute_goldset_hash(gs) == expected


def test_builder_copies_every_scientific_field_verbatim() -> None:
    base = _load(C2026 / "prereg/manifest.json")
    gh = compute_goldset_hash(CRARR / "calibration/adjudicated_goldset.jsonl")
    rarr = build_rarr_manifest(base, gh)
    for f in SCIENTIFIC_FIELDS:
        assert getattr(rarr, f) == getattr(base, f), f"scientific field drifted: {f}"


def test_builder_sets_governance_and_identity() -> None:
    # R6: positive __post_init__ activation — schema_version=4 with F6+D6 fields set
    # must not raise ValueError, proving guards activated cleanly.
    base = _load(C2026 / "prereg/manifest.json")
    gh = compute_goldset_hash(CRARR / "calibration/adjudicated_goldset.jsonl")
    rarr = build_rarr_manifest(base, gh)  # would raise if guards misfired
    assert rarr.cycle_id == "2026-rarr"
    assert rarr.schema_version == 4
    assert rarr.goldset_hash == gh
    assert rarr.recall_min_denominator == 8
    assert rarr.recall_min_denominator_gate is False
    assert rarr.recall_floor_epsilon == 0.0
    assert rarr.prospective_power_target_kappa == 0.40
    assert rarr.prospective_power_confidence_level == 0.95
    assert rarr.prospective_power_1_minus_beta == 0.80


def test_committed_manifest_roundtrips_and_lock_verifies() -> None:
    committed = _load(CRARR / "prereg/manifest.json")
    base = _load(C2026 / "prereg/manifest.json")
    gh = compute_goldset_hash(CRARR / "calibration/adjudicated_goldset.jsonl")
    rebuilt = build_rarr_manifest(base, gh)
    assert compute_lock_hash(committed) == compute_lock_hash(rebuilt)  # committed == builder output
    verify_lock(committed, CRARR / "prereg/manifest.lock")  # raises on mismatch


def test_r4_guard_raises_for_dst_inside_cycles_2026() -> None:
    """R4: path-containment guard in author_rarr_manifest must reject paths inside cycles/2026/."""
    from tools.author_rarr_manifest import _check_in_rarr_root

    bad_dst = Path("projects/owasp-llm/cycles/2026/prereg/manifest.json").resolve()
    with pytest.raises(ValueError, match="Path-containment violation"):
        _check_in_rarr_root(bad_dst)
