"""Science-identity diff-audit: permanent regression guard for the RARR cycle.

Every PreregManifest field that is NOT in GOVERNANCE_OVERRIDE_FIELDS must be
byte-identical between the 2026 reference cycle and the 2026-rarr cycle.  If this
test fails, inference science has drifted — fix the builder, not this test.

Intentional stage2 schema delta vs 2026 (lock-consistent; R6 disclosure)
------------------------------------------------------------------------
The stage2_manifest.json introduces 4 fields that have no counterpart in the
2026 inference schema:
  - selected_from        — names the authoritative bake-off grid
  - abort_factor         — cost-safety multiplier (1.2×); not a science parameter
  - injection_gate_passed — post-hoc gate result; null at lock time
  - injection_gate_revision_sha — provenance for the gate; null at lock time

These fields are consumed ONLY via snapshot_hash / stage2 bookkeeping and do NOT
alter the inference science (priors, specs, thresholds, goldset, rubric).  They are
therefore NOT in GOVERNANCE_OVERRIDE_FIELDS and NOT in SCIENTIFIC_FIELDS — they
live exclusively in stage2_manifest.json, not in the PreregManifest dataclass.

Pre-registered K∈{6,8,10} sensitivity-grid obligation
------------------------------------------------------
The K=8 flag-not-widen decision (recall_min_denominator=8) carries a commitment to
report sensitivity over K∈{6,8,10} in U9.  This is a disclosure obligation, not a
science parameter — it does not affect any manifest field beyond
recall_min_denominator_rationale (which IS in GOVERNANCE_OVERRIDE_FIELDS).
"""

import dataclasses
from pathlib import Path

from engine.prereg.manifest import PreregManifest
from engine.prereg.rarr_lock import (
    GOVERNANCE_OVERRIDE_FIELDS,
    SCIENTIFIC_FIELDS,
    build_rarr_manifest,
    compute_goldset_hash,
    load_manifest,
)

C2026 = Path("projects/owasp-llm/cycles/2026/prereg/manifest.json")
CRARR = Path("projects/owasp-llm/cycles/2026-rarr/prereg/manifest.json")

_load = load_manifest  # verified loader; PreregManifest has NO from_dict


def test_every_scientific_field_is_identical_between_2026_and_rarr() -> None:
    a, b = _load(C2026), _load(CRARR)
    diffs = {
        f: (getattr(a, f), getattr(b, f))
        for f in SCIENTIFIC_FIELDS
        if getattr(a, f) != getattr(b, f)
    }
    assert diffs == {}, f"science drifted between 2026 and RARR: {diffs}"


def test_the_only_differing_fields_are_the_allowlist() -> None:
    a, b = _load(C2026), _load(CRARR)
    all_fields = [f.name for f in dataclasses.fields(PreregManifest)]
    differing = {f for f in all_fields if getattr(a, f) != getattr(b, f)}
    assert differing.issubset(set(GOVERNANCE_OVERRIDE_FIELDS)), (
        f"unexpected non-allowlisted difference: "
        f"{differing - set(GOVERNANCE_OVERRIDE_FIELDS)}"
    )


def test_snapshot_and_taxonomy_and_rubric_hashes_match() -> None:
    a, b = _load(C2026), _load(CRARR)
    assert a.snapshot_hash == b.snapshot_hash
    assert a.taxonomy_hash == b.taxonomy_hash
    assert a.rubric_hash == b.rubric_hash


def test_builder_override_kwargs_match_governance_set() -> None:
    """R6: guard against silent drift — the set of fields the builder actually sets
    must stay in sync with GOVERNANCE_OVERRIDE_FIELDS in both directions.

    Two failure modes caught:
    (A) A field in GOVERNANCE_OVERRIDE_FIELDS is no longer a real PreregManifest field
        (stale name after a refactor).
    (B) build_rarr_manifest changes a field that is NOT in GOVERNANCE_OVERRIDE_FIELDS
        (inadvertent science mutation).

    Note: some governance fields (recall_min_denominator_gate=False,
    recall_floor_epsilon=0.0) share the same default value in 2026 and RARR, so
    comparing "differing fields" would miss them.  We therefore also assert the
    expected RARR values directly rather than relying purely on the diff.
    """
    all_field_names = {f.name for f in dataclasses.fields(PreregManifest)}

    # (A) Every governance field name must exist in the dataclass.
    stale = set(GOVERNANCE_OVERRIDE_FIELDS) - all_field_names
    assert stale == set(), f"stale GOVERNANCE_OVERRIDE_FIELDS name(s): {stale}"

    base = _load(C2026)
    gs_hash = compute_goldset_hash(
        Path("projects/owasp-llm/cycles/2026-rarr/calibration/adjudicated_goldset.jsonl")
    )
    rarr = build_rarr_manifest(base, gs_hash)

    # Verify the builder sets every known RARR governance value correctly.
    expected: dict[str, object] = {
        "cycle_id": "2026-rarr",
        "schema_version": 4,
        "goldset_hash": gs_hash,
        "recall_min_denominator": 8,
        "recall_min_denominator_gate": False,
        "recall_floor_epsilon": 0.0,
        "prospective_power_target_kappa": 0.40,
        "prospective_power_confidence_level": 0.95,
        "prospective_power_1_minus_beta": 0.80,
    }
    for field, exp_val in expected.items():
        assert getattr(rarr, field) == exp_val, (
            f"governance field {field!r} not set to expected value by builder"
        )
    assert rarr.recall_min_denominator_rationale != "", (
        "recall_min_denominator_rationale must be non-empty (K=8 rationale)"
    )

    # (B) No field outside the governance set was changed.
    changed_outside = {
        f
        for f in all_field_names
        if f not in GOVERNANCE_OVERRIDE_FIELDS
        and getattr(base, f) != getattr(rarr, f)
    }
    assert changed_outside == set(), (
        f"build_rarr_manifest changed non-governance fields: {changed_outside}"
    )
