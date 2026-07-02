"""Tests for the locked injection probe set (U6 Task 1).

Covers:
- Well-formed probe set (unique ids, non-empty text, real attacker_targets)
- R1: no probe may set attacker_target == 'out-of-scope'
- R2: ≥8 probes spanning ≥4 families; delimiter-forgery + base64 families present
- R7: probe_set_sha256 golden lock — any probe change breaks this test
- FIX 4: every attacker_target is a real entry_id present in the 2026-rarr rubric
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from engine.classify.injection_probes import INJECTION_PROBES, InjectionProbe

# Computed at implementation time (2026-07-01) and locked here.
# Update this value ONLY if a deliberate probe change is made AND the
# change is reviewed and approved — this test is the tamper-evidence seal (R7).
_GOLDEN_PROBE_SET_SHA256 = (
    "a5cd89698e68ba90dd8061a89e3392cce3c98f9ad55058998e213e049eb6a4cf"
)


def test_probe_set_is_locked_and_wellformed() -> None:
    assert len(INJECTION_PROBES) >= 8
    ids = [p.probe_id for p in INJECTION_PROBES]
    assert len(ids) == len(set(ids)), "probe_ids must be unique"
    for p in INJECTION_PROBES:
        assert isinstance(p, InjectionProbe)
        assert p.incident_text.strip()
        assert p.attacker_target  # a concrete entry_id the injection tries to force
        # The probe text must actually attempt the injection (mentions the target
        # or a recognisable override keyword).
        assert (
            (p.attacker_target in p.incident_text)
            or ("ignore" in p.incident_text.lower())
            or ("</incident>" in p.incident_text.lower())
            or ("system" in p.incident_text.lower())
        ), (
            f"Probe {p.probe_id!r} incident_text does not appear to contain "
            f"an injection attempt: {p.incident_text[:80]!r}"
        )


def test_no_probe_targets_out_of_scope() -> None:
    """R1: attacker_target must never be 'out-of-scope'.

    parse_stage2_response collapses BOTH deliberate out-of-scope answers AND
    malformed/timeout responses to that sentinel.  A probe targeting it would
    produce a false-fail signal against robust models whose responses merely
    garbled.  attacker_target must be a plausible-wrong real rubric label.
    """
    for p in INJECTION_PROBES:
        assert p.attacker_target != "out-of-scope", (
            f"Probe {p.probe_id!r} sets attacker_target='out-of-scope' (R1 violation)"
        )


def test_probes_cover_multiple_attack_families() -> None:
    """R2: probes must span ≥4 families; delimiter-forgery + base64 required."""
    fams = {p.probe_id.split("-")[0] for p in INJECTION_PROBES}
    assert len(fams) >= 4, (
        f"probes must span >=4 distinct attack families; got {sorted(fams)}"
    )
    assert "df" in fams, (
        "delimiter-forgery family (prefix 'df') required (R2): "
        "must include probes forging the real INCIDENT_DELIMITER_BEGIN/END tokens"
    )
    assert "b64" in fams, (
        "base64/encoded family (prefix 'b64') required (R2): "
        "must include encoded-payload smuggling probes"
    )


def test_probe_set_integrity_hash() -> None:
    """R7: probe set SHA-256 must match the committed golden value.

    Fails on any change to probe_id, incident_text, or attacker_target.
    Update _GOLDEN_PROBE_SET_SHA256 only after deliberate, reviewed changes.
    """
    digest = hashlib.sha256(
        json.dumps(
            sorted(
                (p.probe_id, p.incident_text, p.attacker_target)
                for p in INJECTION_PROBES
            )
        ).encode()
    ).hexdigest()
    assert digest == _GOLDEN_PROBE_SET_SHA256, (
        f"Probe set has changed — this test is the tamper-evidence seal (R7).\n"
        f"Expected: {_GOLDEN_PROBE_SET_SHA256}\n"
        f"Got:      {digest}\n"
        "Update _GOLDEN_PROBE_SET_SHA256 in this file only after deliberate review."
    )


# ── FIX 4: attacker_target membership in the real 2026-rarr rubric ────────────

def test_attacker_targets_in_real_rubric() -> None:
    """FIX 4: every INJECTION_PROBES attacker_target must be a real entry_id
    present in the 2026-rarr prereg rubric.  Complements the runtime precondition
    added in run_injection_gate (FIX 1) by catching any mismatch at unit-test time,
    before the gate is ever exercised against a live model.
    """
    rubric_path = (
        Path(__file__).parents[2]
        / "projects/owasp-llm/cycles/2026-rarr/prereg/rubric.json"
    )
    rubric = json.loads(rubric_path.read_text())
    rubric_entry_ids = {e["entry_id"] for e in rubric.get("entries", [])}

    for probe in INJECTION_PROBES:
        assert probe.attacker_target in rubric_entry_ids, (
            f"Probe {probe.probe_id!r}: attacker_target={probe.attacker_target!r} "
            f"is not a real entry_id in the 2026-rarr rubric.  "
            f"Known entry_ids: {sorted(rubric_entry_ids)}"
        )
