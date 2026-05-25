"""Unit tests for engine.threats.register."""

from __future__ import annotations

from engine.threats.register import Threat, get_threats_register


def test_register_non_empty() -> None:
    threats = get_threats_register()
    assert len(threats) > 0


def test_f_defenseindepth_present() -> None:
    threats = get_threats_register()
    ids = {t.threat_id for t in threats}
    assert "F-defenseindepth" in ids


def test_all_threat_ids_unique() -> None:
    threats = get_threats_register()
    ids = [t.threat_id for t in threats]
    assert len(ids) == len(set(ids)), f"Duplicate threat IDs: {ids}"


def test_all_threats_are_frozen_dataclass() -> None:
    threats = get_threats_register()
    for t in threats:
        assert isinstance(t, Threat)
        # frozen: can't mutate
        try:
            t.threat_id = "mutated"  # type: ignore[misc]
            raise AssertionError("Should be frozen")  # noqa: TRY301
        except AttributeError:
            pass


def test_all_fields_nonempty() -> None:
    for t in get_threats_register():
        assert t.threat_id.strip(), "Empty threat_id"
        assert t.description.strip(), f"Empty description for {t.threat_id}"
        assert t.mitigation.strip(), f"Empty mitigation for {t.threat_id}"
        assert t.residual_risk.strip(), f"Empty residual_risk for {t.threat_id}"


def test_f_aiharm_precision_present() -> None:
    threats = get_threats_register()
    ids = {t.threat_id for t in threats}
    assert "F-aiharm-precision" in ids


def test_f_aiharm_precision_content() -> None:
    threats = get_threats_register()
    threat = next(t for t in threats if t.threat_id == "F-aiharm-precision")
    assert "Beta(1,1)" in threat.description or "Uniform(0,1)" in threat.description
    assert "conservative" not in threat.mitigation.lower()
    assert "3 of 20" in threat.residual_risk
