"""Unit tests for engine.schema canonical data types."""

from __future__ import annotations

import dataclasses

import pytest

from engine.schema import (
    BiasProfile,
    EntryDefinition,
    IncidentRecord,
    make_stratum_size,
)

# ---------------------------------------------------------------------------
# IncidentRecord
# ---------------------------------------------------------------------------

def _make_record(**overrides: object) -> IncidentRecord:
    defaults: dict[str, object] = {
        "id": "CVE-2024-0001",
        "date": "2024-01-15",
        "text": "Critical buffer overflow in libfoo 1.2",
        "severity": "Critical",
        "source_class": "cve",
        "corpus_stratum": "security",
        "quality": "curated",
        "native_labels": ("buffer-overflow", "libfoo"),
        "source_url": "https://nvd.nist.gov/vuln/detail/CVE-2024-0001",
    }
    defaults.update(overrides)
    return IncidentRecord(**defaults)  # type: ignore[arg-type]


class TestIncidentRecord:
    def test_all_fields_accessible(self) -> None:
        r = _make_record()
        assert r.id == "CVE-2024-0001"
        assert r.date == "2024-01-15"
        assert r.text == "Critical buffer overflow in libfoo 1.2"
        assert r.severity == "Critical"
        assert r.source_class == "cve"
        assert r.corpus_stratum == "security"
        assert r.quality == "curated"
        assert r.native_labels == ("buffer-overflow", "libfoo")
        assert r.source_url == "https://nvd.nist.gov/vuln/detail/CVE-2024-0001"

    def test_frozen(self) -> None:
        r = _make_record()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.id = "changed"  # type: ignore[misc]

    def test_severity_may_be_none(self) -> None:
        r = _make_record(severity=None)
        assert r.severity is None

    def test_native_labels_is_tuple(self) -> None:
        r = _make_record(native_labels=("label-a",))
        assert isinstance(r.native_labels, tuple)

    def test_native_labels_empty_tuple_allowed(self) -> None:
        r = _make_record(native_labels=())
        assert r.native_labels == ()

    def test_has_slots(self) -> None:
        # slots=True means __dict__ is absent on instances
        assert not hasattr(_make_record(), "__dict__")


# ---------------------------------------------------------------------------
# BiasProfile
# ---------------------------------------------------------------------------

def _make_bias_profile(**overrides: object) -> BiasProfile:
    defaults: dict[str, object] = {
        "stratum": "security",
        "description": "NVD-sourced; skews toward published CVEs with CVSS scores.",
        "known_blind_spots": ("supply-chain-compromise", "insider-threat"),
        "contamination_description": "bare LLM03 default seed",
        "quarantine_rule": "drop bare ['LLM03'] CVE singletons",
    }
    defaults.update(overrides)
    return BiasProfile(**defaults)  # type: ignore[arg-type]


class TestBiasProfile:
    def test_all_fields_accessible(self) -> None:
        bp = _make_bias_profile()
        assert bp.stratum == "security"
        assert "NVD-sourced" in bp.description
        assert "supply-chain-compromise" in bp.known_blind_spots
        assert bp.contamination_description == "bare LLM03 default seed"
        assert bp.quarantine_rule == "drop bare ['LLM03'] CVE singletons"

    def test_frozen(self) -> None:
        bp = _make_bias_profile()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            bp.stratum = "changed"  # type: ignore[misc]

    def test_known_blind_spots_is_tuple(self) -> None:
        bp = _make_bias_profile()
        assert isinstance(bp.known_blind_spots, tuple)

    def test_known_blind_spots_empty_allowed(self) -> None:
        bp = _make_bias_profile(known_blind_spots=())
        assert bp.known_blind_spots == ()

    def test_has_slots(self) -> None:
        assert not hasattr(_make_bias_profile(), "__dict__")


# ---------------------------------------------------------------------------
# StratumSize / make_stratum_size
# ---------------------------------------------------------------------------

class TestStratumSize:
    def test_positive_value_accepted(self) -> None:
        s = make_stratum_size(1)
        assert s == 1

    def test_large_value_accepted(self) -> None:
        s = make_stratum_size(100_000)
        assert s == 100_000

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="stratum size must be positive"):
            make_stratum_size(0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="stratum size must be positive"):
            make_stratum_size(-1)

    def test_large_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="stratum size must be positive"):
            make_stratum_size(-999)

    def test_is_int(self) -> None:
        s = make_stratum_size(42)
        assert isinstance(s, int)

    def test_is_stratum_size_newtype(self) -> None:
        # NewType is transparent at runtime — the value is just an int
        s = make_stratum_size(7)
        assert type(s) is int  # noqa: E721


# ---------------------------------------------------------------------------
# EntryDefinition
# ---------------------------------------------------------------------------

class TestEntryDefinition:
    def test_all_fields_accessible(self) -> None:
        e = EntryDefinition(entry_id="LLM01", name="Prompt Injection", frame_blind=False)
        assert e.entry_id == "LLM01"
        assert e.name == "Prompt Injection"
        assert e.frame_blind is False

    def test_frame_blind_defaults_to_false(self) -> None:
        e = EntryDefinition(entry_id="STR01", name="Supply Chain Risk")
        assert e.frame_blind is False

    def test_frame_blind_true(self) -> None:
        e = EntryDefinition(entry_id="LLM03", name="Training Data Poisoning", frame_blind=True)
        assert e.frame_blind is True

    def test_frozen(self) -> None:
        e = EntryDefinition(entry_id="LLM01", name="Prompt Injection")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            e.entry_id = "changed"  # type: ignore[misc]

    def test_has_slots(self) -> None:
        e = EntryDefinition(entry_id="LLM01", name="Prompt Injection")
        assert not hasattr(e, "__dict__")
