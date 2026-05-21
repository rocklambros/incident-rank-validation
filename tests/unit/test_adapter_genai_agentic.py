"""Tests for the genai_agentic corpus A adapter.

All test counts and field references are derived from the audit in
HANDOFF §3 (owasp-mapping-quality-audit.md, N=7,714) and confirmed
against the vendored snapshot in Task 0 Step 7.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.adapters.genai_agentic import GenAIAgenticAdapter
from engine.adapters.genai_agentic_bias import (
    build_bias_profiles,
    is_bare_llm03_contaminated,
    is_double_default_contaminated,
)
from engine.model.overlap import OverlapWeights
from engine.schema import BiasProfile, IncidentRecord


class TestBiasProfiles:
    """Per-stratum bias profile declarations (HANDOFF §3 Mixture, §4 row, §5.1)."""

    def test_one_profile_per_stratum(self) -> None:
        profiles = build_bias_profiles()
        strata = {p.stratum for p in profiles}
        assert "security" in strata
        assert "ai-harm" in strata
        assert len(profiles) >= 2

    def test_profiles_are_biasprofile_instances(self) -> None:
        for p in build_bias_profiles():
            assert isinstance(p, BiasProfile)

    def test_security_stratum_declares_contamination(self) -> None:
        sec = next(p for p in build_bias_profiles() if p.stratum == "security")
        assert "LLM03" in sec.contamination_description
        assert sec.quarantine_rule != ""

    def test_ai_harm_stratum_declares_known_blind_spots(self) -> None:
        ah = next(p for p in build_bias_profiles() if p.stratum == "ai-harm")
        assert len(ah.known_blind_spots) > 0

    def test_construction_time_validation_rejects_empty_stratum(self) -> None:
        """C2 pattern: invalid input fails at construction, not at use time."""
        from engine.adapters.genai_agentic_bias import _validate_bias_profile
        with pytest.raises(ValueError, match="stratum"):
            _validate_bias_profile(BiasProfile(
                stratum="",
                description="empty",
                known_blind_spots=(),
                contamination_description="none",
                quarantine_rule="none",
            ))


class TestQuarantinePredicates:
    """Contamination quarantine rules (HANDOFF §3 F2, §5.2 out-of-scope sink)."""

    def test_bare_llm03_detected(self) -> None:
        assert is_bare_llm03_contaminated(["LLM03"]) is True

    def test_bare_llm03_not_triggered_on_multi_label(self) -> None:
        assert is_bare_llm03_contaminated(["LLM03", "LLM05"]) is False

    def test_double_default_detected(self) -> None:
        assert is_double_default_contaminated(["LLM03", "ASI04"]) is True
        assert is_double_default_contaminated(["ASI04", "LLM03"]) is True

    def test_double_default_not_triggered_on_triple(self) -> None:
        assert is_double_default_contaminated(["LLM03", "ASI04", "LLM05"]) is False

    def test_empty_labels_not_contaminated(self) -> None:
        assert is_bare_llm03_contaminated([]) is False
        assert is_double_default_contaminated([]) is False


# ---------------------------------------------------------------------------
# Fixtures for adapter core tests (Task 4)
# ---------------------------------------------------------------------------

@pytest.fixture()
def vendored_snapshot(tmp_path: Path) -> Path:
    """Create a minimal vendored snapshot for testing."""
    records = [
        {
            "id": "INC-001",
            "title": "Prompt injection in chatbot",
            "description": "Attacker injected malicious prompts.",
            "date": "2024-03-15",
            "severity": "High",
            "corpus": "security",
            "category": "real-world",
            "owasp_llm": ["LLM01"],
            "quality_tier": "curated",
            "references": [{"title": "Example", "url": "https://example.com/inc-001"}],
        },
        {
            "id": "INC-002",
            "title": "AI bias incident",
            "description": "Model produced biased outputs.",
            "date": "2024-06-20",
            "severity": "Medium",
            "corpus": "ai-harm",
            "category": "real-world",
            "owasp_llm": ["LLM06"],
            "quality_tier": "reviewed",
            "references": [{"title": "Example", "url": "https://example.com/inc-002"}],
        },
        {
            "id": "INC-003",
            "title": "Generic CVE with default label",
            "description": "CVE with no human OWASP review.",
            "date": "2024-01-10",
            "severity": "Medium",
            "corpus": "security",
            "category": "vulnerability-disclosure",
            "owasp_llm": ["LLM03"],
            "quality_tier": "reviewed",
            "references": [{"title": "Example", "url": "https://example.com/inc-003"}],
        },
        {
            "id": "INC-004",
            "title": "Double default label",
            "description": "CVE with LLM03+ASI04 default.",
            "date": "2024-02-28",
            "severity": "Medium",
            "corpus": "security",
            "category": "vulnerability-disclosure",
            "owasp_llm": ["LLM03", "ASI04"],
            "quality_tier": "reviewed",
            "references": [{"title": "Example", "url": "https://example.com/inc-004"}],
        },
        {
            "id": "INC-005",
            "title": "Future-dated incident",
            "description": "This incident is dated after snapshot.",
            "date": "2027-01-01",
            "severity": "Low",
            "corpus": "security",
            "category": "real-world",
            "owasp_llm": ["LLM05"],
            "quality_tier": "reviewed",
            "references": [{"title": "Example", "url": "https://example.com/inc-005"}],
        },
    ]
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    # Write in the wrapper format that the real corpus uses
    wrapped = {
        "version": "2.0.0",
        "generated": "2026-05-17",
        "description": "test",
        "schema": {},
        "incident_count": len(records),
        "incidents": records,
    }
    (snapshot_dir / "incidents.json").write_text(json.dumps(wrapped))
    return snapshot_dir


# ---------------------------------------------------------------------------
# Adapter core tests (Task 4)
# ---------------------------------------------------------------------------

class TestGenAIAgenticAdapter:

    def test_emits_incident_records(self, vendored_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        records = list(adapter.iter_incidents())
        assert len(records) > 0
        for r in records:
            assert isinstance(r, IncidentRecord)

    def test_future_dated_records_are_excluded(self, vendored_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        records = list(adapter.iter_incidents())
        # INC-005 is dated 2027-01-01 — must be excluded
        ids = {r.id for r in records}
        assert "INC-005" not in ids
        assert len(records) == 4

    def test_schema_round_trip_all_fields_populated(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        for r in adapter.iter_incidents():
            assert r.id != ""
            assert r.date != ""
            assert r.text != ""
            assert r.source_class != ""
            assert r.corpus_stratum in ("security", "ai-harm")
            assert r.quality in ("curated", "reviewed", "auto")
            assert isinstance(r.native_labels, tuple)
            assert r.source_url.startswith("http")

    def test_corpus_stratum_matches_source_corpus_field(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        strata = {r.corpus_stratum for r in adapter.iter_incidents()}
        assert "security" in strata
        assert "ai-harm" in strata

    def test_source_class_mapping(self, vendored_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        records = {r.id: r for r in adapter.iter_incidents()}
        # "real-world" -> "harm-report"
        assert records["INC-001"].source_class == "harm-report"
        # "vulnerability-disclosure" -> "cve"
        assert records["INC-003"].source_class == "cve"

    def test_native_labels_are_non_authoritative_metadata(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        for r in adapter.iter_incidents():
            assert isinstance(r.native_labels, tuple)

    def test_source_url_extracted_from_references(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        records = {r.id: r for r in adapter.iter_incidents()}
        assert records["INC-001"].source_url == "https://example.com/inc-001"

    def test_severity_defaulted_for_non_curated_medium(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        records = {r.id: r for r in adapter.iter_incidents()}
        # INC-001: curated + High -> severity preserved
        assert records["INC-001"].severity == "High"
        # INC-002: reviewed + Medium -> severity defaulted to None
        assert records["INC-002"].severity is None
        # INC-003: reviewed + Medium -> severity defaulted to None
        assert records["INC-003"].severity is None

    def test_construction_validates_snapshot_dir_exists(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            GenAIAgenticAdapter(
                snapshot_dir=tmp_path / "nonexistent",
                snapshot_date="2026-05-20",
            )

    def test_construction_validates_incidents_json_exists(
        self, tmp_path: Path
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="incidents.json"):
            GenAIAgenticAdapter(
                snapshot_dir=empty_dir,
                snapshot_date="2026-05-20",
            )


class TestTextLengthCap:
    def test_oversized_text_is_truncated(self, tmp_path: Path) -> None:
        from engine.adapters.genai_agentic import _MAX_TEXT_LENGTH

        oversized_desc = "x" * (_MAX_TEXT_LENGTH + 1000)
        records = [
            {
                "id": "OVER-001",
                "title": "Normal title",
                "description": oversized_desc,
                "date": "2024-01-01",
                "severity": "High",
                "corpus": "security",
                "category": "real-world",
                "owasp_llm": ["LLM01"],
                "quality_tier": "curated",
                "references": [
                    {"title": "Example", "url": "https://example.com/over"}
                ],
            },
        ]
        snap = tmp_path / "snap"
        snap.mkdir()
        wrapped = {"version": "2.0.0", "incidents": records}
        (snap / "incidents.json").write_text(json.dumps(wrapped))
        adapter = GenAIAgenticAdapter(snapshot_dir=snap, snapshot_date="2026-05-20")
        r = next(adapter.iter_incidents())
        assert len(r.text) <= _MAX_TEXT_LENGTH

    def test_normal_text_is_not_truncated(self, tmp_path: Path) -> None:
        records = [
            {
                "id": "NORM-001",
                "title": "Short",
                "description": "Also short",
                "date": "2024-01-01",
                "severity": "Low",
                "corpus": "security",
                "category": "real-world",
                "owasp_llm": ["LLM02"],
                "quality_tier": "curated",
                "references": [
                    {"title": "Example", "url": "https://example.com/norm"}
                ],
            },
        ]
        snap = tmp_path / "snap"
        snap.mkdir()
        wrapped = {"version": "2.0.0", "incidents": records}
        (snap / "incidents.json").write_text(json.dumps(wrapped))
        adapter = GenAIAgenticAdapter(snapshot_dir=snap, snapshot_date="2026-05-20")
        r = next(adapter.iter_incidents())
        assert r.text == "Short Also short"


class TestSeverityDefaultDetection:
    """HANDOFF §3: severity defaulted to 'Medium' is an artifact, not truth."""

    def test_curated_medium_is_not_defaulted(
        self, vendored_snapshot: Path
    ) -> None:
        """Curated records have human-confirmed severity — keep 'Medium'."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_001 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-001"
        )
        # INC-001 is curated with severity High — kept as-is
        assert inc_001.severity == "High"

    def test_reviewed_medium_is_treated_as_unknown(
        self, vendored_snapshot: Path
    ) -> None:
        """Non-curated 'Medium' severity is a source-ingest artifact → None."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_002 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-002"
        )
        # INC-002 is reviewed (not curated) with severity Medium → defaulted → None
        assert inc_002.severity is None


class TestFutureDatedRowDrop:
    """HANDOFF §4 Temporal: adapter drops rows dated after the snapshot date."""

    def test_future_dated_rows_are_dropped(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        ids = {r.id for r in adapter.iter_incidents()}
        assert "INC-005" not in ids  # dated 2027-01-01

    def test_past_dated_rows_are_kept(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        ids = {r.id for r in adapter.iter_incidents()}
        assert "INC-001" in ids  # dated 2024-03-15


class TestContaminationQuarantine:
    """HANDOFF §3 F2, §5.2: bare-LLM03 and double-default quarantine."""

    def test_bare_llm03_record_is_emitted_but_flagged(
        self, vendored_snapshot: Path
    ) -> None:
        """Quarantined records are still emitted — downstream routes to sink."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_003 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-003"
        )
        assert inc_003.native_labels == ("LLM03",)
        assert is_bare_llm03_contaminated(list(inc_003.native_labels))

    def test_double_default_record_is_emitted_but_flagged(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_004 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-004"
        )
        assert set(inc_004.native_labels) == {"LLM03", "ASI04"}
        assert is_double_default_contaminated(list(inc_004.native_labels))

    def test_clean_record_is_not_quarantined(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_001 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-001"
        )
        assert not is_bare_llm03_contaminated(list(inc_001.native_labels))
        assert not is_double_default_contaminated(list(inc_001.native_labels))


class TestAdapterInterface:
    """All five CorpusAdapter ABC methods return valid data."""

    def test_bias_profiles_cover_all_strata(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        profile_strata = {p.stratum for p in adapter.bias_profiles()}
        record_strata = {r.corpus_stratum for r in adapter.iter_incidents()}
        assert record_strata.issubset(profile_strata), (
            f"Records have strata {record_strata} but profiles only cover {profile_strata}"
        )

    def test_stratum_sizes_are_positive(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        sizes = adapter.stratum_sizes()
        assert len(sizes) > 0
        for stratum, size in sizes.items():
            assert size > 0, f"Stratum {stratum} has non-positive size {size}"

    def test_stratum_size_gte_observed_count(
        self, vendored_snapshot: Path
    ) -> None:
        """M3 sanity contract: stratum_size >= observed incident count."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        sizes = adapter.stratum_sizes()
        counts: dict[str, int] = {}
        for r in adapter.iter_incidents():
            counts[r.corpus_stratum] = counts.get(r.corpus_stratum, 0) + 1
        for stratum, count in counts.items():
            assert sizes[stratum] >= count, (
                f"M3 violation: stratum {stratum} size {sizes[stratum]} < count {count}"
            )

    def test_entry_definitions_returns_twenty_entries(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        entries = adapter.entry_definitions()
        assert len(entries) == 20

    def test_entry_definitions_include_frame_blind_entries(
        self, vendored_snapshot: Path
    ) -> None:
        """HANDOFF §3 F4: LLM04, LLM08, LLM10 are near-absent."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        entries = {e.entry_id: e for e in adapter.entry_definitions()}
        assert entries["LLM04"].frame_blind is True
        assert entries["LLM08"].frame_blind is True
        assert entries["LLM10"].frame_blind is True

    def test_overlap_weights_returns_valid_structure(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        ow = adapter.overlap_weights()
        assert isinstance(ow, OverlapWeights)


class TestEntryDefinitions:
    """Entry definition expansion: 10 core + 6 NEW-* + 4 ROLL-* = 20 entries."""

    def test_twenty_entries(self, vendored_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        entries = adapter.entry_definitions()
        assert len(entries) == 20

    def test_frame_blind_entries(self, vendored_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        entries = adapter.entry_definitions()
        fb = {e.entry_id for e in entries if e.frame_blind}
        assert fb == {"LLM04", "LLM08", "LLM10"}

    def test_all_entry_ids_present(self, vendored_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        entries = adapter.entry_definitions()
        ids = {e.entry_id for e in entries}
        expected = {
            "LLM01", "LLM02", "LLM03", "LLM04", "LLM05",
            "LLM06", "LLM07", "LLM08", "LLM09", "LLM10",
            "NEW-PMP", "NEW-MTIE", "NEW-MA", "NEW-ITSCD", "NEW-WLA", "NEW-MSDA",
            "ROLL-CMSB", "ROLL-LAPTF", "ROLL-SICG", "ROLL-CFAS",
        }
        assert ids == expected


# ---------------------------------------------------------------------------
# Real-snapshot integration tests (Task 7)
# ---------------------------------------------------------------------------

_VENDOR_BASE = Path("projects/owasp-llm/cycles/2026/corpora/genai_agentic")


def _find_vendored_snapshot() -> Path | None:
    """Find the vendored snapshot directory, or None if not yet vendored."""
    if not _VENDOR_BASE.exists():
        return None
    subdirs = [d for d in _VENDOR_BASE.iterdir() if d.is_dir()]
    if len(subdirs) != 1:
        return None
    return subdirs[0]


@pytest.fixture()
def real_snapshot() -> Path:
    """Return the path to the real vendored snapshot, skip if not available."""
    snap = _find_vendored_snapshot()
    if snap is None:
        pytest.skip("Vendored snapshot not available — run Task 3 first")
    return snap


class TestRealSnapshotIntegration:
    """Tests against the actual vendored genai_agentic snapshot."""

    EXPECTED_TOTAL_MIN = 7_000
    EXPECTED_TOTAL_MAX = 9_000
    EXPECTED_SECURITY_MIN = 6_500
    EXPECTED_SECURITY_MAX = 8_500
    EXPECTED_AI_HARM_MIN = 300
    EXPECTED_AI_HARM_MAX = 500

    def test_total_record_count_within_tolerance(
        self, real_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        total = sum(1 for _ in adapter.iter_incidents())
        assert self.EXPECTED_TOTAL_MIN <= total <= self.EXPECTED_TOTAL_MAX, (
            f"Total records {total} outside expected range "
            f"[{self.EXPECTED_TOTAL_MIN}, {self.EXPECTED_TOTAL_MAX}]"
        )

    def test_security_stratum_count(self, real_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        count = sum(
            1 for r in adapter.iter_incidents() if r.corpus_stratum == "security"
        )
        assert self.EXPECTED_SECURITY_MIN <= count <= self.EXPECTED_SECURITY_MAX

    def test_ai_harm_stratum_count(self, real_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        count = sum(
            1 for r in adapter.iter_incidents() if r.corpus_stratum == "ai-harm"
        )
        assert self.EXPECTED_AI_HARM_MIN <= count <= self.EXPECTED_AI_HARM_MAX

    def test_every_stratum_has_a_bias_profile(
        self, real_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        profile_strata = {p.stratum for p in adapter.bias_profiles()}
        record_strata = {r.corpus_stratum for r in adapter.iter_incidents()}
        assert record_strata.issubset(profile_strata)

    def test_snapshot_hash_is_stable(self, real_snapshot: Path) -> None:
        """Content hash must be byte-stable across platforms."""
        from engine.snapshot.hashing import snapshot_hash

        h1 = snapshot_hash(real_snapshot / "incidents.json")
        h2 = snapshot_hash(real_snapshot / "incidents.json")
        assert h1 == h2
        assert h1 == real_snapshot.name

    def test_provenance_has_all_six_fields(self, real_snapshot: Path) -> None:
        from engine.snapshot.provenance import SnapshotProvenance

        prov = SnapshotProvenance.read(real_snapshot / "provenance.json")
        assert prov.source_repo != ""
        assert prov.source_commit_sha != ""
        assert prov.pull_date != ""
        assert prov.adapter_name == "genai_agentic"
        assert prov.adapter_version != ""
        assert prov.snapshot_hash == real_snapshot.name

    def test_bare_llm03_contamination_count(
        self, real_snapshot: Path
    ) -> None:
        """HANDOFF §3 F2: ~907 bare-LLM03 entries expected."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        contaminated = sum(
            1
            for r in adapter.iter_incidents()
            if is_bare_llm03_contaminated(list(r.native_labels))
        )
        assert 700 <= contaminated <= 1200, (
            f"Bare-LLM03 count {contaminated} outside expected range [700, 1200]"
        )
