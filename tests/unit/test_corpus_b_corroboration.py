"""Unit tests for corpus B corroboration: overlap detection and agreement."""
from __future__ import annotations

import pytest

from engine.decide.corpus_b_corroboration import (
    CorpusBCorroboration,
    IncidentOverlap,
    OverlapMethod,
    compute_agreement,
    detect_overlaps,
)
from engine.schema import IncidentRecord


def _make_record(
    record_id: str,
    text: str = "test",
    source_url: str = "",
    native_labels: tuple[str, ...] = (),
) -> IncidentRecord:
    return IncidentRecord(
        id=record_id,
        date="2025-01-01",
        text=text,
        severity=None,
        source_class="advisory",
        corpus_stratum="security",
        quality="auto",
        native_labels=native_labels,
        source_url=source_url,
    )


class TestOverlapDetection:
    def test_url_match(self) -> None:
        corpus_a = [
            _make_record("INC-001", source_url="https://nvd.nist.gov/vuln/detail/CVE-2025-64110"),
        ]
        corpus_b = [
            _make_record("ASIB-001", source_url="https://nvd.nist.gov/vuln/detail/CVE-2025-64110"),
        ]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 1
        assert overlaps[0].corpus_a_id == "INC-001"
        assert overlaps[0].corpus_b_id == "ASIB-001"
        assert overlaps[0].method == OverlapMethod.URL

    def test_cve_match_from_url(self) -> None:
        corpus_a = [
            _make_record(
                "INC-002",
                text="Vulnerability CVE-2023-48022 exploited",
                source_url="https://example.com/some-page",
            ),
        ]
        corpus_b = [
            _make_record(
                "ASIB-002",
                source_url="https://nvd.nist.gov/vuln/detail/CVE-2023-48022",
            ),
        ]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 1
        assert overlaps[0].method == OverlapMethod.CVE

    def test_no_overlap_when_different(self) -> None:
        corpus_a = [_make_record("INC-001", text="Something unrelated")]
        corpus_b = [_make_record("ASIB-001", text="Completely different")]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 0

    def test_title_keyword_match(self) -> None:
        corpus_a = [
            _make_record(
                "INC-003",
                text="Claude Skills Ransomware Deployment demonstrated by Cato Networks",
            ),
        ]
        corpus_b = [
            _make_record(
                "ASIB-003",
                text="Claude Skills Ransomware Deployment via MedusaLocker",
            ),
        ]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 1
        assert overlaps[0].method == OverlapMethod.TITLE_KEYWORD

    def test_deduplicates_matches(self) -> None:
        url = "https://nvd.nist.gov/vuln/detail/CVE-2025-64110"
        corpus_a = [
            _make_record("INC-001", text="CVE-2025-64110 details", source_url=url),
        ]
        corpus_b = [
            _make_record("ASIB-001", text="CVE-2025-64110 info", source_url=url),
        ]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 1


class TestAgreementComputation:
    def test_full_agreement(self) -> None:
        overlaps = [
            IncidentOverlap("INC-001", "ASIB-001", OverlapMethod.URL, "url match"),
            IncidentOverlap("INC-002", "ASIB-002", OverlapMethod.CVE, "cve match"),
        ]
        a_labels = {"INC-001": "LLM01", "INC-002": "LLM05"}
        b_labels = {"ASIB-001": "LLM01", "ASIB-002": "LLM05"}
        b_records = {
            "ASIB-001": _make_record("ASIB-001", text="Incident Alpha"),
            "ASIB-002": _make_record("ASIB-002", text="Incident Beta"),
        }

        result = compute_agreement(
            overlaps, a_labels, b_labels, b_records,
            baseline_kappa=0.275, corpus_a_count=100, corpus_b_count=10,
        )
        assert result.agreement_count == 2
        assert result.disagreement_count == 0
        assert result.agreement_rate == 1.0

    def test_partial_agreement(self) -> None:
        overlaps = [
            IncidentOverlap("INC-001", "ASIB-001", OverlapMethod.URL, "match"),
            IncidentOverlap("INC-002", "ASIB-002", OverlapMethod.CVE, "match"),
        ]
        a_labels = {"INC-001": "LLM01", "INC-002": "LLM05"}
        b_labels = {"ASIB-001": "LLM01", "ASIB-002": "LLM03"}
        b_records = {
            "ASIB-001": _make_record("ASIB-001", text="Incident Alpha"),
            "ASIB-002": _make_record("ASIB-002", text="Incident Beta"),
        }

        result = compute_agreement(
            overlaps, a_labels, b_labels, b_records,
            baseline_kappa=0.275, corpus_a_count=100, corpus_b_count=10,
        )
        assert result.agreement_count == 1
        assert result.disagreement_count == 1
        assert result.agreement_rate == 0.5

    def test_systematic_divergence_detected(self) -> None:
        overlaps = [
            IncidentOverlap("INC-001", "ASIB-001", OverlapMethod.URL, "m"),
            IncidentOverlap("INC-002", "ASIB-002", OverlapMethod.URL, "m"),
            IncidentOverlap("INC-003", "ASIB-003", OverlapMethod.URL, "m"),
        ]
        a_labels = {"INC-001": "LLM05", "INC-002": "LLM05", "INC-003": "LLM01"}
        b_labels = {"ASIB-001": "LLM03", "ASIB-002": "LLM03", "ASIB-003": "LLM01"}
        b_records = {
            f"ASIB-{i:03d}": _make_record(f"ASIB-{i:03d}", text=f"Inc {i}")
            for i in range(1, 4)
        }

        result = compute_agreement(
            overlaps, a_labels, b_labels, b_records,
            baseline_kappa=0.275, corpus_a_count=100, corpus_b_count=10,
        )
        assert len(result.systematic_divergences) >= 1
        assert any(d.count >= 2 for d in result.systematic_divergences)

    def test_empty_overlap_produces_zero_rate(self) -> None:
        result = compute_agreement(
            [], {}, {}, {},
            baseline_kappa=0.275, corpus_a_count=100, corpus_b_count=10,
        )
        assert result.overlap_count == 0
        assert result.agreement_rate == 0.0

    def test_baseline_kappa_propagated(self) -> None:
        result = compute_agreement(
            [], {}, {}, {},
            baseline_kappa=0.275, corpus_a_count=6674, corpus_b_count=46,
        )
        assert result.baseline_kappa == 0.275
        assert result.corpus_a_incident_count == 6674
        assert result.corpus_b_incident_count == 46
