"""Unit tests for corpus B corroboration: overlap detection and agreement."""
from __future__ import annotations

import pytest

from engine.decide.corpus_b_corroboration import (
    CorpusBCorroboration,
    IncidentOverlap,
    OverlapMethod,
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
