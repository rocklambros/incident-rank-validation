"""Tests for engine.calibrate.two_frame_sampler — two-frame gold-set sampling."""
from __future__ import annotations

import pytest

from engine.calibrate.sampler import SampleFrame, SampleRequest, SampleResult, Sampler
from engine.calibrate.two_frame_sampler import TwoFrameSampler
from engine.classify.stub import Classification, ClassificationResult
from engine.schema import IncidentRecord


def _make_incidents(n: int, stratum: str = "security") -> list[IncidentRecord]:
    return [
        IncidentRecord(
            id=f"GA-{i:05d}",
            date="2026-01-01",
            text=f"incident text {i}",
            severity="High",
            source_class="advisory",
            corpus_stratum=stratum,
            quality="curated",
            native_labels=("LLM01",) if i % 3 == 0 else (),
            source_url="https://example.com",
        )
        for i in range(n)
    ]


def _make_classifications(
    incidents: list[IncidentRecord],
    entry_id: str = "LLM01",
) -> ClassificationResult:
    classifications = tuple(
        Classification(
            incident_id=inc.id,
            entry_id=entry_id,
            confidence=1.0,
            stage=1,
            rationale="test",
        )
        for inc in incidents
        if "LLM01" in inc.native_labels
    )
    return ClassificationResult(
        classifications=classifications,
        classifier_version="test",
        classifier_rule_hash="test_hash",
    )


class TestTwoFrameSampler:
    def test_implements_sampler_protocol(self) -> None:
        sampler = TwoFrameSampler(
            classification_result=_make_classifications([]),
        )
        assert isinstance(sampler, Sampler)

    def test_precision_frame_samples_classifier_positives(self) -> None:
        incidents = _make_incidents(100)
        classifications = _make_classifications(incidents, "LLM01")
        sampler = TwoFrameSampler(classification_result=classifications)
        request = SampleRequest(
            frame=SampleFrame.PRECISION,
            entry_id="LLM01",
            stratum="security",
            n=10,
        )
        result = sampler.draw(request, incidents, seed=42)
        assert result.actual_n <= 10
        assert all(inc.corpus_stratum == "security" for inc in result.incidents)
        classified_ids = {
            c.incident_id for c in classifications.classifications
            if c.entry_id == "LLM01"
        }
        assert all(inc.id in classified_ids for inc in result.incidents)

    def test_recall_frame_samples_all_incidents(self) -> None:
        incidents = _make_incidents(200)
        classifications = _make_classifications(incidents)
        sampler = TwoFrameSampler(classification_result=classifications)
        request = SampleRequest(
            frame=SampleFrame.RECALL,
            entry_id=None,
            stratum="security",
            n=50,
        )
        result = sampler.draw(request, incidents, seed=42)
        assert result.actual_n == 50
        assert all(inc.corpus_stratum == "security" for inc in result.incidents)

    def test_precision_frame_census_when_under_threshold(self) -> None:
        incidents = _make_incidents(30)
        classifications = _make_classifications(incidents, "LLM01")
        sampler = TwoFrameSampler(classification_result=classifications)
        classifier_positive_count = len([
            c for c in classifications.classifications if c.entry_id == "LLM01"
        ])
        request = SampleRequest(
            frame=SampleFrame.PRECISION,
            entry_id="LLM01",
            stratum="security",
            n=40,
        )
        result = sampler.draw(request, incidents, seed=42)
        if classifier_positive_count < 20:
            assert result.actual_n == classifier_positive_count

    def test_sample_hash_is_deterministic(self) -> None:
        incidents = _make_incidents(100)
        classifications = _make_classifications(incidents)
        sampler = TwoFrameSampler(classification_result=classifications)
        request = SampleRequest(
            frame=SampleFrame.RECALL,
            entry_id=None,
            stratum="security",
            n=20,
        )
        r1 = sampler.draw(request, incidents, seed=42)
        r2 = sampler.draw(request, incidents, seed=42)
        assert r1.sample_hash == r2.sample_hash
        assert tuple(i.id for i in r1.incidents) == tuple(i.id for i in r2.incidents)

    def test_precision_requires_entry_id(self) -> None:
        sampler = TwoFrameSampler(classification_result=_make_classifications([]))
        request = SampleRequest(
            frame=SampleFrame.PRECISION,
            entry_id=None,
            stratum="security",
            n=10,
        )
        with pytest.raises(ValueError, match="entry_id required"):
            sampler.draw(request, [], seed=42)
