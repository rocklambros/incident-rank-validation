"""Unit tests for engine.classify.stub."""

from __future__ import annotations

import hashlib

import pytest

from engine.classify.stub import Classification, classify_stub
from engine.schema import IncidentRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_incident(
    inc_id: str,
    native_labels: tuple[str, ...],
    corpus_stratum: str = "stratum_a",
) -> IncidentRecord:
    return IncidentRecord(
        id=inc_id,
        date="2024-01-01",
        text="test incident",
        severity="High",
        source_class="advisory",
        corpus_stratum=corpus_stratum,
        quality="curated",
        native_labels=native_labels,
        source_url="https://example.com",
    )


# ---------------------------------------------------------------------------
# Classification dataclass
# ---------------------------------------------------------------------------


class TestClassificationDataclass:
    def test_fields(self) -> None:
        c = Classification(
            incident_id="INC-1",
            entry_id="E01",
            confidence=1.0,
            stage=1,
            rationale="synthetic ground truth",
        )
        assert c.incident_id == "INC-1"
        assert c.entry_id == "E01"
        assert c.confidence == 1.0
        assert c.stage == 1
        assert c.rationale == "synthetic ground truth"

    def test_frozen(self) -> None:
        c = Classification(
            incident_id="INC-1",
            entry_id="E01",
            confidence=1.0,
            stage=1,
            rationale="x",
        )
        with pytest.raises((AttributeError, TypeError)):
            c.entry_id = "E02"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# classify_stub — basic behaviour
# ---------------------------------------------------------------------------


class TestClassifyStubBasic:
    def test_empty_incidents_produces_empty_classifications(self) -> None:
        result = classify_stub((), ("E01", "E02"))
        assert result.classifications == ()

    def test_produces_classification_objects(self) -> None:
        inc = make_incident("INC-1", ("E01",))
        result = classify_stub((inc,), ("E01",))
        assert len(result.classifications) == 1
        assert isinstance(result.classifications[0], Classification)

    def test_correct_incident_id_and_entry_id(self) -> None:
        inc = make_incident("INC-42", ("E03",))
        result = classify_stub((inc,), ("E03",))
        c = result.classifications[0]
        assert c.incident_id == "INC-42"
        assert c.entry_id == "E03"

    def test_confidence_is_always_one(self) -> None:
        inc = make_incident("INC-1", ("E01", "E02"))
        result = classify_stub((inc,), ("E01", "E02"))
        for c in result.classifications:
            assert c.confidence == 1.0

    def test_stage_is_one(self) -> None:
        inc = make_incident("INC-1", ("E01",))
        result = classify_stub((inc,), ("E01",))
        assert result.classifications[0].stage == 1

    def test_rationale_is_synthetic_ground_truth(self) -> None:
        inc = make_incident("INC-1", ("E01",))
        result = classify_stub((inc,), ("E01",))
        assert result.classifications[0].rationale == "synthetic ground truth"

    def test_multiple_labels_per_incident(self) -> None:
        inc = make_incident("INC-1", ("E01", "E02", "E03"))
        result = classify_stub((inc,), ("E01", "E02", "E03"))
        entry_ids = {c.entry_id for c in result.classifications}
        assert entry_ids == {"E01", "E02", "E03"}
        assert all(c.incident_id == "INC-1" for c in result.classifications)

    def test_multiple_incidents(self) -> None:
        incidents = (
            make_incident("INC-1", ("E01",)),
            make_incident("INC-2", ("E02",)),
        )
        result = classify_stub(incidents, ("E01", "E02"))
        assert len(result.classifications) == 2
        inc_ids = {c.incident_id for c in result.classifications}
        assert inc_ids == {"INC-1", "INC-2"}


# ---------------------------------------------------------------------------
# classify_stub — unknown label filtering
# ---------------------------------------------------------------------------


class TestClassifyStubLabelFiltering:
    def test_unknown_labels_are_ignored(self) -> None:
        inc = make_incident("INC-1", ("E01", "UNKNOWN_LABEL", "E02"))
        result = classify_stub((inc,), ("E01", "E02"))
        entry_ids = {c.entry_id for c in result.classifications}
        assert "UNKNOWN_LABEL" not in entry_ids
        assert entry_ids == {"E01", "E02"}

    def test_all_unknown_labels_produces_no_classifications(self) -> None:
        inc = make_incident("INC-1", ("BOGUS_A", "BOGUS_B"))
        result = classify_stub((inc,), ("E01", "E02"))
        assert result.classifications == ()

    def test_empty_entry_ids_produces_no_classifications(self) -> None:
        inc = make_incident("INC-1", ("E01",))
        result = classify_stub((inc,), ())
        assert result.classifications == ()

    def test_no_native_labels_produces_no_classifications(self) -> None:
        inc = make_incident("INC-1", ())
        result = classify_stub((inc,), ("E01",))
        assert result.classifications == ()


# ---------------------------------------------------------------------------
# ClassificationResult metadata
# ---------------------------------------------------------------------------


class TestClassificationResultMetadata:
    def test_classifier_version(self) -> None:
        result = classify_stub((), ("E01",))
        assert result.classifier_version == "stub-0.1.0"

    def test_rule_hash_is_deterministic(self) -> None:
        result_a = classify_stub((), ("E01",))
        result_b = classify_stub((), ("E01",))
        assert result_a.classifier_rule_hash == result_b.classifier_rule_hash

    def test_rule_hash_value(self) -> None:
        expected = hashlib.sha256(b"stub-classifier-v0.1.0").hexdigest()
        result = classify_stub((), ("E01",))
        assert result.classifier_rule_hash == expected

    def test_rule_hash_independent_of_input(self) -> None:
        inc = make_incident("INC-1", ("E01",))
        result_empty = classify_stub((), ("E01",))
        result_with_data = classify_stub((inc,), ("E01",))
        assert result_empty.classifier_rule_hash == result_with_data.classifier_rule_hash


# ---------------------------------------------------------------------------
# counts_by_entry_stratum
# ---------------------------------------------------------------------------


class TestCountsByEntryStratum:
    def test_single_incident_single_label(self) -> None:
        inc = make_incident("INC-1", ("E01",), corpus_stratum="stratum_a")
        result = classify_stub((inc,), ("E01",))
        counts = result.counts_by_entry_stratum({"INC-1": inc})
        assert counts == {("E01", "stratum_a"): 1}

    def test_multiple_incidents_same_stratum(self) -> None:
        inc1 = make_incident("INC-1", ("E01",), corpus_stratum="stratum_a")
        inc2 = make_incident("INC-2", ("E01",), corpus_stratum="stratum_a")
        result = classify_stub((inc1, inc2), ("E01",))
        counts = result.counts_by_entry_stratum({"INC-1": inc1, "INC-2": inc2})
        assert counts == {("E01", "stratum_a"): 2}

    def test_multiple_strata(self) -> None:
        inc1 = make_incident("INC-1", ("E01",), corpus_stratum="stratum_a")
        inc2 = make_incident("INC-2", ("E01",), corpus_stratum="stratum_b")
        result = classify_stub((inc1, inc2), ("E01",))
        counts = result.counts_by_entry_stratum({"INC-1": inc1, "INC-2": inc2})
        assert counts == {("E01", "stratum_a"): 1, ("E01", "stratum_b"): 1}

    def test_multiple_entry_ids_multiple_strata(self) -> None:
        inc1 = make_incident("INC-1", ("E01", "E02"), corpus_stratum="stratum_a")
        inc2 = make_incident("INC-2", ("E01",), corpus_stratum="stratum_b")
        result = classify_stub((inc1, inc2), ("E01", "E02"))
        counts = result.counts_by_entry_stratum({"INC-1": inc1, "INC-2": inc2})
        assert counts[("E01", "stratum_a")] == 1
        assert counts[("E02", "stratum_a")] == 1
        assert counts[("E01", "stratum_b")] == 1

    def test_missing_incident_in_mapping_is_skipped(self) -> None:
        inc = make_incident("INC-1", ("E01",), corpus_stratum="stratum_a")
        result = classify_stub((inc,), ("E01",))
        # Pass empty dict — INC-1 is not found, should be skipped
        counts = result.counts_by_entry_stratum({})
        assert counts == {}

    def test_empty_classifications_returns_empty_dict(self) -> None:
        result = classify_stub((), ("E01",))
        counts = result.counts_by_entry_stratum({})
        assert counts == {}
