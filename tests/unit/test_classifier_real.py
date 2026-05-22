"""Tests for engine.classify.classifier — real Stage-1 keyword/indicator classifier."""
from __future__ import annotations

import hashlib

import pytest

from engine.classify.classifier import (
    ClassifierRules,
    EntryClassifierRule,
    classify_real,
)
from engine.schema import IncidentRecord


def _make_incident(
    id: str = "GA-001",
    text: str = "prompt injection attack on LLM",
    labels: tuple[str, ...] = (),
) -> IncidentRecord:
    return IncidentRecord(
        id=id,
        date="2026-01-01",
        text=text,
        severity="High",
        source_class="advisory",
        corpus_stratum="security",
        quality="curated",
        native_labels=labels,
        source_url="https://example.com",
    )


class TestEntryClassifierRule:
    def test_frozen(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=("not a prompt",),
            confidence_threshold=0.3,
        )
        with pytest.raises(AttributeError):
            rule.entry_id = "X"  # type: ignore[misc]


class TestClassifierRules:
    def test_rule_hash_deterministic(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=("benign",),
            confidence_threshold=0.3,
        )
        h1 = ClassifierRules.compute_rule_hash({"LLM01": rule})
        h2 = ClassifierRules.compute_rule_hash({"LLM01": rule})
        assert h1 == h2

    def test_rule_hash_changes_with_threshold(self) -> None:
        rule_a = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        rule_b = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.5,
        )
        h_a = ClassifierRules.compute_rule_hash({"LLM01": rule_a})
        h_b = ClassifierRules.compute_rule_hash({"LLM01": rule_b})
        assert h_a != h_b


class TestClassifyReal:
    def test_positive_match_case_insensitive(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash=ClassifierRules.compute_rule_hash({"LLM01": rule}),
        )
        incident = _make_incident(text="A PROMPT INJECTION was discovered")
        result = classify_real((incident,), rules)
        assert len(result.classifications) == 1
        assert result.classifications[0].entry_id == "LLM01"
        assert result.classifications[0].confidence >= 0.3

    def test_negative_pattern_suppresses(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("injection",),
            negative_patterns=("sql injection",),
            confidence_threshold=0.3,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash=ClassifierRules.compute_rule_hash({"LLM01": rule}),
        )
        incident = _make_incident(text="sql injection vulnerability found")
        result = classify_real((incident,), rules)
        assert len(result.classifications) == 0

    def test_below_threshold_excluded(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection", "jailbreak", "adversarial input"),
            negative_patterns=(),
            confidence_threshold=0.5,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash=ClassifierRules.compute_rule_hash({"LLM01": rule}),
        )
        incident = _make_incident(text="A prompt injection was found")
        result = classify_real((incident,), rules)
        assert len(result.classifications) == 0

    def test_multi_entry_classification(self) -> None:
        rule_01 = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        rule_02 = EntryClassifierRule(
            entry_id="LLM02",
            positive_patterns=("data leak",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule_01, "LLM02": rule_02},
            rule_hash=ClassifierRules.compute_rule_hash(
                {"LLM01": rule_01, "LLM02": rule_02}
            ),
        )
        incident = _make_incident(text="prompt injection caused a data leak")
        result = classify_real((incident,), rules)
        entry_ids = {c.entry_id for c in result.classifications}
        assert entry_ids == {"LLM01", "LLM02"}

    def test_result_uses_real_rule_hash(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        expected_hash = ClassifierRules.compute_rule_hash({"LLM01": rule})
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash=expected_hash,
        )
        incident = _make_incident(text="prompt injection attack")
        result = classify_real((incident,), rules)
        assert result.classifier_rule_hash == expected_hash
        stub_hash = hashlib.sha256(b"stub-classifier-v0.1.0").hexdigest()
        assert result.classifier_rule_hash != stub_hash
