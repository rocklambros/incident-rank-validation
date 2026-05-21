"""Real Stage-1 deterministic keyword/indicator classifier.

Built from the frozen rubric's positive_indicators and negative_indicators.
Matching semantics: case-insensitive substring search. For each indicator
string P and incident text T, a match occurs when P.lower() is found in
T.lower(). No regex, no word-boundary constraints.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from engine.classify.stub import Classification, ClassificationResult
from engine.prereg.rubric import Rubric
from engine.schema import IncidentRecord


@dataclass(frozen=True, slots=True)
class EntryClassifierRule:
    entry_id: str
    positive_patterns: tuple[str, ...]
    negative_patterns: tuple[str, ...]
    confidence_threshold: float


@dataclass(frozen=True, slots=True)
class ClassifierRules:
    rules_by_entry: dict[str, EntryClassifierRule]
    rule_hash: str

    @staticmethod
    def compute_rule_hash(
        rules_by_entry: dict[str, EntryClassifierRule],
    ) -> str:
        canonical = {}
        for eid in sorted(rules_by_entry):
            r = rules_by_entry[eid]
            canonical[eid] = {
                "confidence_threshold": r.confidence_threshold,
                "negative_patterns": list(r.negative_patterns),
                "positive_patterns": list(r.positive_patterns),
            }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_rules_from_rubric(
    rubric: Rubric,
    confidence_threshold: float = 0.3,
) -> ClassifierRules:
    rules: dict[str, EntryClassifierRule] = {}
    for entry in rubric.entries:
        rules[entry.entry_id] = EntryClassifierRule(
            entry_id=entry.entry_id,
            positive_patterns=entry.positive_indicators,
            negative_patterns=entry.negative_indicators,
            confidence_threshold=confidence_threshold,
        )
    return ClassifierRules(
        rules_by_entry=rules,
        rule_hash=ClassifierRules.compute_rule_hash(rules),
    )


def _compute_confidence(
    text_lower: str,
    rule: EntryClassifierRule,
) -> float:
    if not rule.positive_patterns:
        return 0.0
    positive_hits = sum(
        1 for p in rule.positive_patterns if p.lower() in text_lower
    )
    negative_hits = sum(
        1 for n in rule.negative_patterns if n.lower() in text_lower
    )
    return max(0, positive_hits - negative_hits) / len(rule.positive_patterns)


def classify_real(
    incidents: tuple[IncidentRecord, ...],
    rules: ClassifierRules,
) -> ClassificationResult:
    classifications: list[Classification] = []
    for inc in incidents:
        text_lower = inc.text.lower()
        for rule in rules.rules_by_entry.values():
            confidence = _compute_confidence(text_lower, rule)
            if confidence >= rule.confidence_threshold:
                classifications.append(
                    Classification(
                        incident_id=inc.id,
                        entry_id=rule.entry_id,
                        confidence=confidence,
                        stage=1,
                        rationale=f"indicator match: confidence={confidence:.3f}",
                    )
                )
    return ClassificationResult(
        classifications=tuple(classifications),
        classifier_version="stage1-keyword-1.0.0",
        classifier_rule_hash=rules.rule_hash,
    )
