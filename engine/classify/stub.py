"""Stub Stage-1 classifier for the synthetic validation cycle.

In a real cycle Stage-1 uses deterministic keyword/indicator rules from the
frozen rubric, and Stage-2 uses LLM-assisted classification.  For the synthetic
cycle the "classifier" reads ground-truth labels directly from each
``IncidentRecord.native_labels`` field — no heuristics, no model.

The overlap weights produced by the adapter flow into the Bayesian model
(Task 10); this module only bridges the classify phase for synthetic testing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from engine.schema import IncidentRecord

__all__ = [
    "Classification",
    "ClassificationResult",
    "classify_stub",
]


@dataclass(frozen=True, slots=True)
class Classification:
    """Result of classifying a single incident."""

    incident_id: str
    entry_id: str
    confidence: float  # 0.0 to 1.0
    stage: int  # 1 = deterministic, 2 = model-assisted
    rationale: str


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Aggregated classification output."""

    classifications: tuple[Classification, ...]
    classifier_version: str
    classifier_rule_hash: str  # hash of the rules used

    def counts_by_entry_stratum(
        self,
        incidents: dict[str, IncidentRecord],
    ) -> dict[tuple[str, str], int]:
        """Return ``{(entry_id, stratum): count}`` from classifications.

        Parameters
        ----------
        incidents:
            Mapping of incident id → ``IncidentRecord``.  Records not found in
            this mapping are silently skipped.
        """
        counts: dict[tuple[str, str], int] = {}
        for c in self.classifications:
            inc = incidents.get(c.incident_id)
            if inc is None:
                continue
            key = (c.entry_id, inc.corpus_stratum)
            counts[key] = counts.get(key, 0) + 1
        return counts


def classify_stub(
    incidents: tuple[IncidentRecord, ...],
    entry_ids: tuple[str, ...],
) -> ClassificationResult:
    """Stub classifier for synthetic data.

    Uses ``native_labels`` from each incident as ground truth.  Only assigns
    labels that match known *entry_ids*; any unknown labels are silently
    dropped.  Confidence is always 1.0 because these are synthetic ground-truth
    labels.

    Parameters
    ----------
    incidents:
        Tuple of ``IncidentRecord`` objects to classify.
    entry_ids:
        Tuple of entry identifiers recognised by the current rubric.  Labels
        outside this set are ignored.

    Returns
    -------
    ClassificationResult
        Contains one ``Classification`` per (incident, matching label) pair.
    """
    entry_set = set(entry_ids)
    classifications: list[Classification] = []
    for inc in incidents:
        for label in inc.native_labels:
            if label in entry_set:
                classifications.append(
                    Classification(
                        incident_id=inc.id,
                        entry_id=label,
                        confidence=1.0,
                        stage=1,
                        rationale="synthetic ground truth",
                    )
                )

    rule_hash = hashlib.sha256(b"stub-classifier-v0.1.0").hexdigest()
    return ClassificationResult(
        classifications=tuple(classifications),
        classifier_version="stub-0.1.0",
        classifier_rule_hash=rule_hash,
    )
