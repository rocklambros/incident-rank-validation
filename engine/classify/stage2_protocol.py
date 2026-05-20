"""Stage-2 LLM-assisted classifier protocol stub.

Stage-2 handles incidents that Stage-1 deterministic rules could not resolve.
It delegates to an external LLM (e.g. Llama-3.1-70B) and captures full
provenance: model identity, weight hash, and prompt hash so that any
classification can be reproduced or audited.

This is a Plan 5 deliverable.  The classes here define the interface only;
both callable methods raise ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.schema import IncidentRecord

__all__ = [
    "Stage2Classification",
    "Stage2Protocol",
]


@dataclass(frozen=True, slots=True)
class Stage2Classification:
    """Provenance-bearing result of a single Stage-2 LLM classification."""

    incident_id: str
    entry_id: str
    confidence: float
    rationale: str
    model_identity: str
    weight_provenance_hash: str  # SHA-256 of model weights
    prompt_hash: str  # SHA-256 of the prompt template used


class Stage2Protocol:
    """Interface for the Stage-2 LLM-assisted classifier."""

    def classify(
        self, incident: IncidentRecord, rubric_hash: str
    ) -> Stage2Classification:
        """Classify a single incident using LLM-assisted Stage-2.

        Plan 5 deliverable — this stub raises NotImplementedError.
        """
        raise NotImplementedError(
            "Stage-2 LLM classifier is a Plan 5 deliverable. "
            "This protocol stub defines the interface only."
        )

    def classify_batch(
        self,
        incidents: tuple[IncidentRecord, ...],
        rubric_hash: str,
    ) -> tuple[Stage2Classification, ...]:
        """Classify a batch of incidents.

        Plan 5 deliverable — this stub raises NotImplementedError.
        """
        raise NotImplementedError(
            "Stage-2 LLM classifier is a Plan 5 deliverable. "
            "This protocol stub defines the interface only."
        )
