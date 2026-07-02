"""Gold calibration schema for Two-Frame Gold Calibration (spec A4)."""
from __future__ import annotations

from dataclasses import dataclass

OUT_OF_SCOPE = "out-of-scope"
"""Sentinel classifier prediction: the classifier assigned no in-scope entry.

A recall MISS for each true entry, and NOT a precision false-positive (no
positive claim was made).  Matches the bake-off harness's OOS class string.
"""


@dataclass(frozen=True, slots=True)
class GoldRecallLabel:
    incident_id: str
    true_entry_ids: list[str]
    classifier_entry_id: str | None
    source: str


@dataclass(frozen=True, slots=True)
class GoldPrecisionLabel:
    incident_id: str
    claimed_entry_id: str
    is_correct: bool
    source: str


@dataclass(frozen=True, slots=True)
class GoldCalibration:
    recall_labels: list[GoldRecallLabel]
    precision_labels: list[GoldPrecisionLabel]
    provenance_hash: str
    rubric_hash: str
    adjudicator_id: str
    session_count: int
