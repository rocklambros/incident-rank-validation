"""JSON I/O for rubric, attestation, and adjudication log artifacts.

All writes produce human-readable, indented JSON.  Reads reconstruct
frozen dataclasses from the JSON.  Hashing uses the Rubric.compute_hash()
canonical form, not the pretty-printed file contents.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.prereg.adjudication import AdjudicationEntry, AdjudicationLog
from engine.prereg.rubric import BoundaryRule, Rubric, RubricEntry
from engine.prereg.rubric_attestation import RubricDraftingAttestation


def write_rubric(rubric: Rubric, path: Path) -> None:
    """Write *rubric* as human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rubric.to_dict()
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def _validate_pair(pair: list[str], entry_id: str) -> tuple[str, str]:
    """Validate and convert a co_occurrence_pair from JSON."""
    if len(pair) != 2:
        raise ValueError(
            f"{entry_id}: co_occurrence_pair must have exactly 2 elements, "
            f"got {len(pair)}: {pair}"
        )
    return (pair[0], pair[1])


def read_rubric(path: Path) -> Rubric:
    """Read a Rubric from JSON."""
    data = json.loads(path.read_text())
    entries = tuple(
        RubricEntry(
            entry_id=e["entry_id"],
            canonical_name=e["canonical_name"],
            in_scope=e["in_scope"],
            exclusions=tuple(e["exclusions"]),
            boundary_rules=tuple(
                BoundaryRule(
                    adjacent_entry_id=br["adjacent_entry_id"],
                    rule=br["rule"],
                    is_ambiguous=br["is_ambiguous"],
                )
                for br in e["boundary_rules"]
            ),
            positive_indicators=tuple(e["positive_indicators"]),
            negative_indicators=tuple(e["negative_indicators"]),
            co_occurrence_pairs=tuple(
                _validate_pair(pair, e["entry_id"])
                for pair in e["co_occurrence_pairs"]
            ),
            is_rollup_candidate=e["is_rollup_candidate"],
            rolled_into=e["rolled_into"],
        )
        for e in data["entries"]
    )
    return Rubric(
        cycle_id=data["cycle_id"],
        version=data["version"],
        entries=entries,
    )


def write_rubric_attestation(
    attestation: RubricDraftingAttestation, path: Path
) -> None:
    """Write drafting attestation as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "viewed_corpus_before_drafting": attestation.viewed_corpus_before_drafting,
        "viewed_corpus_details": attestation.viewed_corpus_details,
        "viewed_vote_data_before_drafting": attestation.viewed_vote_data_before_drafting,
        "viewed_vote_data_details": attestation.viewed_vote_data_details,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_rubric_attestation(path: Path) -> RubricDraftingAttestation:
    """Read a RubricDraftingAttestation from JSON."""
    data = json.loads(path.read_text())
    return RubricDraftingAttestation(
        viewed_corpus_before_drafting=data["viewed_corpus_before_drafting"],
        viewed_corpus_details=data["viewed_corpus_details"],
        viewed_vote_data_before_drafting=data["viewed_vote_data_before_drafting"],
        viewed_vote_data_details=data["viewed_vote_data_details"],
    )


def write_adjudication_log(log: AdjudicationLog, path: Path) -> None:
    """Write adjudication log as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "rubric_hash": log.rubric_hash,
        "entries": [
            {
                "entry_id_a": e.entry_id_a,
                "entry_id_b": e.entry_id_b,
                "decision": e.decision,
                "rationale": e.rationale,
                "adjudicator": e.adjudicator,
                "date": e.date,
            }
            for e in log.entries
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_adjudication_log(path: Path) -> AdjudicationLog:
    """Read an AdjudicationLog from JSON."""
    data = json.loads(path.read_text())
    entries = tuple(
        AdjudicationEntry(
            entry_id_a=e["entry_id_a"],
            entry_id_b=e["entry_id_b"],
            decision=e["decision"],
            rationale=e["rationale"],
            adjudicator=e["adjudicator"],
            date=e["date"],
        )
        for e in data["entries"]
    )
    return AdjudicationLog(
        rubric_hash=data["rubric_hash"],
        entries=entries,
    )
