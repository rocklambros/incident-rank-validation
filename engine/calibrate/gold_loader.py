# engine/calibrate/gold_loader.py
"""Gold calibration loader — manual curation + precision verification (spec A5)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from engine.calibrate.gold_schema import (
    GoldCalibration,
    GoldPrecisionLabel,
    GoldRecallLabel,
)

_PREFIX_PATTERN = re.compile(
    r"^MANUAL-((?:ROLL-|NEW-)?[A-Z][A-Z0-9]*)-(\d+)$"
)

_SHORT_PREFIX_TO_ENTRY_ID: dict[str, str] = {
    "MTIE": "NEW-MTIE",
    "ITSCD": "NEW-ITSCD",
    "CMSB": "ROLL-CMSB",
    "LAPTF": "ROLL-LAPTF",
    "CFAS": "ROLL-CFAS",
}


def parse_entry_id_from_prefix(incident_id: str) -> str:
    m = _PREFIX_PATTERN.match(incident_id)
    if not m:
        raise ValueError(
            f"Cannot parse entry ID from incident ID '{incident_id}'. "
            f"Expected format: MANUAL-{{ENTRY_ID}}-{{NNN}}"
        )
    raw = m.group(1)
    return _SHORT_PREFIX_TO_ENTRY_ID.get(raw, raw)


def _load_recall_from_curation(
    path: Path,
    valid_entry_ids: set[str],
    base_incident_ids: set[str] | None,
) -> list[GoldRecallLabel]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels: list[GoldRecallLabel] = []

    for record in data:
        incident_id = record["id"]

        if base_incident_ids and incident_id in base_incident_ids:
            continue

        native = record.get("native_labels", [])
        if native:
            entry_ids = list(native)
        else:
            entry_id = parse_entry_id_from_prefix(incident_id)
            entry_ids = [entry_id]

        for eid in entry_ids:
            if eid not in valid_entry_ids:
                raise ValueError(
                    f"Entry ID '{eid}' from incident '{incident_id}' "
                    f"not in rubric. Valid: {sorted(valid_entry_ids)}"
                )

        labels.append(GoldRecallLabel(
            incident_id=incident_id,
            true_entry_ids=entry_ids,
            classifier_entry_id=None,
            source="manual-curated",
        ))

    return labels


def _load_recall_from_adjudicated(
    path: Path,
    valid_entry_ids: set[str],
) -> tuple[list[GoldRecallLabel], list[GoldPrecisionLabel]]:
    recall: list[GoldRecallLabel] = []
    precision: list[GoldPrecisionLabel] = []

    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        labels = record.get("labels", [])
        consensus = record.get("llm_consensus")
        adjudicated = record.get("adjudicated", "")

        if adjudicated == "uncertain":
            continue

        for eid in labels:
            if eid not in valid_entry_ids:
                raise ValueError(
                    f"Entry ID '{eid}' from adjudicated incident "
                    f"'{record['incident_id']}' not in rubric."
                )

        recall.append(GoldRecallLabel(
            incident_id=record["incident_id"],
            true_entry_ids=labels if labels else [],
            classifier_entry_id=consensus,
            source="llm-adjudicated",
        ))

        if consensus and labels:
            precision.append(GoldPrecisionLabel(
                incident_id=record["incident_id"],
                claimed_entry_id=consensus,
                is_correct=(consensus in labels),
                source="llm-adjudicated",
            ))

    return recall, precision


def _load_precision_from_jsonl(path: Path) -> list[GoldPrecisionLabel]:
    labels: list[GoldPrecisionLabel] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        labels.append(GoldPrecisionLabel(
            incident_id=record["incident_id"],
            claimed_entry_id=record["claimed_entry_id"],
            is_correct=bool(record["is_correct"]),
            source=record.get("source", "stage2-verified"),
        ))
    return labels


def load_gold_calibration(
    *,
    curation_path: Path | None = None,
    precision_path: Path | None = None,
    gold_dir: Path | None = None,
    valid_entry_ids: set[str],
    rubric_hash: str,
    adjudicator_id: str,
    base_incident_ids: set[str] | None = None,
    session_count: int = 1,
) -> GoldCalibration:
    recall_labels: list[GoldRecallLabel] = []
    precision_labels: list[GoldPrecisionLabel] = []

    adjudicated_path: Path | None = None

    if gold_dir is not None:
        curation_candidate = gold_dir / "manual_curated_incidents.json"
        if curation_candidate.exists():
            curation_path = curation_candidate
        precision_candidate = gold_dir / "precision_verification.jsonl"
        if precision_candidate.exists():
            precision_path = precision_candidate
        adjudicated_candidate = gold_dir / "adjudicated_goldset.jsonl"
        if adjudicated_candidate.exists():
            adjudicated_path = adjudicated_candidate

    if curation_path is not None:
        recall_labels = _load_recall_from_curation(
            curation_path, valid_entry_ids, base_incident_ids,
        )

    if adjudicated_path is not None:
        adj_recall, adj_precision = _load_recall_from_adjudicated(
            adjudicated_path, valid_entry_ids,
        )
        recall_labels.extend(adj_recall)
        precision_labels.extend(adj_precision)

    if precision_path is not None:
        precision_labels.extend(_load_precision_from_jsonl(precision_path))

    hash_inputs: list[str] = []
    if curation_path is not None:
        hash_inputs.append(curation_path.read_text(encoding="utf-8"))
    if adjudicated_path is not None:
        hash_inputs.append(adjudicated_path.read_text(encoding="utf-8"))
    if precision_path is not None:
        hash_inputs.append(precision_path.read_text(encoding="utf-8"))
    provenance_hash = hashlib.sha256("".join(hash_inputs).encode()).hexdigest()

    return GoldCalibration(
        recall_labels=recall_labels,
        precision_labels=precision_labels,
        provenance_hash=provenance_hash,
        rubric_hash=rubric_hash,
        adjudicator_id=adjudicator_id,
        session_count=session_count,
    )
