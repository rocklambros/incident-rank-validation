"""Bake-off input loaders: floor predictions, goldset incidents (text-only), corpus counts.

All three loaders are leakage-safe by construction:
- ``load_goldset_incidents`` returns IncidentRecord values that carry only
  ``id`` and ``text``; adjudicated labels, confidence, and consensus from the
  goldset file are never included (R7 firewall).
- ``load_floor_predictions`` and ``compute_corpus_class_counts`` read Stage-1
  labels only; they never see the locked goldset.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.schema import IncidentRecord


def load_floor_predictions(labeled_incidents_path: Path) -> dict[str, str]:
    """Return Stage-1 status-quo entry per incident_id.

    Parameters
    ----------
    labeled_incidents_path:
        Path to ``labeled_incidents.json`` — a JSON array of objects each
        containing at minimum ``incident_id`` and ``entry_id`` fields.

    Returns
    -------
    dict[str, str]
        ``{incident_id: entry_id}`` mapping representing the status-quo
        (Stage-1) classification for every labelled incident.
    """
    records: list[dict[str, object]] = json.loads(labeled_incidents_path.read_text())
    return {str(r["incident_id"]): str(r["entry_id"]) for r in records}


def load_goldset_incidents(
    goldset_path: Path,
    snapshot_path: Path,
) -> dict[str, IncidentRecord]:
    """Load goldset incident ids and hydrate text from the snapshot.

    R7 leakage firewall: returns ``IncidentRecord`` objects carrying ONLY
    ``id`` (== ``incident_id``) and ``text``.  Adjudicated labels, confidence,
    consensus, and any other goldset fields are deliberately ignored here and
    will never reach the ``predict_fn``.

    Parameters
    ----------
    goldset_path:
        Path to ``adjudicated_goldset.jsonl``.  Each non-blank line must be a
        JSON object with an ``incident_id`` key.  All other fields (labels,
        confidence, consensus) are intentionally skipped.
    snapshot_path:
        Path to the ``incidents.json`` snapshot file.  Accepts either the
        ``{"incidents": [...]}`` dict wrapper or a bare JSON array.

    Returns
    -------
    dict[str, IncidentRecord]
        ``{incident_id: IncidentRecord}`` where every record has ``.id`` and
        ``.text`` populated and all other fields carry safe empty/None
        defaults.  ``set(returned.keys()) == set(goldset_ids)``.
    """
    # Load goldset ids — label fields deliberately not read
    goldset_ids: list[str] = []
    for line in goldset_path.read_text().splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            goldset_ids.append(str(rec["incident_id"]))
    goldset_id_set = set(goldset_ids)

    # Load snapshot
    raw_data = json.loads(snapshot_path.read_text())
    if isinstance(raw_data, dict) and "incidents" in raw_data:
        incidents_list: list[dict[str, object]] = raw_data["incidents"]
    elif isinstance(raw_data, list):
        incidents_list = raw_data
    else:
        raise TypeError(
            f"Unexpected snapshot format in {snapshot_path}: "
            f"expected a JSON array or dict with an 'incidents' key, "
            f"got {type(raw_data).__name__}"
        )

    # Build a text-only index from the snapshot (no labels)
    snapshot_text: dict[str, str] = {}
    for raw_inc in incidents_list:
        inc_id = str(raw_inc.get("id", ""))
        if not inc_id or inc_id not in goldset_id_set:
            continue
        title = str(raw_inc.get("title", ""))
        description = str(raw_inc.get("description", ""))
        impact = str(raw_inc.get("impact", ""))
        text = " ".join(p for p in [title, description, impact] if p)
        snapshot_text[inc_id] = text

    # Assemble one IncidentRecord per goldset id (text-only; no label fields)
    result: dict[str, IncidentRecord] = {}
    for inc_id in goldset_ids:
        result[inc_id] = IncidentRecord(
            id=inc_id,
            date="",
            text=snapshot_text.get(inc_id, ""),
            severity=None,
            source_class="",
            corpus_stratum="",
            quality="auto",
            native_labels=(),
            source_url="",
        )
    return result


def compute_corpus_class_counts(labeled_incidents_path: Path) -> dict[str, int]:
    """Return the class distribution from ``labeled_incidents.json``.

    Used by ``run_bakeoff`` for the goldset/corpus TV-divergence audit.

    Parameters
    ----------
    labeled_incidents_path:
        Path to ``labeled_incidents.json`` (same file used by
        ``load_floor_predictions``).

    Returns
    -------
    dict[str, int]
        ``{entry_id: incident_count}`` over the full Stage-1 labeled corpus.
    """
    records: list[dict[str, object]] = json.loads(labeled_incidents_path.read_text())
    counts: dict[str, int] = {}
    for r in records:
        entry_id = str(r["entry_id"])
        counts[entry_id] = counts.get(entry_id, 0) + 1
    return counts
