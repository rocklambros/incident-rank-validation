"""Batch file generation, validation, and synthetic coding.

This module sits between the sampler and tally stages.  It:

  - Generates JSON batch files for human coders (``generate_batch``).
  - Validates returned coded batch files against expected provenance hashes
    and the canonical entry-ID taxonomy (``validate_coded_batch``).
  - Provides synthetic coding paths for automated testing
    (``code_synthetic``, ``code_synthetic_with_ground_truth``).

Batch file format
-----------------
A batch file is a JSON object with two top-level keys:

  ``header``
      Provenance metadata: cycle_id, batch_id, frame, entry_id, stratum,
      sample_hash, rubric_hash, manifest_lock_hash, coder_id, generated_at,
      and optionally coding_checklist.

  ``incidents``
      List of incident objects: incident_id, text, labels (null = uncoded,
      [] = no match, [...] = one or more entry IDs), rollup_sub_labels,
      notes, amendment.

See HANDOFF §6 for the batch-file contract.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.calibrate.sampler import SampleResult
from engine.schema import IncidentRecord

__all__ = [
    "BatchHeader",
    "BatchIncident",
    "CodingBatch",
    "ValidationError",
    "generate_batch",
    "validate_coded_batch",
    "code_synthetic",
    "code_synthetic_with_ground_truth",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BatchHeader:
    """Provenance metadata for a coding batch.

    Fields
    ------
    cycle_id
        Calibration cycle identifier, e.g. ``"2026"``.
    batch_id
        UUID assigned at batch generation time.
    frame
        Sampling frame: ``"precision"`` or ``"recall"``.
    entry_id
        Entry identifier for precision frames; ``None`` for recall frames.
    stratum
        Corpus stratum from which the sample was drawn.
    sample_hash
        Deterministic hash of the sample for provenance tracking.
    rubric_hash
        Hash of the coding rubric in effect at generation time.
    manifest_lock_hash
        Hash of the manifest lock file in effect at generation time.
    coder_id
        Identifier of the assigned coder (human or ``"synthetic"``).
    generated_at
        ISO 8601 UTC timestamp of batch generation.
    coding_checklist
        Optional mapping of entry_id -> name used for recall-frame coding.
    """

    cycle_id: str
    batch_id: str
    frame: str
    entry_id: str | None
    stratum: str
    sample_hash: str
    rubric_hash: str
    manifest_lock_hash: str
    coder_id: str
    generated_at: str = ""
    coding_checklist: dict[str, str] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "cycle_id": self.cycle_id,
            "batch_id": self.batch_id,
            "frame": self.frame,
            "entry_id": self.entry_id,
            "stratum": self.stratum,
            "sample_hash": self.sample_hash,
            "rubric_hash": self.rubric_hash,
            "manifest_lock_hash": self.manifest_lock_hash,
            "coder_id": self.coder_id,
            "generated_at": self.generated_at,
        }
        if self.coding_checklist is not None:
            d["coding_checklist"] = self.coding_checklist
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchHeader:
        return cls(
            cycle_id=data["cycle_id"],
            batch_id=data["batch_id"],
            frame=data["frame"],
            entry_id=data.get("entry_id"),
            stratum=data["stratum"],
            sample_hash=data["sample_hash"],
            rubric_hash=data["rubric_hash"],
            manifest_lock_hash=data["manifest_lock_hash"],
            coder_id=data["coder_id"],
            generated_at=data["generated_at"],
            coding_checklist=data.get("coding_checklist"),
        )


@dataclass
class BatchIncident:
    """A single incident within a coding batch.

    Fields
    ------
    incident_id
        The incident's canonical identifier.
    text
        The incident text presented to the coder.
    labels
        ``None`` means uncoded (blank for human to fill in).
        ``[]`` means the coder determined no entry matches.
        ``[...]`` means one or more matching entry IDs.
    rollup_sub_labels
        Optional sub-labels for rollup entries.
    notes
        Optional coder notes.
    amendment
        Optional amendment note if the coding was revised.
    _native_labels
        Internal field carrying original native_labels for synthetic coding.
        Not serialised to JSON.
    """

    incident_id: str
    text: str
    labels: list[str] | None = field(default=None)
    rollup_sub_labels: list[str] | None = field(default=None)
    notes: str | None = field(default=None)
    amendment: str | None = field(default=None)
    # Internal field — not serialised.  Populated by generate_batch so that
    # code_synthetic can fill labels without needing an external lookup.
    _native_labels: tuple[str, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "incident_id": self.incident_id,
            "text": self.text,
            "labels": self.labels,
        }
        if self.rollup_sub_labels is not None:
            d["rollup_sub_labels"] = self.rollup_sub_labels
        if self.notes is not None:
            d["notes"] = self.notes
        if self.amendment is not None:
            d["amendment"] = self.amendment
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchIncident:
        return cls(
            incident_id=data["incident_id"],
            text=data["text"],
            labels=data.get("labels"),
            rollup_sub_labels=data.get("rollup_sub_labels"),
            notes=data.get("notes"),
            amendment=data.get("amendment"),
        )


@dataclass
class CodingBatch:
    """A complete coding batch: header + incidents.

    Attributes
    ----------
    header
        Provenance metadata.
    incidents
        List of BatchIncident objects.
    """

    header: BatchHeader
    incidents: list[BatchIncident]

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "incidents": [inc.to_dict() for inc in self.incidents],
        }

    def write(self, path: Path, indent: int = 2) -> None:
        """Serialise the batch to JSON at *path*."""
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> CodingBatch:
        """Deserialise a CodingBatch from a JSON file at *path*."""
        data = json.loads(path.read_text(encoding="utf-8"))
        header = BatchHeader.from_dict(data["header"])
        incidents = [BatchIncident.from_dict(inc) for inc in data["incidents"]]
        return cls(header=header, incidents=incidents)


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationError:
    """A single validation problem found in a coded batch file.

    Fields
    ------
    file
        Path to the batch file that was validated.
    incident_id
        The incident_id where the problem was found, or ``None`` for header
        errors.
    message
        Human-readable description of the problem.
    """

    file: Path
    incident_id: str | None
    message: str

    def __str__(self) -> str:
        if self.incident_id:
            return f"{self.file}[{self.incident_id}]: {self.message}"
        return f"{self.file}: {self.message}"


# ---------------------------------------------------------------------------
# generate_batch
# ---------------------------------------------------------------------------


def generate_batch(
    *,
    sample_result: SampleResult,
    rubric_hash: str,
    manifest_lock_hash: str,
    coder_id: str,
    cycle_id: str,
    coding_checklist: dict[str, str] | None = None,
) -> CodingBatch:
    """Create a :class:`CodingBatch` from a :class:`SampleResult`.

    Parameters
    ----------
    sample_result:
        The sampling result whose incidents should be batched for coding.
    rubric_hash:
        Hash of the coding rubric in effect.
    manifest_lock_hash:
        Hash of the manifest lock file in effect.
    coder_id:
        Identifier for the assigned coder.
    cycle_id:
        Calibration cycle identifier (e.g. ``"2026"``).
    coding_checklist:
        Optional mapping of entry_id -> name for recall-frame coding.

    Returns
    -------
    CodingBatch
        A new batch with ``labels=None`` on every incident (uncoded).
    """
    req = sample_result.request
    header = BatchHeader(
        cycle_id=cycle_id,
        batch_id=str(uuid.uuid4()),
        frame=req.frame.value,
        entry_id=req.entry_id,
        stratum=req.stratum,
        sample_hash=sample_result.sample_hash,
        rubric_hash=rubric_hash,
        manifest_lock_hash=manifest_lock_hash,
        coder_id=coder_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        coding_checklist=coding_checklist,
    )
    incidents = [
        BatchIncident(
            incident_id=inc.id,
            text=inc.text,
            labels=None,  # uncoded — coder fills this in
            _native_labels=inc.native_labels,
        )
        for inc in sample_result.incidents
    ]
    return CodingBatch(header=header, incidents=incidents)


# ---------------------------------------------------------------------------
# validate_coded_batch
# ---------------------------------------------------------------------------


def validate_coded_batch(
    path: Path,
    *,
    valid_entry_ids: set[str],
    rollup_entry_ids: set[str],
    expected_sample_hash: str,
    expected_rubric_hash: str,
    expected_lock_hash: str,
    expected_incident_ids: set[str] | None = None,
) -> list[ValidationError]:
    """Validate a coded batch file against expected provenance and taxonomy.

    Parameters
    ----------
    path:
        Path to the JSON batch file to validate.
    valid_entry_ids:
        The complete set of canonical entry IDs (e.g. ``{"LLM01", ..., "LLM10"}``).
    rollup_entry_ids:
        Entry IDs that are rollup entries and require sub-labels.
    expected_sample_hash:
        The expected value of ``header.sample_hash``.
    expected_rubric_hash:
        The expected value of ``header.rubric_hash``.
    expected_lock_hash:
        The expected value of ``header.manifest_lock_hash``.
    expected_incident_ids:
        If provided, every incident_id in the batch must be a member of this
        set.  Detects ID corruption or injection of fabricated incidents.

    Returns
    -------
    list[ValidationError]
        Empty list when the batch is fully valid.  Non-empty when there are
        problems.  Uncoded incidents (``labels=None``) produce warnings
        rather than hard errors but are still returned as ValidationError
        objects so the caller can decide how to treat them.
    """
    errors: list[ValidationError] = []

    data = json.loads(path.read_text(encoding="utf-8"))
    header_data = data.get("header", {})

    # --- hash provenance checks ---
    actual_sample_hash = header_data.get("sample_hash", "")
    if actual_sample_hash != expected_sample_hash:
        errors.append(ValidationError(
            file=path,
            incident_id=None,
            message=(
                f"sample_hash mismatch: expected {expected_sample_hash!r}, "
                f"got {actual_sample_hash!r}"
            ),
        ))

    actual_rubric_hash = header_data.get("rubric_hash", "")
    if actual_rubric_hash != expected_rubric_hash:
        errors.append(ValidationError(
            file=path,
            incident_id=None,
            message=(
                f"rubric_hash mismatch: expected {expected_rubric_hash!r}, "
                f"got {actual_rubric_hash!r}"
            ),
        ))

    actual_lock_hash = header_data.get("manifest_lock_hash", "")
    if actual_lock_hash != expected_lock_hash:
        errors.append(ValidationError(
            file=path,
            incident_id=None,
            message=(
                f"manifest_lock_hash mismatch: expected {expected_lock_hash!r}, "
                f"got {actual_lock_hash!r}"
            ),
        ))

    # --- per-incident checks ---
    for inc_data in data.get("incidents", []):
        incident_id = inc_data.get("incident_id", "<unknown>")
        labels = inc_data.get("labels")

        if expected_incident_ids is not None and incident_id not in expected_incident_ids:
            errors.append(ValidationError(
                file=path,
                incident_id=incident_id,
                message=f"incident_id {incident_id!r} not in expected corpus",
            ))

        if labels is None:
            errors.append(ValidationError(
                file=path,
                incident_id=incident_id,
                message=f"uncoded incident: labels is null — coding not yet complete",
            ))
            continue

        # labels must be a list
        if not isinstance(labels, list):
            errors.append(ValidationError(
                file=path,
                incident_id=incident_id,
                message=f"labels must be a list or null, got {type(labels).__name__}",
            ))
            continue

        # each label must be a known entry ID
        for label in labels:
            all_valid = valid_entry_ids | rollup_entry_ids
            if label not in all_valid:
                errors.append(ValidationError(
                    file=path,
                    incident_id=incident_id,
                    message=(
                        f"unknown label {label!r} — not in valid_entry_ids "
                        f"or rollup_entry_ids"
                    ),
                ))

    return errors


# ---------------------------------------------------------------------------
# code_synthetic
# ---------------------------------------------------------------------------


def code_synthetic(
    batch: CodingBatch,
    *,
    valid_entry_ids: set[str],
) -> CodingBatch:
    """Fill labels from the native_labels stored in each BatchIncident.

    This produces a coded batch using the native labels that were embedded
    at batch-generation time (via ``_native_labels``).  Labels that are not
    in ``valid_entry_ids`` are dropped.

    This function is intended for automated testing only.  It does not
    write to disk; callers are responsible for persisting the result.

    Parameters
    ----------
    batch:
        The uncoded CodingBatch produced by ``generate_batch``.
    valid_entry_ids:
        The set of canonical entry IDs to filter native labels against.

    Returns
    -------
    CodingBatch
        A new CodingBatch with labels filled from native_labels, filtered
        to only include entries in valid_entry_ids.
    """
    coded_incidents = []
    for inc in batch.incidents:
        valid_labels = [lbl for lbl in inc._native_labels if lbl in valid_entry_ids]
        coded_incidents.append(BatchIncident(
            incident_id=inc.incident_id,
            text=inc.text,
            labels=valid_labels,
            rollup_sub_labels=inc.rollup_sub_labels,
            notes=inc.notes,
            amendment=inc.amendment,
            _native_labels=inc._native_labels,
        ))
    return CodingBatch(header=batch.header, incidents=coded_incidents)


# ---------------------------------------------------------------------------
# code_synthetic_with_ground_truth
# ---------------------------------------------------------------------------


def code_synthetic_with_ground_truth(
    batch: CodingBatch,
    *,
    incidents_by_id: dict[str, IncidentRecord],
    valid_entry_ids: set[str],
) -> CodingBatch:
    """Fill labels from an external incidents_by_id lookup.

    Unlike ``code_synthetic``, this variant uses an external mapping from
    incident_id to IncidentRecord so it can be used even when the batch was
    deserialised from disk (losing ``_native_labels``).

    Parameters
    ----------
    batch:
        The uncoded CodingBatch produced by ``generate_batch`` (or read from
        disk via ``CodingBatch.read``).
    incidents_by_id:
        Mapping from incident_id to the original IncidentRecord.
    valid_entry_ids:
        The set of canonical entry IDs to filter native labels against.

    Returns
    -------
    CodingBatch
        A new CodingBatch with labels filled from native_labels for incidents
        found in ``incidents_by_id``.  Incidents not found receive ``[]``.
    """
    coded_incidents = []
    for inc in batch.incidents:
        record = incidents_by_id.get(inc.incident_id)
        if record is not None:
            valid_labels = [
                lbl for lbl in record.native_labels if lbl in valid_entry_ids
            ]
        else:
            valid_labels = []
        coded_incidents.append(BatchIncident(
            incident_id=inc.incident_id,
            text=inc.text,
            labels=valid_labels,
            rollup_sub_labels=inc.rollup_sub_labels,
            notes=inc.notes,
            amendment=inc.amendment,
        ))
    return CodingBatch(header=batch.header, incidents=coded_incidents)
