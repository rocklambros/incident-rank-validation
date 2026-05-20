"""Snapshot vendoring for corpus data.

Vendors a source corpus JSON file into a content-addressed directory with
provenance metadata.  See HANDOFF §5.1, §6 control 9.

Premortem M2: vendor_snapshot() UNCONDITIONALLY writes incidents.jsonl alongside
incidents.json.  The drift detector (engine/snapshot/drift.py) reads JSONL
(one JSON object per line), not JSON arrays.  This is not optional — without
the JSONL file, detect_drift() will raise JSONDecodeError on the vendored
snapshot.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import click

from engine.snapshot.hashing import snapshot_hash
from engine.snapshot.provenance import SnapshotProvenance


@dataclass(frozen=True, slots=True)
class VendorResult:
    """Result of a snapshot vendoring operation."""

    snapshot_dir: Path
    snapshot_hash: str
    provenance: SnapshotProvenance


def _write_jsonl(source_json_path: Path, dest_jsonl_path: Path) -> None:
    """Convert a JSON array file to JSONL format (one JSON object per line).

    Required by engine/snapshot/drift.py which reads JSONL, not JSON arrays.
    Premortem M2: this is MANDATORY, not conditional.

    Handles both flat JSON arrays and dicts with an ``incidents`` key
    (the real corpus uses ``{"version": ..., "incidents": [...]}``;
    the adapter extracts records, but the vendoring script copies
    the source byte-for-byte, so the JSONL conversion must handle
    whatever top-level structure the source file has).
    """
    data = json.loads(source_json_path.read_text())
    if isinstance(data, dict) and "incidents" in data:
        records = data["incidents"]
    elif isinstance(data, list):
        records = data
    else:
        raise TypeError(
            f"Expected JSON array or dict with 'incidents' key, "
            f"got {type(data).__name__}"
        )
    with dest_jsonl_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def vendor_snapshot(
    *,
    source_path: Path,
    dest_base: Path,
    source_repo: str,
    source_commit_sha: str,
    adapter_version: str,
) -> VendorResult:
    """Vendor a source corpus file into a content-addressed snapshot directory."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source corpus not found: {source_path}")

    content_hash = snapshot_hash(source_path)
    snapshot_dir = dest_base / content_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    dest_file = snapshot_dir / "incidents.json"
    if not dest_file.exists():
        shutil.copy2(source_path, dest_file)

    jsonl_file = snapshot_dir / "incidents.jsonl"
    if not jsonl_file.exists():
        _write_jsonl(dest_file, jsonl_file)

    prov_path = snapshot_dir / "provenance.json"
    if prov_path.exists():
        provenance = SnapshotProvenance.read(prov_path)
    else:
        provenance = SnapshotProvenance(
            source_repo=source_repo,
            source_commit_sha=source_commit_sha,
            pull_date=date.today().isoformat(),
            adapter_name="genai_agentic",
            adapter_version=adapter_version,
            snapshot_hash=content_hash,
        )
        provenance.write(prov_path)

    return VendorResult(
        snapshot_dir=snapshot_dir,
        snapshot_hash=content_hash,
        provenance=provenance,
    )


@click.command("vendor-snapshot")
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to source corpus JSON file.",
)
@click.option(
    "--dest",
    required=True,
    type=click.Path(path_type=Path),
    help="Base directory for vendored snapshots.",
)
@click.option("--source-repo", required=True, help="Name of the source repository.")
@click.option("--source-commit", required=True, help="Git commit SHA of the source.")
@click.option(
    "--adapter-version",
    default="0.2.0",
    help="Adapter semver version.",
)
def vendor_snapshot_cmd(
    source: Path,
    dest: Path,
    source_repo: str,
    source_commit: str,
    adapter_version: str,
) -> None:
    """Vendor a corpus snapshot with content-addressed hashing and provenance."""
    result = vendor_snapshot(
        source_path=source,
        dest_base=dest,
        source_repo=source_repo,
        source_commit_sha=source_commit,
        adapter_version=adapter_version,
    )
    click.echo(f"Snapshot vendored to: {result.snapshot_dir}")
    click.echo(f"Content hash: {result.snapshot_hash}")
    click.echo(f"Provenance: {result.snapshot_dir / 'provenance.json'}")
