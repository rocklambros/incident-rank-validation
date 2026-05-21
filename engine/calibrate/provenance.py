"""Stage provenance chain — binds each pipeline stage's inputs to outputs.

Each CLI stage writes a StageProvenance record. The next stage verifies
input_hashes match the previous stage's output_hash, preventing stale
intermediate artifacts from propagating.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StageProvenance:
    stage_name: str
    manifest_lock_hash: str
    input_hashes: dict[str, str]
    output_hash: str
    timestamp: str
    engine_version: str


def write_provenance(prov: StageProvenance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "stage_name": prov.stage_name,
        "manifest_lock_hash": prov.manifest_lock_hash,
        "input_hashes": prov.input_hashes,
        "output_hash": prov.output_hash,
        "timestamp": prov.timestamp,
        "engine_version": prov.engine_version,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_provenance(path: Path) -> StageProvenance:
    data = json.loads(path.read_text())
    return StageProvenance(
        stage_name=data["stage_name"],
        manifest_lock_hash=data["manifest_lock_hash"],
        input_hashes=data["input_hashes"],
        output_hash=data["output_hash"],
        timestamp=data["timestamp"],
        engine_version=data["engine_version"],
    )


def verify_input_hashes(
    expected: dict[str, str],
    provenance_dir: Path,
) -> None:
    for stage_name, expected_hash in expected.items():
        prov_path = provenance_dir / f"{stage_name}_provenance.json"
        if not prov_path.exists():
            raise ValueError(
                f"provenance file not found for stage '{stage_name}': {prov_path}"
            )
        prov = read_provenance(prov_path)
        if prov.output_hash != expected_hash:
            raise ValueError(
                f"provenance mismatch for stage '{stage_name}': "
                f"expected output_hash={expected_hash}, got {prov.output_hash}"
            )


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_json(data: object) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
