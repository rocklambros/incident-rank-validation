from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReproductionBundle:
    cycle_id: str
    engine_version: str
    snapshot_hash: str
    manifest_hash: str
    lockfile_hash: str
    provenance: dict[str, str]

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "cycle_id": self.cycle_id,
                    "engine_version": self.engine_version,
                    "snapshot_hash": self.snapshot_hash,
                    "manifest_hash": self.manifest_hash,
                    "lockfile_hash": self.lockfile_hash,
                    "provenance": self.provenance,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: Path) -> ReproductionBundle:
        d = json.loads(path.read_text())
        return cls(**d)
