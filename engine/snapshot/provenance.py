"""Snapshot provenance metadata.

Records the origin of a vendored corpus snapshot so that the gold-set artifact
can be bound to a specific snapshot content hash.  See HANDOFF §5.1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SnapshotProvenance:
    source_repo: str          # e.g., "genai_agentic_incidents"
    source_commit_sha: str    # git commit of the source at snapshot time
    pull_date: str            # ISO 8601 date
    adapter_name: str         # e.g., "corpus_a", "synthetic"
    adapter_version: str      # semver
    snapshot_hash: str        # SHA-256 of snapshot file

    def to_json(self) -> str:
        """Serialize to canonical JSON (sorted keys, no timestamps in hash payload)."""
        return (
            json.dumps(
                {
                    "source_repo": self.source_repo,
                    "source_commit_sha": self.source_commit_sha,
                    "pull_date": self.pull_date,
                    "adapter_name": self.adapter_name,
                    "adapter_version": self.adapter_version,
                    "snapshot_hash": self.snapshot_hash,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> SnapshotProvenance:
        d = json.loads(text)
        return cls(**d)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: Path) -> SnapshotProvenance:
        return cls.from_json(path.read_text())
