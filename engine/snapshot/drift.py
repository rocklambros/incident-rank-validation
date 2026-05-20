"""Between-snapshot drift and anomaly detection with manual signoff enforcement.

See HANDOFF §5.1 and §6 control 9:
  A between-snapshot drift and anomaly check runs at snapshot time.
  A drifted or anomalous snapshot requires manual sign-off before a cycle consumes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DriftAnomaly:
    """A single per-entry drift finding."""

    entry_id: str
    metric: str  # e.g., "count_delta", "burst"
    previous_value: float
    current_value: float
    description: str


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Summary of drift between two snapshots."""

    previous_snapshot_hash: str
    current_snapshot_hash: str
    anomalies: tuple[DriftAnomaly, ...]
    requires_signoff: bool  # True if any anomaly exceeds threshold

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "previous_snapshot_hash": self.previous_snapshot_hash,
                    "current_snapshot_hash": self.current_snapshot_hash,
                    "anomalies": [
                        {
                            "entry_id": a.entry_id,
                            "metric": a.metric,
                            "previous_value": a.previous_value,
                            "current_value": a.current_value,
                            "description": a.description,
                        }
                        for a in self.anomalies
                    ],
                    "requires_signoff": self.requires_signoff,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )


class DriftSignoffRequired(Exception):
    """Raised when drift requires manual signoff."""

    def __init__(self, report: DriftReport) -> None:
        self.report = report
        super().__init__(
            f"Drift signoff required: {len(report.anomalies)} anomalies detected. "
            "Pass --accept-drift-signoff '<reason ≥30 chars>' to proceed."
        )


def _count_entries(path: Path) -> dict[str, int]:
    """Count per-entry occurrences from a JSONL snapshot file.

    Each line is a JSON object.  The ``owasp_llm`` field, if present, is a
    list of entry IDs; each occurrence increments that entry's count.
    """
    counts: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for entry_id in record.get("owasp_llm", []):
            counts[entry_id] = counts.get(entry_id, 0) + 1
    return counts


def detect_drift(
    prev: Path,
    curr: Path,
    relative_threshold: float = 0.2,
    absolute_threshold: int = 50,
    burst_threshold: int = 10,
) -> DriftReport:
    """Detect drift and anomalies between two consecutive weekly snapshots.

    Drift is flagged when an entry's count changes by more than
    *both* ``relative_threshold`` (relative) *and* ``absolute_threshold``
    (absolute) — whichever bound is larger governs, so the two parameters
    work as an OR condition: flag if the change exceeds the relative threshold
    OR if the absolute change exceeds ``absolute_threshold``.

    Burst is flagged independently: a previously-zero entry that now has more
    than ``burst_threshold`` incidents is suspicious regardless of relative
    change.

    Args:
        prev: Path to the previous JSONL snapshot file.
        curr: Path to the current JSONL snapshot file.
        relative_threshold: Fractional change that triggers a drift anomaly
            (default 0.20 = 20 %).
        absolute_threshold: Absolute count change that triggers a drift anomaly
            (default 50).
        burst_threshold: Minimum count in the current snapshot for a
            previously-zero entry to trigger a burst anomaly (default 10).

    Returns:
        A :class:`DriftReport` with ``requires_signoff=True`` if any anomalies
        were found.
    """
    from engine.snapshot.hashing import snapshot_hash

    prev_hash = snapshot_hash(prev)
    curr_hash = snapshot_hash(curr)

    prev_counts = _count_entries(prev)
    curr_counts = _count_entries(curr)

    all_entry_ids = set(prev_counts) | set(curr_counts)
    anomalies: list[DriftAnomaly] = []

    for entry_id in sorted(all_entry_ids):
        prev_val = float(prev_counts.get(entry_id, 0))
        curr_val = float(curr_counts.get(entry_id, 0))
        delta = curr_val - prev_val
        abs_delta = abs(delta)

        # Burst detection: zero → above burst_threshold
        if prev_val == 0.0 and curr_val > burst_threshold:
            anomalies.append(
                DriftAnomaly(
                    entry_id=entry_id,
                    metric="burst",
                    previous_value=prev_val,
                    current_value=curr_val,
                    description=(
                        f"Entry {entry_id!r} surged from 0 to {int(curr_val)} "
                        f"(burst threshold: {burst_threshold})."
                    ),
                )
            )
            # Do not double-count as count_delta for the same entry.
            continue

        # Count-delta drift detection.
        # Flag if absolute change > absolute_threshold OR relative change >
        # relative_threshold (denominator is the previous count; skip when
        # prev_val == 0 to avoid division-by-zero — burst covers that case).
        if prev_val == 0.0:
            # prev=0, curr<=burst_threshold: no anomaly.
            continue

        relative_change = abs_delta / prev_val
        if abs_delta > absolute_threshold or relative_change > relative_threshold:
            anomalies.append(
                DriftAnomaly(
                    entry_id=entry_id,
                    metric="count_delta",
                    previous_value=prev_val,
                    current_value=curr_val,
                    description=(
                        f"Entry {entry_id!r} changed by {delta:+.0f} "
                        f"({relative_change:.1%} relative, "
                        f"absolute_threshold={absolute_threshold}, "
                        f"relative_threshold={relative_threshold:.0%})."
                    ),
                )
            )

    return DriftReport(
        previous_snapshot_hash=prev_hash,
        current_snapshot_hash=curr_hash,
        anomalies=tuple(anomalies),
        requires_signoff=len(anomalies) > 0,
    )
