"""F-C integration tests: real-snapshot minimal-cycle fixture.

The fixture is tmp-only; it never reads from or writes under
``projects/owasp-llm/cycles/2026/``.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.mark.integration
def test_builder_happy_path(
    real_minimal_cycle: Callable[..., Path], tmp_path: Path
) -> None:
    from engine.calibrate.coverage import (
        COVERAGE_FILENAME,
        _resolve_snapshot_incidents,
        verify_labeled_completeness,
    )
    from engine.snapshot.hashing import snapshot_hash

    cycle = real_minimal_cycle(tmp_path)
    assert tmp_path in cycle.parents or cycle == tmp_path
    manifest = json.loads((cycle / "prereg" / "manifest.json").read_text())
    snap_dirs = list((cycle / "corpora" / "genai_agentic").iterdir())
    assert len(snap_dirs) == 1
    H = snap_dirs[0].name
    assert manifest["snapshot_hash"] == H == snapshot_hash(snap_dirs[0] / "incidents.json")
    assert manifest["overlap_min_fp"] == 1
    assert _resolve_snapshot_incidents(cycle, H) is not None  # guard is LIVE
    marker = json.loads((cycle / "classify" / COVERAGE_FILENAME).read_text())
    assert (marker["n_corpus"], marker["n_in_scope"], marker["n_oos"]) == (6, 4, 2)
    labeled = json.loads((cycle / "classify" / "labeled_incidents.json").read_text())
    labeled_ids = {str(r["incident_id"]) for r in labeled}
    verify_labeled_completeness(cycle, H, labeled_ids)  # must NOT raise
