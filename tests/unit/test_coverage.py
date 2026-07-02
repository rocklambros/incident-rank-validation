"""Unit tests for the F-B labeled_incidents completeness guard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.calibrate.coverage import (
    COVERAGE_FILENAME,
    ClassifyCoverage,
    LabeledIncidentsIncompleteError,
    read_snapshot_universe_ids,
    verify_labeled_completeness,
    write_classify_coverage,
)


def _make_snapshot(cycle: Path, snapshot_hash: str, ids: list[str]) -> None:
    snap_dir = cycle / "corpora" / "genai_agentic" / snapshot_hash
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "incidents.json").write_text(
        json.dumps({"incident_count": len(ids), "incidents": [{"id": i} for i in ids]})
    )


def _write_marker(cycle: Path, **fields: object) -> None:
    (cycle / "classify").mkdir(parents=True, exist_ok=True)
    (cycle / "classify" / COVERAGE_FILENAME).write_text(json.dumps(fields))


# --- writer ---------------------------------------------------------------

def test_write_marker_records_split_and_invariant(tmp_path: Path) -> None:
    cov = write_classify_coverage(
        tmp_path / "classify",
        snapshot_hash="h",
        corpus_incident_ids={"a", "b", "c", "d"},
        in_scope_incident_ids={"a", "b"},
    )
    assert cov == ClassifyCoverage(snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=2)
    data = json.loads((tmp_path / "classify" / COVERAGE_FILENAME).read_text())
    assert data == {"snapshot_hash": "h", "n_corpus": 4, "n_in_scope": 2, "n_oos": 2}


def test_write_marker_rejects_in_scope_id_absent_from_corpus(tmp_path: Path) -> None:
    with pytest.raises(LabeledIncidentsIncompleteError, match="absent from the corpus"):
        write_classify_coverage(
            tmp_path / "classify",
            snapshot_hash="h",
            corpus_incident_ids={"a", "b"},
            in_scope_incident_ids={"a", "z"},
        )


# --- verifier: no-op paths ------------------------------------------------

def test_verify_noop_when_synthetic_snapshot(tmp_path: Path) -> None:
    # No corpora dir resolvable -> no authoritative universe -> no-op.
    verify_labeled_completeness(tmp_path, "synthetic-no-snapshot", {"a", "b"})


def test_verify_noop_when_snapshot_dir_absent(tmp_path: Path) -> None:
    verify_labeled_completeness(tmp_path, "deadbeef", {"a", "b"})


# --- verifier: pass path (genuine OOS does NOT raise) ---------------------

def test_verify_passes_with_complete_marker_and_genuine_oos(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])  # 4 corpus incidents
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=2)
    # labeled = {a, b}; c and d are genuinely OOS (absent). goldset references c
    # (a recall incident the classifier put OOS) -> must NOT raise.
    verify_labeled_completeness(tmp_path, "h", {"a", "b"}, goldset_recall_ids={"a", "c"})


# --- verifier: proven-gap raises -----------------------------------------

def test_verify_raises_on_foreign_labeled_id(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=2, n_in_scope=2, n_oos=0)
    with pytest.raises(LabeledIncidentsIncompleteError, match="absent from the pinned"):
        verify_labeled_completeness(tmp_path, "h", {"a", "z"})


def test_verify_raises_on_missing_marker(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b", "c"])
    with pytest.raises(LabeledIncidentsIncompleteError, match="coverage marker"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_on_marker_snapshot_hash_mismatch(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b"])
    _write_marker(tmp_path, snapshot_hash="OTHER", n_corpus=2, n_in_scope=2, n_oos=0)
    with pytest.raises(LabeledIncidentsIncompleteError, match="different snapshot"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_on_partial_corpus_coverage(tmp_path: Path) -> None:
    # Snapshot has 4 incidents but the producer recorded n_corpus=2 -> partial run.
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=2, n_in_scope=2, n_oos=0)
    with pytest.raises(LabeledIncidentsIncompleteError, match="partial corpus"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_on_inconsistent_marker(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=1)
    with pytest.raises(LabeledIncidentsIncompleteError, match="internally inconsistent"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_when_labeled_count_below_marker(tmp_path: Path) -> None:
    # Marker says 3 in-scope but labeled_incidents.json now has 2 -> truncated file.
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=3, n_oos=1)
    with pytest.raises(LabeledIncidentsIncompleteError, match="truncated"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})


def test_verify_raises_when_labeled_count_above_marker(tmp_path: Path) -> None:
    # Marker says 2 in-scope but labeled_incidents.json now has 3 -> altered file.
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=2)
    with pytest.raises(LabeledIncidentsIncompleteError, match="does not reconcile"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b", "c"})


def test_verify_raises_on_goldset_incident_absent_from_snapshot(tmp_path: Path) -> None:
    _make_snapshot(tmp_path, "h", ["a", "b", "c", "d"])
    _write_marker(tmp_path, snapshot_hash="h", n_corpus=4, n_in_scope=2, n_oos=2)
    with pytest.raises(LabeledIncidentsIncompleteError, match="goldset"):
        verify_labeled_completeness(
            tmp_path, "h", {"a", "b"}, goldset_recall_ids={"a", "MANUAL-X-001"}
        )


# --- shared reader: unfiltered raw universe --------------------------------

def test_read_snapshot_universe_ids_is_unfiltered(tmp_path: Path) -> None:
    snap = tmp_path / "incidents.json"
    # one record carries a future date the corpus adapter would drop; the
    # universe reader must still include it (producer + verifier agree on RAW).
    snap.write_text(json.dumps({"incidents": [
        {"id": "INC-1", "date": "2026-01-01"},
        {"id": "INC-2", "date": "2099-12-31"},
    ]}))
    assert read_snapshot_universe_ids(snap) == {"INC-1", "INC-2"}


# --- 1-level layout resolution --------------------------------------------

def _make_snapshot_1level(cycle: Path, snapshot_hash: str, ids: list[str]) -> None:
    snap_dir = cycle / "corpora" / snapshot_hash
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "incidents.json").write_text(
        json.dumps({"incident_count": len(ids), "incidents": [{"id": i} for i in ids]})
    )


def test_verify_resolves_one_level_layout(tmp_path: Path) -> None:
    _make_snapshot_1level(tmp_path, "h", ["a", "b", "c"])
    # snapshot resolves (1-level) but no marker -> guard must fire, not no-op.
    with pytest.raises(LabeledIncidentsIncompleteError, match="coverage marker"):
        verify_labeled_completeness(tmp_path, "h", {"a", "b"})
