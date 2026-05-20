"""Tests for the snapshot vendoring CLI module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.cli.snapshot import vendor_snapshot
from engine.snapshot.drift import DriftReport, detect_drift
from engine.snapshot.provenance import SnapshotProvenance


@pytest.fixture()
def source_corpus(tmp_path: Path) -> Path:
    """Create a minimal source corpus JSON file."""
    records = [
        {"id": "INC-001", "title": "Test incident", "corpus": "security"},
        {"id": "INC-002", "title": "Another incident", "corpus": "ai-harm"},
    ]
    src = tmp_path / "source" / "incidents.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps(records, indent=2))
    return src


@pytest.fixture()
def dest_dir(tmp_path: Path) -> Path:
    return tmp_path / "dest"


class TestVendorSnapshot:

    def test_creates_content_addressed_directory(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="genai_agentic_incidents",
            source_commit_sha="abc123",
            adapter_version="0.2.0",
        )
        assert result.snapshot_dir.exists()
        assert result.snapshot_dir.name == result.snapshot_hash

    def test_snapshot_file_is_byte_identical_to_source(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="genai_agentic_incidents",
            source_commit_sha="abc123",
            adapter_version="0.2.0",
        )
        vendored = result.snapshot_dir / "incidents.json"
        assert vendored.read_bytes() == source_corpus.read_bytes()

    def test_provenance_json_has_all_six_fields(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="genai_agentic_incidents",
            source_commit_sha="abc123",
            adapter_version="0.2.0",
        )
        prov_path = result.snapshot_dir / "provenance.json"
        assert prov_path.exists()
        prov = SnapshotProvenance.read(prov_path)
        assert prov.source_repo == "genai_agentic_incidents"
        assert prov.source_commit_sha == "abc123"
        assert prov.pull_date != ""
        assert prov.adapter_name == "genai_agentic"
        assert prov.adapter_version == "0.2.0"
        assert prov.snapshot_hash == result.snapshot_hash

    def test_hash_is_deterministic_across_calls(
        self, source_corpus: Path, tmp_path: Path
    ) -> None:
        r1 = vendor_snapshot(
            source_path=source_corpus,
            dest_base=tmp_path / "d1",
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        r2 = vendor_snapshot(
            source_path=source_corpus,
            dest_base=tmp_path / "d2",
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        assert r1.snapshot_hash == r2.snapshot_hash

    def test_idempotent_rerun_does_not_fail(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        assert result.snapshot_dir.exists()

    def test_idempotent_rerun_preserves_provenance(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        """Premortem R2: re-running vendor_snapshot must not overwrite provenance."""
        r1 = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        original_date = r1.provenance.pull_date
        r2 = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        assert r2.provenance.pull_date == original_date

    def test_jsonl_file_written_alongside_json(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        """Premortem M2: JSONL must be written unconditionally for drift detector."""
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        jsonl_path = result.snapshot_dir / "incidents.jsonl"
        assert jsonl_path.exists(), "incidents.jsonl not created by vendor_snapshot"
        lines = [ln for ln in jsonl_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_jsonl_round_trips_json_content(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        """JSONL records must be identical to the JSON array entries."""
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        json_data = json.loads((result.snapshot_dir / "incidents.json").read_text())
        jsonl_lines = [
            json.loads(ln)
            for ln in (result.snapshot_dir / "incidents.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert json_data == jsonl_lines


class TestDriftIntegration:

    def test_first_snapshot_produces_baseline_report(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        """First snapshot has no predecessor — drift report exists but has no anomalies."""
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        snapshot_jsonl = result.snapshot_dir / "incidents.jsonl"
        assert snapshot_jsonl.exists(), "vendor_snapshot must write incidents.jsonl"
        report = detect_drift(snapshot_jsonl, snapshot_jsonl)
        assert isinstance(report, DriftReport)
        assert report.requires_signoff is False
        assert len(report.anomalies) == 0

    def test_drift_detected_on_changed_snapshot(self, tmp_path: Path) -> None:
        """A modified snapshot triggers drift anomalies."""
        prev_path = tmp_path / "prev.jsonl"
        curr_path = tmp_path / "curr.jsonl"

        prev_lines = [
            json.dumps({"owasp_llm": ["LLM03"]}) for _ in range(100)
        ]
        prev_path.write_text("\n".join(prev_lines))

        curr_lines = [
            json.dumps({"owasp_llm": ["LLM03"]}) for _ in range(200)
        ]
        curr_path.write_text("\n".join(curr_lines))

        report = detect_drift(prev_path, curr_path)
        assert isinstance(report, DriftReport)
        assert report.requires_signoff is True
        assert len(report.anomalies) > 0
