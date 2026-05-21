"""Tests for engine.calibrate.batch — batch generation, validation, synthetic coding."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.calibrate.batch import (
    BatchHeader,
    BatchIncident,
    CodingBatch,
    ValidationError,
    code_synthetic,
    generate_batch,
    validate_coded_batch,
)
from engine.calibrate.sampler import SampleFrame, SampleRequest, SampleResult
from engine.schema import IncidentRecord


def _make_incident(
    id: str = "GA-001",
    text: str = "test incident",
    labels: tuple[str, ...] = ("LLM01",),
    stratum: str = "security",
) -> IncidentRecord:
    return IncidentRecord(
        id=id, date="2026-01-01", text=text, severity="High",
        source_class="advisory", corpus_stratum=stratum, quality="curated",
        native_labels=labels, source_url="https://example.com",
    )


def _sample_result(incidents: tuple[IncidentRecord, ...]) -> SampleResult:
    return SampleResult(
        incidents=incidents,
        request=SampleRequest(
            frame=SampleFrame.PRECISION, entry_id="LLM01", stratum="security", n=10,
        ),
        actual_n=len(incidents),
        sample_hash="abc123",
    )


class TestGenerateBatch:
    def test_generates_precision_batch(self) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr,
            rubric_hash="rub_hash",
            manifest_lock_hash="lock_hash",
            coder_id="rock-lambros",
            cycle_id="2026",
        )
        assert batch.header.frame == "precision"
        assert batch.header.entry_id == "LLM01"
        assert batch.header.sample_hash == "abc123"
        assert len(batch.incidents) == 1
        assert batch.incidents[0].labels is None

    def test_recall_batch_includes_checklist(self) -> None:
        inc = _make_incident()
        sr = SampleResult(
            incidents=(inc,),
            request=SampleRequest(
                frame=SampleFrame.RECALL, entry_id=None, stratum="security", n=10,
            ),
            actual_n=1,
            sample_hash="abc123",
        )
        checklist = {"LLM01": "Prompt Injection", "LLM02": "Data Leak"}
        batch = generate_batch(
            sample_result=sr,
            rubric_hash="rub_hash",
            manifest_lock_hash="lock_hash",
            coder_id="rock-lambros",
            cycle_id="2026",
            coding_checklist=checklist,
        )
        assert batch.header.coding_checklist == checklist


class TestValidateCodedBatch:
    def test_valid_batch_passes(self, tmp_path: Path) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="rock", cycle_id="2026",
        )
        batch_data = batch.to_dict()
        batch_data["incidents"][0]["labels"] = ["LLM01"]
        path = tmp_path / "batch.json"
        path.write_text(json.dumps(batch_data, indent=2))
        errors = validate_coded_batch(
            path, valid_entry_ids={"LLM01", "LLM02"},
            rollup_entry_ids=set(),
            expected_sample_hash="abc123",
            expected_rubric_hash="rub",
            expected_lock_hash="lock",
        )
        assert errors == []

    def test_null_labels_generates_warning(self, tmp_path: Path) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="rock", cycle_id="2026",
        )
        batch_data = batch.to_dict()
        path = tmp_path / "batch.json"
        path.write_text(json.dumps(batch_data, indent=2))
        errors = validate_coded_batch(
            path, valid_entry_ids={"LLM01"},
            rollup_entry_ids=set(),
            expected_sample_hash="abc123",
            expected_rubric_hash="rub",
            expected_lock_hash="lock",
        )
        assert any("uncoded" in str(e).lower() for e in errors)

    def test_unknown_label_raises(self, tmp_path: Path) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="rock", cycle_id="2026",
        )
        batch_data = batch.to_dict()
        batch_data["incidents"][0]["labels"] = ["UNKNOWN_ENTRY"]
        path = tmp_path / "batch.json"
        path.write_text(json.dumps(batch_data, indent=2))
        errors = validate_coded_batch(
            path, valid_entry_ids={"LLM01"},
            rollup_entry_ids=set(),
            expected_sample_hash="abc123",
            expected_rubric_hash="rub",
            expected_lock_hash="lock",
        )
        assert any("unknown" in str(e).lower() for e in errors)

    def test_hash_mismatch_raises(self, tmp_path: Path) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="rock", cycle_id="2026",
        )
        batch_data = batch.to_dict()
        batch_data["incidents"][0]["labels"] = ["LLM01"]
        path = tmp_path / "batch.json"
        path.write_text(json.dumps(batch_data, indent=2))
        errors = validate_coded_batch(
            path, valid_entry_ids={"LLM01"},
            rollup_entry_ids=set(),
            expected_sample_hash="WRONG",
            expected_rubric_hash="rub",
            expected_lock_hash="lock",
        )
        assert any("sample_hash" in str(e).lower() for e in errors)


class TestCodeSynthetic:
    def test_fills_labels_from_native(self) -> None:
        inc = _make_incident(labels=("LLM01", "LLM05"))
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="synthetic", cycle_id="2026",
        )
        coded = code_synthetic(batch, valid_entry_ids={"LLM01", "LLM02", "LLM05"})
        assert coded.incidents[0].labels == ["LLM01", "LLM05"]

    def test_empty_labels_when_no_native(self) -> None:
        inc = _make_incident(labels=())
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="synthetic", cycle_id="2026",
        )
        coded = code_synthetic(batch, valid_entry_ids={"LLM01"})
        assert coded.incidents[0].labels == []
