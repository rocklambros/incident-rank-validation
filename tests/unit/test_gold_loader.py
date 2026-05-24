# tests/unit/test_gold_loader.py
"""Tests for gold loader — manual curation + precision verification."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.calibrate.gold_loader import (
    load_gold_calibration,
    parse_entry_id_from_prefix,
)


class TestParseEntryIdFromPrefix:
    def test_simple_entry(self) -> None:
        assert parse_entry_id_from_prefix("MANUAL-LLM06-001") == "LLM06"

    def test_rollup_entry(self) -> None:
        assert parse_entry_id_from_prefix("MANUAL-ROLL-CFAS-001") == "ROLL-CFAS"

    def test_new_entry(self) -> None:
        assert parse_entry_id_from_prefix("MANUAL-NEW-MTIE-003") == "NEW-MTIE"

    def test_short_prefix_mapped_to_full_entry_id(self) -> None:
        assert parse_entry_id_from_prefix("MANUAL-MTIE-001") == "NEW-MTIE"
        assert parse_entry_id_from_prefix("MANUAL-ITSCD-004") == "NEW-ITSCD"
        assert parse_entry_id_from_prefix("MANUAL-CMSB-001") == "ROLL-CMSB"
        assert parse_entry_id_from_prefix("MANUAL-LAPTF-001") == "ROLL-LAPTF"
        assert parse_entry_id_from_prefix("MANUAL-CFAS-003") == "ROLL-CFAS"

    def test_invalid_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse entry ID"):
            parse_entry_id_from_prefix("BADPREFIX")

    def test_no_manual_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse entry ID"):
            parse_entry_id_from_prefix("GA-04821")


class TestLoadGoldCalibration:
    def test_loads_manual_curation_with_native_labels(self, tmp_path: Path) -> None:
        incidents = [
            {
                "id": "MANUAL-LLM01-001",
                "date": "2025-01-01",
                "text": "Test incident",
                "severity": "High",
                "source_class": "advisory",
                "corpus_stratum": "security",
                "quality": "curated",
                "native_labels": ["LLM01"],
                "source_url": "https://example.com",
            }
        ]
        curation_path = tmp_path / "manual_curated_incidents.json"
        curation_path.write_text(json.dumps(incidents))

        gold = load_gold_calibration(
            curation_path=curation_path,
            valid_entry_ids={"LLM01", "LLM06"},
            rubric_hash="test-hash",
            adjudicator_id="RL",
        )

        assert len(gold.recall_labels) == 1
        assert gold.recall_labels[0].true_entry_ids == ["LLM01"]
        assert gold.recall_labels[0].source == "manual-curated"

    def test_derives_entry_id_from_prefix_when_native_empty(
        self, tmp_path: Path
    ) -> None:
        incidents = [
            {
                "id": "MANUAL-LLM06-001",
                "date": "2025-01-01",
                "text": "Test incident",
                "severity": "High",
                "source_class": "advisory",
                "corpus_stratum": "ai-harm",
                "quality": "curated",
                "native_labels": [],
                "source_url": "https://example.com",
            }
        ]
        curation_path = tmp_path / "manual_curated_incidents.json"
        curation_path.write_text(json.dumps(incidents))

        gold = load_gold_calibration(
            curation_path=curation_path,
            valid_entry_ids={"LLM06"},
            rubric_hash="test-hash",
            adjudicator_id="RL",
        )

        assert len(gold.recall_labels) == 1
        assert gold.recall_labels[0].true_entry_ids == ["LLM06"]

    def test_rejects_invalid_entry_id(self, tmp_path: Path) -> None:
        incidents = [
            {
                "id": "MANUAL-FAKE-001",
                "date": "2025-01-01",
                "text": "Test incident",
                "severity": "High",
                "source_class": "advisory",
                "corpus_stratum": "security",
                "quality": "curated",
                "native_labels": [],
                "source_url": "https://example.com",
            }
        ]
        curation_path = tmp_path / "manual_curated_incidents.json"
        curation_path.write_text(json.dumps(incidents))

        with pytest.raises(ValueError, match="not in rubric"):
            load_gold_calibration(
                curation_path=curation_path,
                valid_entry_ids={"LLM01", "LLM06"},
                rubric_hash="test-hash",
                adjudicator_id="RL",
            )

    def test_loads_precision_verification(self, tmp_path: Path) -> None:
        precision_lines = [
            json.dumps({
                "incident_id": "GA-04821",
                "claimed_entry_id": "LLM06",
                "is_correct": True,
                "source": "stage2-verified",
                "adjudicator_id": "RL",
                "session_timestamp": "2026-06-15T14:30:00Z",
            })
        ]
        precision_path = tmp_path / "precision_verification.jsonl"
        precision_path.write_text("\n".join(precision_lines) + "\n")

        gold = load_gold_calibration(
            precision_path=precision_path,
            valid_entry_ids={"LLM06"},
            rubric_hash="test-hash",
            adjudicator_id="RL",
        )

        assert len(gold.precision_labels) == 1
        assert gold.precision_labels[0].is_correct is True
        assert gold.precision_labels[0].claimed_entry_id == "LLM06"

    def test_loads_directory_with_both_files(self, tmp_path: Path) -> None:
        incidents = [
            {
                "id": "MANUAL-LLM06-001",
                "date": "2025-01-01",
                "text": "Test",
                "severity": "High",
                "source_class": "advisory",
                "corpus_stratum": "ai-harm",
                "quality": "curated",
                "native_labels": [],
                "source_url": "https://example.com",
            }
        ]
        (tmp_path / "manual_curated_incidents.json").write_text(json.dumps(incidents))
        precision_line = json.dumps({
            "incident_id": "GA-002",
            "claimed_entry_id": "LLM06",
            "is_correct": False,
            "source": "stage2-verified",
            "adjudicator_id": "RL",
            "session_timestamp": "2026-06-15T14:30:00Z",
        })
        (tmp_path / "precision_verification.jsonl").write_text(precision_line + "\n")

        gold = load_gold_calibration(
            gold_dir=tmp_path,
            valid_entry_ids={"LLM06"},
            rubric_hash="test-hash",
            adjudicator_id="RL",
        )

        assert len(gold.recall_labels) == 1
        assert len(gold.precision_labels) == 1

    def test_deduplicates_against_base_ids(self, tmp_path: Path) -> None:
        incidents = [
            {
                "id": "MANUAL-LLM06-001",
                "date": "2025-01-01",
                "text": "Test",
                "severity": "High",
                "source_class": "advisory",
                "corpus_stratum": "ai-harm",
                "quality": "curated",
                "native_labels": [],
                "source_url": "https://example.com",
            }
        ]
        curation_path = tmp_path / "manual_curated_incidents.json"
        curation_path.write_text(json.dumps(incidents))

        gold = load_gold_calibration(
            curation_path=curation_path,
            valid_entry_ids={"LLM06"},
            rubric_hash="test-hash",
            adjudicator_id="RL",
            base_incident_ids={"MANUAL-LLM06-001"},
        )

        assert len(gold.recall_labels) == 0
