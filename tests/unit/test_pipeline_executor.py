from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.classify.cost_tracker import CostTracker
from engine.classify.runpod_client import RunPodResponse
from engine.classify.stage2_protocol import Stage2Classification
from engine.schema import IncidentRecord


class TestClassifyPhase:
    def test_stage1_routes_ambiguous_to_stage2(self) -> None:
        from engine.cli.pipeline_executor import route_to_stage2
        from engine.classify.stub import Classification

        classifications = (
            Classification(incident_id="INC-001", entry_id="LLM01", confidence=0.8, stage=1, rationale="high"),
            Classification(incident_id="INC-002", entry_id="LLM02", confidence=0.1, stage=1, rationale="low"),
            Classification(incident_id="INC-003", entry_id="LLM01", confidence=0.25, stage=1, rationale="ambig"),
        )
        ambiguous = route_to_stage2(classifications, confidence_threshold=0.3)
        assert set(ambiguous) == {"INC-002", "INC-003"}

    def test_merge_stage1_stage2_results(self) -> None:
        from engine.cli.pipeline_executor import merge_classifications
        from engine.classify.stub import Classification

        stage1 = (
            Classification(incident_id="INC-001", entry_id="LLM01", confidence=0.8, stage=1, rationale="high"),
            Classification(incident_id="INC-002", entry_id="LLM02", confidence=0.1, stage=1, rationale="low"),
        )
        stage2 = (
            Stage2Classification(
                incident_id="INC-002", entry_id="LLM03", confidence=0.9,
                rationale="Stage-2 reclassified", model_identity="test",
                weight_provenance_hash="h", prompt_hash="p",
            ),
        )
        merged = merge_classifications(stage1, stage2, confidence_threshold=0.3)
        ids = {c.incident_id: c.entry_id for c in merged}
        assert ids["INC-001"] == "LLM01"
        assert ids["INC-002"] == "LLM03"

    def test_classify_writes_artifacts(self, tmp_path: Path) -> None:
        from engine.cli.pipeline_executor import write_classify_artifacts
        from engine.classify.stub import Classification, ClassificationResult

        result = ClassificationResult(
            classifications=(
                Classification(incident_id="INC-001", entry_id="LLM01", confidence=0.8, stage=1, rationale="test"),
            ),
            classifier_version="stage1-keyword-1.0.0",
            classifier_rule_hash="abc123",
        )
        out_dir = tmp_path / "classify"
        write_classify_artifacts(result, out_dir, stage2_results=())
        assert (out_dir / "labeled_incidents.json").exists()
        assert (out_dir / "stage1_results.json").exists()
        data = json.loads((out_dir / "labeled_incidents.json").read_text())
        assert len(data) == 1
        assert data[0]["incident_id"] == "INC-001"
