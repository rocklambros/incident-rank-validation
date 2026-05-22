from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

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


class TestInferPhase:
    def test_rejects_missing_calibration(self, tmp_path: Path) -> None:
        from engine.cli.pipeline_executor import execute_infer_phase

        cycle = tmp_path / "cycle"
        (cycle / "classify").mkdir(parents=True)
        (cycle / "classify" / "labeled_incidents.json").write_text("[]")
        (cycle / "prereg").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="posteriors"):
            execute_infer_phase(cycle)

    def test_nuts_failure_writes_diagnostics_file(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "infer"
        from engine.cli.pipeline_executor import write_nuts_failure

        write_nuts_failure(
            out_dir,
            error_message="R-hat exceeded 1.01 for lambda[3]",
            partial_samples=None,
        )
        assert (out_dir / "diagnostics_failure.txt").exists()
        assert "R-hat" in (out_dir / "diagnostics_failure.txt").read_text()

    def test_writes_inference_artifacts(self, tmp_path: Path) -> None:
        from engine.cli.pipeline_executor import write_infer_artifacts
        from engine.model.inference import InferenceResult

        result = InferenceResult(
            lambda_samples=np.array([[0.1, 0.2], [0.11, 0.19]]),
            entry_ids=("E1", "E2"),
            r_hat={"lambda[0]": 1.001, "lambda[1]": 1.002},
            ess={"lambda[0]": 800.0, "lambda[1]": 750.0},
            divergences=0,
            num_warmup=200,
            num_samples=500,
        )
        out_dir = tmp_path / "infer"
        write_infer_artifacts(result, out_dir)
        assert (out_dir / "lambda_samples.npy").exists()
        assert (out_dir / "inference_summary.json").exists()
        summary = json.loads((out_dir / "inference_summary.json").read_text())
        assert summary["num_samples"] == 500
        assert summary["divergences"] == 0
        assert summary["num_chains"] == 4


class TestDecidePhase:
    def test_writes_concordance_artifacts(self, tmp_path: Path) -> None:
        from engine.cli.pipeline_executor import write_decide_artifacts
        from engine.decide.concordance import ConcordanceResult, STANDING_CAVEAT

        concordance = ConcordanceResult(
            weighted_kappa_median=0.72,
            weighted_kappa_ci=(0.55, 0.88),
            measurable_count=15,
            total_count=20,
            coverage_ratio=0.75,
            below_prereg_minimum=False,
            meaningful_kappa_n=4,
            flags=(),
            standing_caveat=STANDING_CAVEAT,
        )
        out_dir = tmp_path / "results"
        write_decide_artifacts(
            concordance=concordance,
            out_dir=out_dir,
            rollup_results=(),
            selection_bias=None,
            twin_agreement=None,
            robustness=None,
        )
        assert (out_dir / "concordance.json").exists()
        data = json.loads((out_dir / "concordance.json").read_text())
        assert data["weighted_kappa_median"] == 0.72

    def test_writes_reproduction_bundle(self, tmp_path: Path) -> None:
        from engine.cli.pipeline_executor import write_reproduction_bundle

        out_dir = tmp_path / "results"
        write_reproduction_bundle(
            out_dir=out_dir,
            cycle_id="2026",
            engine_version="1.0.0",
            snapshot_hash="snap123",
            manifest_hash="man456",
            lockfile_hash="lock789",
            stage2_manifest_hash="s2hash",
            calibration_hash="calhash",
            vote_data_hash="votehash",
        )
        assert (out_dir / "repro_bundle.json").exists()
        data = json.loads((out_dir / "repro_bundle.json").read_text())
        assert data["provenance"]["stage2_manifest_hash"] == "s2hash"
