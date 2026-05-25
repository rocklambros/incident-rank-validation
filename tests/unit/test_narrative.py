"""Tests for engine.report.narrative — standalone narrative report generator."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

CYCLE_DIR = Path("projects/owasp-llm/cycles/2026")
SKIP_NO_CYCLE = pytest.mark.skipif(not CYCLE_DIR.exists(), reason="Cycle data not present")


@SKIP_NO_CYCLE
class TestNarrativeDataLoading:
    def test_load_data_returns_dict(self) -> None:
        from engine.report.narrative_data import load_narrative_data
        data = load_narrative_data(CYCLE_DIR)
        assert isinstance(data, dict)

    def test_load_data_has_required_keys(self) -> None:
        from engine.report.narrative_data import load_narrative_data
        data = load_narrative_data(CYCLE_DIR)
        required = {"rubric", "incidents", "prelabels", "goldset", "precision_verification", "posteriors", "diagnostic", "inference_summary", "lambda_samples", "concordance", "selection_bias", "rank_comparison_md"}
        missing = required - set(data.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_lambda_samples_shape(self) -> None:
        from engine.report.narrative_data import load_narrative_data
        data = load_narrative_data(CYCLE_DIR)
        assert data["lambda_samples"].shape == (16000, 20)

    def test_concordance_has_ci_method(self) -> None:
        from engine.report.narrative_data import load_narrative_data
        data = load_narrative_data(CYCLE_DIR)
        assert "ci_method" in data["concordance"]
