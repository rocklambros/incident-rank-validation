"""Flip-site wiring for the F-B completeness guard (execute_infer_phase, cal_tally)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

from pathlib import Path

import pytest

from engine.calibrate.coverage import (
    COVERAGE_FILENAME,
    LabeledIncidentsIncompleteError,
    write_classify_coverage,
)


def _snapshot(cycle: Path, snap: str, ids: list[str]) -> None:
    d = cycle / "corpora" / "genai_agentic" / snap
    d.mkdir(parents=True, exist_ok=True)
    (d / "incidents.json").write_text(
        json.dumps({"incident_count": len(ids), "incidents": [{"id": i} for i in ids]})
    )


def _labeled(cycle: Path, rows: list[tuple[str, str]]) -> None:
    d = cycle / "classify"
    d.mkdir(parents=True, exist_ok=True)
    (d / "labeled_incidents.json").write_text(
        json.dumps([
            {"incident_id": iid, "entry_id": eid, "confidence": 0.9,
             "stage": 1, "rationale": "x", "stratum": "security"}
            for iid, eid in rows
        ])
    )


def test_verify_passes_on_complete_cycle(tmp_path: Path) -> None:
    # The verifier itself is the unit under test here (the flip sites call it).
    from engine.calibrate.coverage import verify_labeled_completeness

    cycle = tmp_path / "cycle"
    _snapshot(cycle, "snap", ["INC-1", "INC-2", "INC-3"])
    _labeled(cycle, [("INC-1", "LLM01"), ("INC-2", "LLM02")])
    write_classify_coverage(
        cycle / "classify",
        snapshot_hash="snap",
        corpus_incident_ids={"INC-1", "INC-2", "INC-3"},
        in_scope_incident_ids={"INC-1", "INC-2"},
    )
    labeled = json.loads((cycle / "classify" / "labeled_incidents.json").read_text())
    labeled_ids = {str(r["incident_id"]) for r in labeled}
    # INC-3 is a genuine-OOS goldset recall incident -> must NOT raise.
    verify_labeled_completeness(cycle, "snap", labeled_ids, goldset_recall_ids={"INC-1", "INC-3"})


def test_infer_raises_on_truncated_labeled(tmp_path: Path) -> None:
    from engine.cli.pipeline_executor import execute_infer_phase

    cycle = tmp_path / "cycle"
    _snapshot(cycle, "snap", ["INC-1", "INC-2", "INC-3", "INC-4"])
    _labeled(cycle, [("INC-1", "LLM01"), ("INC-2", "LLM02")])  # only 2 rows
    # Marker claims 3 in-scope but the labeled file has 2 -> truncated.
    (cycle / "classify" / COVERAGE_FILENAME).write_text(
        json.dumps({"snapshot_hash": "snap", "n_corpus": 4, "n_in_scope": 3, "n_oos": 1})
    )
    (cycle / "calibration").mkdir(parents=True)
    (cycle / "calibration" / "posteriors.json").write_text(
        json.dumps({"recall": {}, "precision": {}})
    )
    (cycle / "prereg").mkdir(parents=True)
    (cycle / "prereg" / "manifest.json").write_text(json.dumps({
        "engine_version": "0.1.0", "engine_version_range_min": "0.1.0",
        "engine_version_range_max": "0.2.0", "cycle_id": "test-cycle-001",
        "taxonomy_hash": "aaa", "snapshot_hash": "snap",
        "primary_spec": "negative_binomial_per_stratum", "robustness_specs": [],
        "flag_threshold_tau": 0.8, "statistic": "weighted_cohens_kappa",
        "measurability_minimum": 10, "prior_scale": 0.5, "concentration_shape": 5.0,
        "concentration_rate": 0.1, "ess_fraction": 0.4, "meaningful_kappa_n": 4,
        "prng_seed": 42, "confidence_threshold": 0.3,
        "rubric_drafting_attestation": None, "rubric_reviewer": None,
        "statistical_reviewer": None, "classifier_rule_hash": None,
        "rubric_hash": None, "post_hoc_register_path": None,
    }))
    # R5: write a valid manifest.lock so _verify_manifest_lock passes and
    # execute_infer_phase reaches the completeness check (the actual test intent).
    from engine.cli.pipeline_executor import _load_manifest as _lm
    from engine.prereg.lock import write_lock as _wl
    _wl(_lm(cycle / "prereg" / "manifest.json"), cycle / "prereg" / "manifest.lock")
    with pytest.raises(LabeledIncidentsIncompleteError, match="truncated"):
        execute_infer_phase(cycle)
