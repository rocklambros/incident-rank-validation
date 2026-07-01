"""U2-5 (J2/J3): End-to-end robustness divergence-gate forcing test (monkeypatched).

Strategy: both run_inference (primary) and run_robustness_inference are monkeypatched
so no NUTS sampler runs — deterministic fast completion with provable termination.
Primary returns a valid InferenceResult (gate passes); robustness spec raises
DiagnosticsFailure("Post-warmup divergences detected: 1"). The pipeline writes
robustness_<spec>_failure.txt and re-raises; diagnostics_failure.txt (primary
failure marker) is NOT written.

Marked @pytest.mark.slow (NUTS-class gate) and ubuntu-only: the structural
invariant tested is OS-independent but excluded from the macOS leg to avoid
cross-platform duplication. Termination is guaranteed by monkeypatching — no
real NUTS sampler runs; the test completes in milliseconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="ubuntu-only e2e robustness divergence-gate forcing test",
)
def test_robustness_divergence_gate_e2e_monkeypatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2E: monkeypatched robustness spec divergence writes failure artifact; primary unaffected.

    J2: run_robustness_inference is monkeypatched to raise
        DiagnosticsFailure("Post-warmup divergences detected: 1"); run_inference
        (primary) is monkeypatched to return a valid InferenceResult.

    J3a: robustness_poisson_flat_failure.txt is written and contains "divergen".
    J3b: diagnostics_failure.txt is NOT written (primary result unaffected).

    Termination guarantee: both samplers are mocked; no real NUTS call is made,
    so this test cannot hang regardless of MCMC parameters.
    """
    from engine.model.inference import DiagnosticsFailure, InferenceResult

    # ---- Minimal cycle directory ------------------------------------------------
    cycle = tmp_path / "cycle"

    cal_dir = cycle / "calibration"
    cal_dir.mkdir(parents=True)
    (cal_dir / "posteriors.json").write_text(
        json.dumps({"recall": {}, "precision": {}}) + "\n"
    )

    cls_dir = cycle / "classify"
    cls_dir.mkdir(parents=True)
    labeled_data = [
        {
            "incident_id": "INC-001",
            "entry_id": "E01",
            "stratum": "all",
            "confidence": 0.9,
            "stage": 1,
            "rationale": "test",
        },
    ]
    (cls_dir / "labeled_incidents.json").write_text(json.dumps(labeled_data) + "\n")

    prereg_dir = cycle / "prereg"
    prereg_dir.mkdir(parents=True)
    manifest_dict: dict[str, Any] = {
        "engine_version": "0.1.0",
        "engine_version_range_min": "0.1.0",
        "engine_version_range_max": "0.2.0",
        "cycle_id": "test-robustness-diverge",
        "taxonomy_hash": "ttt",
        "snapshot_hash": "sss",
        "primary_spec": "negative_binomial_per_stratum",
        "robustness_specs": ["poisson_flat"],   # one robustness spec declared
        "flag_threshold_tau": 0.8,
        "statistic": "weighted_cohens_kappa",
        "measurability_minimum": 1,
        "prior_scale": 0.5,
        "concentration_shape": 5.0,
        "concentration_rate": 0.1,
        "ess_fraction": 0.1,
        "meaningful_kappa_n": 1,
        "prng_seed": 42,
        "confidence_threshold": 0.3,
        "rubric_drafting_attestation": None,
        "rubric_reviewer": None,
        "statistical_reviewer": None,
        "classifier_rule_hash": None,
        "rubric_hash": None,
        "post_hoc_register_path": None,
    }
    (prereg_dir / "manifest.json").write_text(json.dumps(manifest_dict) + "\n")

    # ---- Monkeypatches ----------------------------------------------------------
    # Primary: returns a valid InferenceResult — no divergences, diagnostics pass.
    fake_primary = InferenceResult(
        lambda_samples=np.array([[0.1], [0.1]]),
        entry_ids=("E01",),
        r_hat={"lambda[0]": 1.001},
        ess={"lambda[0]": 800.0},
        divergences=0,
        num_warmup=10,
        num_samples=20,
    )

    def _fake_run_inference(*args: Any, **kwargs: Any) -> InferenceResult:
        return fake_primary

    # Robustness spec: raises DiagnosticsFailure as if NUTS found divergences>0.
    def _fake_robustness(*args: Any, **kwargs: Any) -> InferenceResult:
        raise DiagnosticsFailure("Post-warmup divergences detected: 1")

    # Bypass snapshot-completeness guard (this test targets the robustness gate).
    def _noop_completeness(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr("engine.model.inference.run_inference", _fake_run_inference)
    monkeypatch.setattr(
        "engine.model.robustness.run_robustness_inference", _fake_robustness
    )
    monkeypatch.setattr(
        "engine.calibrate.coverage.verify_labeled_completeness",
        _noop_completeness,
    )

    # ---- Execute and assert -----------------------------------------------------
    from engine.cli.pipeline_executor import execute_infer_phase

    with pytest.raises(DiagnosticsFailure, match="divergen"):
        execute_infer_phase(cycle)

    infer_dir = cycle / "infer"

    # J3a: robustness failure artifact written and names a divergence.
    failure_path = infer_dir / "robustness_poisson_flat_failure.txt"
    assert failure_path.exists(), (
        "robustness_poisson_flat_failure.txt was not written after DiagnosticsFailure "
        "from the robustness spec"
    )
    failure_text = failure_path.read_text()
    assert "divergen" in failure_text, (
        f"failure artifact must mention divergence; got: {failure_text!r}"
    )

    # J3b: primary diagnostics_failure.txt must NOT exist (primary passed).
    assert not (infer_dir / "diagnostics_failure.txt").exists(), (
        "diagnostics_failure.txt must NOT be written when only the robustness spec "
        "fails — it would falsely imply the primary run failed"
    )
