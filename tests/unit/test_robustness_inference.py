# tests/unit/test_robustness_inference.py
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.overlap import OverlapWeights
from engine.model.robustness import run_robustness_inference
from engine.prereg.manifest import PreregManifest


def _make_manifest() -> PreregManifest:
    return PreregManifest(
        engine_version="1.0.0",
        engine_version_range_min="1.0.0",
        engine_version_range_max="1.0.0",
        cycle_id="test",
        taxonomy_hash="testhash",
        snapshot_hash="snaphash",
        primary_spec="negative_binomial_per_stratum",
        robustness_specs=("poisson_flat",),
        flag_threshold_tau=0.8,
        statistic="weighted_cohens_kappa",
        measurability_minimum=4,
        prior_scale=0.5,
        concentration_shape=5.0,
        concentration_rate=0.1,
        ess_fraction=0.4,
        meaningful_kappa_n=4,
        prng_seed=42,
        confidence_threshold=0.3,
        rubric_drafting_attestation=None,
        rubric_reviewer=None,
        statistical_reviewer=None,
        classifier_rule_hash=None,
        rubric_hash=None,
        post_hoc_register_path=None,
    )


class TestRobustnessInference:
    @pytest.mark.timeout(120)
    def test_poisson_flat_returns_inference_result(self) -> None:
        manifest = _make_manifest()
        entries = ("E1", "E2")
        strata = ("security",)
        counts = {("E1", "security"): 50, ("E2", "security"): 30}
        sizes = {"security": 1000}
        uniform = BetaPosterior(alpha=1.0, beta=1.0)
        cal = Calibration(
            recall={(e, s): uniform for e in entries for s in strata},
            precision={(e, s): uniform for e in entries for s in strata},
        )
        overlap = OverlapWeights(weights={})
        result = run_robustness_inference(
            manifest=manifest,
            spec_name="poisson_flat",
            measurable_entries=entries,
            strata=strata,
            observed_counts=counts,
            stratum_sizes=sizes,
            calibration=cal,
            overlap=overlap,
            num_warmup=100,
            num_samples=200,
        )
        assert result.lambda_samples.shape == (200, 2)
        assert result.entry_ids == entries
