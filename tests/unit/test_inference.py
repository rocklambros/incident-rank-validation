"""Unit tests for engine.model.inference — NUTS inference engine."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import numpy as np
import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.inference import InferenceResult, run_inference
from engine.model.overlap import OverlapWeights
from engine.prereg.manifest import PreregManifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_WARMUP = 200
NUM_SAMPLES = 500


def _make_manifest(**overrides: Any) -> PreregManifest:
    defaults: dict[str, Any] = {
        "engine_version": "0.1.0",
        "engine_version_range_min": "0.1.0",
        "engine_version_range_max": "0.2.0",
        "cycle_id": "test-cycle-001",
        "taxonomy_hash": "aaa",
        "snapshot_hash": "bbb",
        "primary_spec": "negative_binomial_per_stratum",
        "robustness_specs": ("poisson_flat",),
        "flag_threshold_tau": 0.8,
        "statistic": "weighted_cohens_kappa",
        "measurability_minimum": 10,
        "prior_scale": 0.5,
        "concentration_shape": 5.0,
        "concentration_rate": 0.1,
        "ess_fraction": 0.1,  # relaxed for small test runs
        "meaningful_kappa_n": 4,
        "prng_seed": 42,
        "confidence_threshold": 0.3,
        "rubric_drafting_attestation": None,
        "rubric_reviewer": None,
        "statistical_reviewer": None,
        "classifier_rule_hash": None,
        "rubric_hash": None,
        "post_hoc_register_path": None,
    }
    defaults.update(overrides)
    return PreregManifest(**defaults)


# ---------------------------------------------------------------------------
# JAX backend check
# ---------------------------------------------------------------------------


class TestJAXBackend:
    def test_cpu_backend(self) -> None:
        assert jax.default_backend() == "cpu"


# ---------------------------------------------------------------------------
# Basic inference
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBasicInference:
    """Run NUTS on a small synthetic problem and verify outputs."""

    def _run_small(
        self,
        manifest: PreregManifest | None = None,
        overlap: OverlapWeights | None = None,
    ) -> InferenceResult:
        if manifest is None:
            manifest = _make_manifest()
        entries = ("E01", "E02")
        strata = ("all",)
        observed: dict[tuple[str, str], int] = {
            ("E01", "all"): 30,
            ("E02", "all"): 10,
        }
        stratum_sizes = {"all": 200}
        calibration = Calibration(
            recall={
                ("E01", "all"): BetaPosterior(alpha=18.0, beta=2.0),  # ~0.9
                ("E02", "all"): BetaPosterior(alpha=16.0, beta=4.0),  # ~0.8
            },
            precision={
                ("E01", "all"): BetaPosterior(alpha=19.0, beta=1.0),  # ~0.95
                ("E02", "all"): BetaPosterior(alpha=17.0, beta=3.0),  # ~0.85
            },
        )
        if overlap is None:
            overlap = OverlapWeights(weights={})

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return run_inference(
                manifest=manifest,
                measurable_entries=entries,
                strata=strata,
                observed_counts=observed,
                stratum_sizes=stratum_sizes,
                calibration=calibration,
                overlap=overlap,
                num_warmup=NUM_WARMUP,
                num_samples=NUM_SAMPLES,
                num_chains=1,
            )

    def test_returns_inference_result(self) -> None:
        result = self._run_small()
        assert isinstance(result, InferenceResult)

    def test_lambda_samples_shape(self) -> None:
        result = self._run_small()
        assert result.lambda_samples.shape == (NUM_SAMPLES, 2)

    def test_entry_ids_match(self) -> None:
        result = self._run_small()
        assert result.entry_ids == ("E01", "E02")

    def test_no_diagnostics_failure(self) -> None:
        # If we get here without exception, diagnostics passed
        result = self._run_small()
        assert result.divergences == 0

    def test_lambda_positive(self) -> None:
        result = self._run_small()
        assert np.all(result.lambda_samples >= 0)

    def test_num_warmup_and_samples_recorded(self) -> None:
        result = self._run_small()
        assert result.num_warmup == NUM_WARMUP
        assert result.num_samples == NUM_SAMPLES


# ---------------------------------------------------------------------------
# Manifest hyperparameters affect posterior
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestManifestHyperparameters:
    """Verify that changing manifest.prior_scale changes the posterior."""

    def test_prior_scale_affects_posterior(self) -> None:
        entries = ("E01",)
        strata = ("all",)
        observed: dict[tuple[str, str], int] = {("E01", "all"): 5}
        stratum_sizes = {"all": 100}
        calibration = Calibration(
            recall={("E01", "all"): BetaPosterior(alpha=10.0, beta=2.0)},
            precision={("E01", "all"): BetaPosterior(alpha=10.0, beta=2.0)},
        )
        overlap = OverlapWeights(weights={})

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Tight prior (small scale)
            manifest_tight = _make_manifest(prior_scale=0.05)
            result_tight = run_inference(
                manifest=manifest_tight,
                measurable_entries=entries,
                strata=strata,
                observed_counts=observed,
                stratum_sizes=stratum_sizes,
                calibration=calibration,
                overlap=overlap,
                num_warmup=NUM_WARMUP,
                num_samples=NUM_SAMPLES,
                num_chains=1,
            )

            # Diffuse prior (large scale)
            manifest_diffuse = _make_manifest(prior_scale=5.0, prng_seed=99)
            result_diffuse = run_inference(
                manifest=manifest_diffuse,
                measurable_entries=entries,
                strata=strata,
                observed_counts=observed,
                stratum_sizes=stratum_sizes,
                calibration=calibration,
                overlap=overlap,
                num_warmup=NUM_WARMUP,
                num_samples=NUM_SAMPLES,
                num_chains=1,
            )

        mean_tight = float(result_tight.lambda_samples.mean())
        mean_diffuse = float(result_diffuse.lambda_samples.mean())

        # Different priors should yield different posterior means
        # (tight prior pulls lambda toward 0 more than diffuse)
        assert mean_tight != pytest.approx(mean_diffuse, abs=0.005), (
            f"Expected different posteriors: tight={mean_tight:.4f}, "
            f"diffuse={mean_diffuse:.4f}"
        )


# ---------------------------------------------------------------------------
# Empty overlap weights
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEmptyOverlap:
    """Verify inference works with empty OverlapWeights."""

    def test_runs_with_empty_overlap(self) -> None:
        manifest = _make_manifest()
        entries = ("E01", "E02")
        strata = ("s1",)
        observed: dict[tuple[str, str], int] = {
            ("E01", "s1"): 20,
            ("E02", "s1"): 8,
        }
        stratum_sizes = {"s1": 150}
        calibration = Calibration(
            recall={
                ("E01", "s1"): BetaPosterior(alpha=15.0, beta=3.0),
                ("E02", "s1"): BetaPosterior(alpha=12.0, beta=4.0),
            },
            precision={
                ("E01", "s1"): BetaPosterior(alpha=18.0, beta=2.0),
                ("E02", "s1"): BetaPosterior(alpha=14.0, beta=3.0),
            },
        )
        overlap = OverlapWeights(weights={})

        # Use slightly more warmup to ensure convergence with this data
        warmup = 400
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            result = run_inference(
                manifest=manifest,
                measurable_entries=entries,
                strata=strata,
                observed_counts=observed,
                stratum_sizes=stratum_sizes,
                calibration=calibration,
                overlap=overlap,
                num_warmup=warmup,
                num_samples=NUM_SAMPLES,
                num_chains=1,
            )

        assert isinstance(result, InferenceResult)
        assert result.lambda_samples.shape == (NUM_SAMPLES, 2)
        assert result.divergences == 0
