"""Unit tests for engine.model.predictive — prior and posterior predictive checks."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.inference import run_inference
from engine.model.overlap import OverlapWeights
from engine.model.predictive import (
    PredictiveResult,
    posterior_predictive,
    prior_predictive,
)
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
        "rubric_drafting_attestation": None,
        "rubric_reviewer": None,
        "statistical_reviewer": None,
        "classifier_rule_hash": None,
        "rubric_hash": None,
        "post_hoc_register_path": None,
    }
    defaults.update(overrides)
    return PreregManifest(**defaults)


def _small_arrays() -> (
    tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]
):
    """Return (stratum_sizes, recall_alpha, recall_beta, prec_alpha, prec_beta, W)
    for a 2-entry, 2-strata problem."""
    sizes = np.array([200.0, 150.0])
    recall_a = np.array([[18.0, 16.0], [14.0, 12.0]])
    recall_b = np.array([[2.0, 4.0], [3.0, 4.0]])
    prec_a = np.array([[19.0, 17.0], [15.0, 13.0]])
    prec_b = np.array([[1.0, 3.0], [2.0, 3.0]])
    W = np.zeros((2, 2))
    return sizes, recall_a, recall_b, prec_a, prec_b, W


# ---------------------------------------------------------------------------
# Prior predictive
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPriorPredictive:
    """Prior predictive check: model produces plausible data before seeing data."""

    def test_shape(self) -> None:
        manifest = _make_manifest()
        sizes, recall_a, recall_b, prec_a, prec_b, W = _small_arrays()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            result = prior_predictive(
                manifest=manifest,
                n_entries=2,
                n_strata=2,
                stratum_sizes=sizes,
                recall_alpha=recall_a,
                recall_beta=recall_b,
                precision_alpha=prec_a,
                precision_beta=prec_b,
                overlap_matrix=W,
                num_samples=NUM_SAMPLES,
            )

        assert isinstance(result, PredictiveResult)
        assert result.predicted_counts.shape == (NUM_SAMPLES, 2, 2)
        assert result.observed_counts.shape == (2, 2)

    def test_non_negative(self) -> None:
        manifest = _make_manifest()
        sizes, recall_a, recall_b, prec_a, prec_b, W = _small_arrays()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            result = prior_predictive(
                manifest=manifest,
                n_entries=2,
                n_strata=2,
                stratum_sizes=sizes,
                recall_alpha=recall_a,
                recall_beta=recall_b,
                precision_alpha=prec_a,
                precision_beta=prec_b,
                overlap_matrix=W,
                num_samples=NUM_SAMPLES,
            )

        assert np.all(result.predicted_counts >= 0)

    def test_plausible_range(self) -> None:
        """With reasonable calibration, most counts should be plausible:
        not all zeros and not all millions."""
        manifest = _make_manifest()
        sizes, recall_a, recall_b, prec_a, prec_b, W = _small_arrays()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            result = prior_predictive(
                manifest=manifest,
                n_entries=2,
                n_strata=2,
                stratum_sizes=sizes,
                recall_alpha=recall_a,
                recall_beta=recall_b,
                precision_alpha=prec_a,
                precision_beta=prec_b,
                overlap_matrix=W,
                num_samples=NUM_SAMPLES,
            )

        counts = result.predicted_counts
        # At least some samples should be non-zero
        assert counts.sum() > 0, "All prior predictive counts are zero"
        # Median across all samples and cells should be below stratum sizes
        median_count = np.median(counts)
        max_stratum = sizes.max()
        assert median_count < max_stratum * 10, (
            f"Median prior predictive count {median_count:.0f} is implausibly "
            f"large relative to stratum size {max_stratum:.0f}"
        )

    def test_observed_counts_are_zeros(self) -> None:
        """Prior predictive has no observed data — observed_counts should be zeros."""
        manifest = _make_manifest()
        sizes, recall_a, recall_b, prec_a, prec_b, W = _small_arrays()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            result = prior_predictive(
                manifest=manifest,
                n_entries=2,
                n_strata=2,
                stratum_sizes=sizes,
                recall_alpha=recall_a,
                recall_beta=recall_b,
                precision_alpha=prec_a,
                precision_beta=prec_b,
                overlap_matrix=W,
                num_samples=NUM_SAMPLES,
            )

        assert np.all(result.observed_counts == 0)


# ---------------------------------------------------------------------------
# Posterior predictive
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPosteriorPredictive:
    """Posterior predictive check: replicated data from posterior samples."""

    def test_shape_and_observed(self) -> None:
        manifest = _make_manifest()
        entries = ("E01", "E02")
        strata = ("s1", "s2")
        observed_counts: dict[tuple[str, str], int] = {
            ("E01", "s1"): 30,
            ("E01", "s2"): 20,
            ("E02", "s1"): 10,
            ("E02", "s2"): 8,
        }
        stratum_sizes_dict = {"s1": 200, "s2": 150}
        calibration = Calibration(
            recall={
                ("E01", "s1"): BetaPosterior(alpha=18.0, beta=2.0),
                ("E01", "s2"): BetaPosterior(alpha=16.0, beta=4.0),
                ("E02", "s1"): BetaPosterior(alpha=14.0, beta=3.0),
                ("E02", "s2"): BetaPosterior(alpha=12.0, beta=4.0),
            },
            precision={
                ("E01", "s1"): BetaPosterior(alpha=19.0, beta=1.0),
                ("E01", "s2"): BetaPosterior(alpha=17.0, beta=3.0),
                ("E02", "s1"): BetaPosterior(alpha=15.0, beta=2.0),
                ("E02", "s2"): BetaPosterior(alpha=13.0, beta=3.0),
            },
        )
        overlap = OverlapWeights(weights={})

        # First, run inference to get posterior samples
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            inference_result = run_inference(
                manifest=manifest,
                measurable_entries=entries,
                strata=strata,
                observed_counts=observed_counts,
                stratum_sizes=stratum_sizes_dict,
                calibration=calibration,
                overlap=overlap,
                num_warmup=NUM_WARMUP,
                num_samples=NUM_SAMPLES,
            )

        # Build arrays matching the model signature
        sizes = np.array([200.0, 150.0])
        recall_a = np.array([[18.0, 16.0], [14.0, 12.0]])
        recall_b = np.array([[2.0, 4.0], [3.0, 4.0]])
        prec_a = np.array([[19.0, 17.0], [15.0, 13.0]])
        prec_b = np.array([[1.0, 3.0], [2.0, 3.0]])
        W = np.zeros((2, 2))
        obs = np.array([[30.0, 20.0], [10.0, 8.0]])

        # Build posterior_samples dict from MCMC — we need all latent variables.
        # run_inference only exposes lambda_samples, so we re-run MCMC
        # to get the full samples dict.  For this test we construct synthetic
        # posterior samples that are plausible.
        n_post = inference_result.lambda_samples.shape[0]
        post_samples: dict[str, npt.NDArray[np.float64]] = {
            "lambda": inference_result.lambda_samples,
            # Use point-mass-like values for recall/precision/concentration
            # so the posterior predictive is tractable in a unit test.
            "recall": np.tile(recall_a / (recall_a + recall_b), (n_post, 1, 1)),
            "precision": np.tile(
                prec_a / (prec_a + prec_b), (n_post, 1, 1)
            ),
            "concentration": np.full(n_post, 50.0),
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            result = posterior_predictive(
                manifest=manifest,
                inference_result=inference_result,
                posterior_samples=post_samples,
                n_entries=2,
                n_strata=2,
                stratum_sizes=sizes,
                recall_alpha=recall_a,
                recall_beta=recall_b,
                precision_alpha=prec_a,
                precision_beta=prec_b,
                overlap_matrix=W,
                observed=obs,
            )

        assert isinstance(result, PredictiveResult)
        assert result.predicted_counts.shape == (n_post, 2, 2)
        assert np.array_equal(result.observed_counts, obs)
        assert np.all(result.predicted_counts >= 0)
