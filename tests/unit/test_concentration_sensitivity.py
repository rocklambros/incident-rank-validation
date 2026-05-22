"""M19: posterior lambda median should be stable to concentration prior choice.

The NB concentration prior was changed from Exponential(1) to Gamma(5, 0.1)
in v2.2.  This test verifies the posterior lambda median is robust (within 10%)
to reasonable variation of the concentration prior hyperparameters.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.inference import run_inference
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


def _run_inference_with_manifest(manifest: PreregManifest) -> npt.NDArray[np.float64]:
    """Run inference on a small 2-entry, 2-strata problem, return lambda_samples."""
    entries = ("E01", "E02")
    strata = ("s1", "s2")
    observed: dict[tuple[str, str], int] = {
        ("E01", "s1"): 25,
        ("E01", "s2"): 18,
        ("E02", "s1"): 12,
        ("E02", "s2"): 8,
    }
    stratum_sizes = {"s1": 200, "s2": 150}
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
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            num_chains=1,
        )
    return result.lambda_samples


# ---------------------------------------------------------------------------
# M19: concentration sensitivity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_concentration_prior_sensitivity_within_10pct() -> None:
    """Posterior lambda median should be stable (within 10%) across
    reasonable concentration prior choices."""
    # Three concentration priors with different shape/rate combos.
    # All have relatively diffuse priors on concentration.
    configs: list[tuple[float, float]] = [
        (5.0, 0.1),   # baseline — Gamma(5, 0.1), mean=50
        (2.0, 0.05),  # variant 1 — Gamma(2, 0.05), mean=40
        (10.0, 0.2),  # variant 2 — Gamma(10, 0.2), mean=50
    ]

    medians: list[npt.NDArray[np.float64]] = []
    for shape, rate in configs:
        manifest = _make_manifest(
            concentration_shape=shape,
            concentration_rate=rate,
        )
        samples = _run_inference_with_manifest(manifest)
        medians.append(np.median(samples, axis=0))

    baseline = medians[0]
    for i, (shape, rate) in enumerate(configs[1:], start=1):
        # Per-entry relative deviation from baseline
        relative_diff = np.abs(medians[i] - baseline) / np.maximum(baseline, 1e-8)
        max_diff = float(relative_diff.max())
        assert max_diff < 0.10, (
            f"Concentration prior Gamma({shape}, {rate}) caused lambda median "
            f"to shift {max_diff:.1%} from baseline — exceeds 10% threshold. "
            f"Baseline medians: {baseline}, variant medians: {medians[i]}"
        )
