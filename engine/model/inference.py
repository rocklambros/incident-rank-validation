"""NUTS inference engine for latent incident intensity.

See HANDOFF §5.4: posterior over each measurable entry's latent prevalence,
accounting for classifier error, contamination, and the mixture structure.

All tuneable hyperparameters come from the PreregManifest — no module-level
constants.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import numpyro
import numpyro.diagnostics
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from engine.calibrate.beta import Calibration
from engine.model.overlap import OverlapWeights
from engine.prereg.manifest import PreregManifest


class DiagnosticsFailure(RuntimeError):
    """NUTS diagnostics failed — report not emitted."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    lambda_samples: npt.NDArray[np.float64]  # shape (num_samples, num_entries)
    entry_ids: tuple[str, ...]  # which entries, in column order
    r_hat: dict[str, float]  # per-parameter R-hat
    ess: dict[str, float]  # per-parameter effective sample size
    divergences: int
    num_warmup: int
    num_samples: int


def _build_observation_arrays(
    measurable_entries: tuple[str, ...],
    strata: tuple[str, ...],
    observed_counts: dict[tuple[str, str], int],
    stratum_sizes: dict[str, int],
    calibration: Calibration,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Build matrix-form arrays from dicts for the NUTS model.

    Returns (obs, sizes, recall_alpha, recall_beta, precision_alpha, precision_beta).
    """
    n_entries = len(measurable_entries)
    n_strata = len(strata)

    obs = np.zeros((n_entries, n_strata), dtype=np.float64)
    recall_alpha = np.ones((n_entries, n_strata), dtype=np.float64)
    recall_beta = np.ones((n_entries, n_strata), dtype=np.float64)
    precision_alpha = np.ones((n_entries, n_strata), dtype=np.float64)
    precision_beta = np.ones((n_entries, n_strata), dtype=np.float64)
    sizes = np.array([stratum_sizes[s] for s in strata], dtype=np.float64)

    for i, eid in enumerate(measurable_entries):
        for j, s in enumerate(strata):
            obs[i, j] = observed_counts.get((eid, s), 0)
            key = (eid, s)
            if key in calibration.recall:
                recall_alpha[i, j] = calibration.recall[key].alpha
                recall_beta[i, j] = calibration.recall[key].beta
            if key in calibration.precision:
                precision_alpha[i, j] = calibration.precision[key].alpha
                precision_beta[i, j] = calibration.precision[key].beta

    return obs, sizes, recall_alpha, recall_beta, precision_alpha, precision_beta


def _build_overlap_matrix(
    measurable_entries: tuple[str, ...],
    overlap: OverlapWeights,
) -> npt.NDArray[np.float64]:
    """Build W[i, j] = fraction of entry j's FPs landing in entry i."""
    n = len(measurable_entries)
    W = np.zeros((n, n), dtype=np.float64)
    for i, target in enumerate(measurable_entries):
        if target in overlap.weights:
            for j, source in enumerate(measurable_entries):
                if source in overlap.weights[target]:
                    W[i, j] = overlap.weights[target][source]
    return W


def run_inference(
    manifest: PreregManifest,
    measurable_entries: tuple[str, ...],
    strata: tuple[str, ...],
    observed_counts: dict[tuple[str, str], int],
    stratum_sizes: dict[str, int],
    calibration: Calibration,
    overlap: OverlapWeights,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    timeout_seconds: float | None = None,
) -> InferenceResult:
    """Run NUTS inference on the measurement-error model.

    All hyperparameters come from *manifest* (prior_scale, concentration_shape,
    concentration_rate, prng_seed).  No module-level constants for tunables.
    """
    # Verify CPU backend for reproducibility
    assert jax.default_backend() == "cpu", (
        f"JAX backend is {jax.default_backend()}, expected 'cpu'. "
        "Set JAX_PLATFORM_NAME=cpu for reproducibility."
    )

    n_entries = len(measurable_entries)

    obs, sizes, recall_a, recall_b, prec_a, prec_b = _build_observation_arrays(
        measurable_entries, strata, observed_counts, stratum_sizes, calibration,
    )
    W = _build_overlap_matrix(measurable_entries, overlap)

    prior_scale = manifest.prior_scale
    conc_shape = manifest.concentration_shape
    conc_rate = manifest.concentration_rate
    ess_fraction = manifest.ess_fraction

    # ------------------------------------------------------------------
    # NumPyro model
    # ------------------------------------------------------------------
    def model(
        obs_data: npt.NDArray[np.float64],
        sizes_data: npt.NDArray[np.float64],
        recall_alpha: npt.NDArray[np.float64],
        recall_beta: npt.NDArray[np.float64],
        precision_alpha: npt.NDArray[np.float64],
        precision_beta: npt.NDArray[np.float64],
        W_data: npt.NDArray[np.float64],
    ) -> None:
        # Latent prevalence per entry
        lam = numpyro.sample(
            "lambda",
            dist.HalfNormal(scale=jnp.full(n_entries, prior_scale)),
        )

        # Per-entry, per-stratum recall and precision
        recall = numpyro.sample(
            "recall",
            dist.Beta(jnp.array(recall_alpha), jnp.array(recall_beta)),
        )
        precision = numpyro.sample(
            "precision",
            dist.Beta(jnp.array(precision_alpha), jnp.array(precision_beta)),
        )

        # Negative-Binomial concentration (over-dispersion)
        concentration = numpyro.sample(
            "concentration",
            dist.Gamma(conc_shape, conc_rate),
        )

        # Expected true counts: lambda_e * stratum_size_s
        true_rate = lam[:, None] * sizes_data[None, :]  # (n_entries, n_strata)

        # True positives: true_rate * recall
        tp = true_rate * recall

        # False positives from leakage: FP_i = sum_j W[i,j] * true_rate_j * (1 - precision_j)
        fp_rate = jnp.einsum(
            "ij,js->is",
            jnp.array(W_data),
            true_rate * (1.0 - precision),
        )

        expected = jnp.clip(tp + fp_rate, 1e-6, None)

        # Negative-Binomial likelihood
        numpyro.sample(
            "obs",
            dist.NegativeBinomial2(mean=expected, concentration=concentration),
            obs=jnp.array(obs_data),
        )

    # ------------------------------------------------------------------
    # Timeout guard
    # ------------------------------------------------------------------
    old_handler: Any = signal.SIG_DFL
    if timeout_seconds is not None:

        def _timeout_handler(signum: int, frame: object) -> None:
            raise TimeoutError(
                f"NUTS inference timed out after {timeout_seconds}s"
            )

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_seconds))

    try:
        kernel = NUTS(model)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )
        mcmc.run(
            jax.random.PRNGKey(manifest.prng_seed),
            obs, sizes, recall_a, recall_b, prec_a, prec_b, W,
        )
    finally:
        if timeout_seconds is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    samples = mcmc.get_samples()
    lambda_samples: npt.NDArray[np.float64] = np.asarray(
        samples["lambda"], dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Diagnostics extraction
    # ------------------------------------------------------------------
    chain_samples = mcmc.get_samples(group_by_chain=True)
    summary: dict[str, Any] = numpyro.diagnostics.summary(chain_samples)

    r_hat_dict: dict[str, float] = {}
    ess_dict: dict[str, float] = {}
    for param_name, stats in summary.items():
        if "r_hat" in stats:
            vals = np.atleast_1d(stats["r_hat"])
            for idx, val in enumerate(vals.flat):
                key = f"{param_name}[{idx}]" if vals.size > 1 else param_name
                r_hat_dict[key] = float(val)
        if "n_eff" in stats:
            vals = np.atleast_1d(stats["n_eff"])
            for idx, val in enumerate(vals.flat):
                key = f"{param_name}[{idx}]" if vals.size > 1 else param_name
                ess_dict[key] = float(val)

    # Divergences
    extra = mcmc.get_extra_fields()
    diverging = extra.get("diverging", np.array([]))
    divergences = int(np.asarray(diverging).sum())

    # ------------------------------------------------------------------
    # Diagnostic gates (HANDOFF §5.4)
    # ------------------------------------------------------------------
    # R-hat <= 1.01
    max_rhat = max(r_hat_dict.values()) if r_hat_dict else 1.0
    if max_rhat > 1.01:
        raise DiagnosticsFailure(
            f"R-hat exceeded threshold: max R-hat = {max_rhat:.4f} > 1.01"
        )

    # Zero post-warmup divergences
    if divergences > 0:
        raise DiagnosticsFailure(
            f"Post-warmup divergences detected: {divergences}"
        )

    # Sufficient ESS
    min_ess_fraction = (
        min(v / num_samples for v in ess_dict.values()) if ess_dict else 1.0
    )
    if min_ess_fraction < ess_fraction:
        raise DiagnosticsFailure(
            f"ESS below threshold: min ESS fraction = {min_ess_fraction:.4f} "
            f"< {ess_fraction}"
        )

    return InferenceResult(
        lambda_samples=lambda_samples,
        entry_ids=measurable_entries,
        r_hat=r_hat_dict,
        ess=ess_dict,
        divergences=divergences,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
