"""Prior and posterior predictive checks for the measurement-error model.

Prior predictive sampling verifies the model produces plausible data ranges
before seeing any observed data.  Posterior predictive sampling generates
replicated datasets from posterior samples for model checking.

See HANDOFF §5.4 for the full model specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from engine.model.inference import InferenceResult
from engine.prereg.manifest import PreregManifest


@dataclass(frozen=True, slots=True)
class PredictiveResult:
    """Predictive samples for model checking."""

    predicted_counts: npt.NDArray[np.float64]  # (num_samples, n_entries, n_strata)
    observed_counts: npt.NDArray[np.float64]  # (n_entries, n_strata) — the actual data


def prior_predictive(
    manifest: PreregManifest,
    n_entries: int,
    n_strata: int,
    stratum_sizes: npt.NDArray[np.float64],
    recall_alpha: npt.NDArray[np.float64],
    recall_beta: npt.NDArray[np.float64],
    precision_alpha: npt.NDArray[np.float64],
    precision_beta: npt.NDArray[np.float64],
    overlap_matrix: npt.NDArray[np.float64],
    num_samples: int = 500,
) -> PredictiveResult:
    """Sample from the prior predictive distribution.

    Used to verify the model produces plausible data ranges before seeing data.
    """
    prior_scale = manifest.prior_scale
    conc_shape = manifest.concentration_shape
    conc_rate = manifest.concentration_rate

    def model(
        sizes_data: jnp.ndarray,
        recall_a: jnp.ndarray,
        recall_b: jnp.ndarray,
        prec_a: jnp.ndarray,
        prec_b: jnp.ndarray,
        W_data: jnp.ndarray,
    ) -> None:
        lam = numpyro.sample(
            "lambda",
            dist.HalfNormal(scale=jnp.full(n_entries, prior_scale)),
        )
        recall = numpyro.sample(
            "recall",
            dist.Beta(jnp.array(recall_a), jnp.array(recall_b)),
        )
        precision = numpyro.sample(
            "precision",
            dist.Beta(jnp.array(prec_a), jnp.array(prec_b)),
        )
        concentration = numpyro.sample(
            "concentration",
            dist.Gamma(conc_shape, conc_rate),
        )
        true_rate = lam[:, None] * sizes_data[None, :]
        tp = true_rate * recall
        fp_rate = jnp.einsum(
            "ij,js->is",
            jnp.array(W_data),
            true_rate * (1.0 - precision),
        )
        expected = jnp.clip(tp + fp_rate, 1e-6, None)
        numpyro.sample(
            "obs",
            dist.NegativeBinomial2(mean=expected, concentration=concentration),
        )

    predictive = Predictive(model, num_samples=num_samples)
    rng = jax.random.PRNGKey(manifest.prng_seed)
    samples = predictive(
        rng,
        stratum_sizes,
        recall_alpha,
        recall_beta,
        precision_alpha,
        precision_beta,
        overlap_matrix,
    )
    return PredictiveResult(
        predicted_counts=np.asarray(samples["obs"], dtype=np.float64),
        observed_counts=np.zeros((n_entries, n_strata), dtype=np.float64),
    )


def posterior_predictive(
    manifest: PreregManifest,
    inference_result: InferenceResult,
    posterior_samples: dict[str, npt.NDArray[np.float64]],
    n_entries: int,
    n_strata: int,
    stratum_sizes: npt.NDArray[np.float64],
    recall_alpha: npt.NDArray[np.float64],
    recall_beta: npt.NDArray[np.float64],
    precision_alpha: npt.NDArray[np.float64],
    precision_beta: npt.NDArray[np.float64],
    overlap_matrix: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
) -> PredictiveResult:
    """Sample from the posterior predictive distribution.

    Uses posterior samples to generate replicated data for model checking.
    """
    prior_scale = manifest.prior_scale
    conc_shape = manifest.concentration_shape
    conc_rate = manifest.concentration_rate

    def model(
        sizes_data: jnp.ndarray,
        recall_a: jnp.ndarray,
        recall_b: jnp.ndarray,
        prec_a: jnp.ndarray,
        prec_b: jnp.ndarray,
        W_data: jnp.ndarray,
    ) -> None:
        lam = numpyro.sample(
            "lambda",
            dist.HalfNormal(scale=jnp.full(n_entries, prior_scale)),
        )
        recall = numpyro.sample(
            "recall",
            dist.Beta(jnp.array(recall_a), jnp.array(recall_b)),
        )
        precision = numpyro.sample(
            "precision",
            dist.Beta(jnp.array(prec_a), jnp.array(prec_b)),
        )
        concentration = numpyro.sample(
            "concentration",
            dist.Gamma(conc_shape, conc_rate),
        )
        true_rate = lam[:, None] * sizes_data[None, :]
        tp = true_rate * recall
        fp_rate = jnp.einsum(
            "ij,js->is",
            jnp.array(W_data),
            true_rate * (1.0 - precision),
        )
        expected = jnp.clip(tp + fp_rate, 1e-6, None)
        numpyro.sample(
            "obs",
            dist.NegativeBinomial2(mean=expected, concentration=concentration),
        )

    predictive = Predictive(model, posterior_samples=posterior_samples)
    rng = jax.random.PRNGKey(manifest.prng_seed + 1)
    samples = predictive(
        rng,
        stratum_sizes,
        recall_alpha,
        recall_beta,
        precision_alpha,
        precision_beta,
        overlap_matrix,
    )
    return PredictiveResult(
        predicted_counts=np.asarray(samples["obs"], dtype=np.float64),
        observed_counts=observed,
    )
