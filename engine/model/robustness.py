"""Robustness-spec inference alternatives (HANDOFF §6 control 4).

Each robustness spec is a complete alternative model specification.
The primary is NegativeBinomial per stratum; robustness specs include
Poisson-flat (no over-dispersion, flat priors).
"""
from __future__ import annotations

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
from engine.model.inference import InferenceResult, _build_observation_arrays, _build_overlap_matrix
from engine.model.overlap import OverlapWeights
from engine.prereg.manifest import PreregManifest


def run_robustness_inference(
    manifest: PreregManifest,
    spec_name: str,
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
    if spec_name == "poisson_flat":
        return _run_poisson_flat(
            manifest, measurable_entries, strata, observed_counts,
            stratum_sizes, calibration, overlap, num_warmup, num_samples,
            num_chains,
        )
    raise ValueError(f"Unknown robustness spec: {spec_name}")


def _run_poisson_flat(
    manifest: PreregManifest,
    measurable_entries: tuple[str, ...],
    strata: tuple[str, ...],
    observed_counts: dict[tuple[str, str], int],
    stratum_sizes: dict[str, int],
    calibration: Calibration,
    overlap: OverlapWeights,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
) -> InferenceResult:
    assert jax.default_backend() == "cpu"

    n_entries = len(measurable_entries)
    obs, sizes, recall_a, recall_b, prec_a, prec_b = _build_observation_arrays(
        measurable_entries, strata, observed_counts, stratum_sizes, calibration,
    )
    W = _build_overlap_matrix(measurable_entries, overlap)

    def model(
        obs_data: npt.NDArray[np.float64],
        sizes_data: npt.NDArray[np.float64],
        recall_alpha: npt.NDArray[np.float64],
        recall_beta: npt.NDArray[np.float64],
        precision_alpha: npt.NDArray[np.float64],
        precision_beta: npt.NDArray[np.float64],
        W_data: npt.NDArray[np.float64],
    ) -> None:
        lam = numpyro.sample(
            "lambda", dist.HalfNormal(scale=jnp.ones(n_entries)),
        )
        recall = numpyro.sample(
            "recall", dist.Beta(jnp.array(recall_alpha), jnp.array(recall_beta)),
        )
        precision = numpyro.sample(
            "precision", dist.Beta(jnp.array(precision_alpha), jnp.array(precision_beta)),
        )
        true_rate = lam[:, None] * sizes_data[None, :]
        tp = true_rate * recall
        fp_rate = jnp.einsum("ij,js->is", jnp.array(W_data), true_rate * (1.0 - precision))
        expected = jnp.clip(tp + fp_rate, 1e-6, None)
        numpyro.sample("obs", dist.Poisson(rate=expected), obs=jnp.array(obs_data))

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(manifest.prng_seed + 1000), obs, sizes, recall_a, recall_b, prec_a, prec_b, W)

    samples = mcmc.get_samples()
    lambda_samples = np.asarray(samples["lambda"], dtype=np.float64)

    # Diagnostics extraction (mirroring run_inference)
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

    extra = mcmc.get_extra_fields()
    diverging = extra.get("diverging", np.array([]))
    divergences = int(np.asarray(diverging).sum())

    return InferenceResult(
        lambda_samples=lambda_samples,
        entry_ids=measurable_entries,
        r_hat=r_hat_dict,
        ess=ess_dict,
        divergences=divergences,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
