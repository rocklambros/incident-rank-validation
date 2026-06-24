"""TDD test for the hierarchical-pooling robustness spec (Plan 8b Task 3)."""
import numpy as np

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.overlap import OverlapWeights
from engine.model.robustness import run_robustness_inference
from tests.unit.test_prereg import _make_manifest


def _tiny_calibration(entries: tuple[str, ...], stratum: str) -> Calibration:
    recall = {(e, stratum): BetaPosterior(8.0, 2.0) for e in entries}
    precision = {(e, stratum): BetaPosterior(9.0, 1.0) for e in entries}
    return Calibration(recall=recall, precision=precision)


def test_hierarchical_returns_sigma_u_and_lambda_shape() -> None:
    entries = ("LLM01", "LLM02", "LLM03", "LLM04")
    stratum = "security"
    manifest = _make_manifest(schema_version=2, sigma_u_hyperprior_scale=1.0)
    observed = {(e, stratum): n for e, n in zip(entries, [50, 30, 10, 1], strict=True)}
    result = run_robustness_inference(
        manifest=manifest, spec_name="hierarchical_pooling",
        measurable_entries=entries, strata=(stratum,),
        observed_counts=observed, stratum_sizes={stratum: 1000},
        calibration=_tiny_calibration(entries, stratum),
        overlap=OverlapWeights(weights={}),
        num_warmup=200, num_samples=200, num_chains=2,
    )
    # lambda contract preserved: (num_samples*num_chains, n_entries)
    assert result.lambda_samples.shape[1] == len(entries)
    # sigma_u captured as a positive scalar
    assert result.sigma_u is not None and result.sigma_u > 0.0
    assert np.all(result.lambda_samples > 0.0)  # exp(...) is strictly positive


def test_unknown_spec_still_raises() -> None:
    import pytest
    manifest = _make_manifest()
    with pytest.raises(ValueError, match="Unknown robustness spec"):
        run_robustness_inference(
            manifest=manifest, spec_name="nonexistent",
            measurable_entries=("A",), strata=("s",),
            observed_counts={("A", "s"): 1}, stratum_sizes={"s": 10},
            calibration=Calibration(recall={}, precision={}),
            overlap=OverlapWeights(weights={}),
        )
