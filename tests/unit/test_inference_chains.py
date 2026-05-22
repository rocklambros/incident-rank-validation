from __future__ import annotations

import inspect


def test_run_inference_accepts_num_chains() -> None:
    """F8: run_inference must accept and pass through num_chains."""
    from engine.model.inference import run_inference

    sig = inspect.signature(run_inference)
    assert "num_chains" in sig.parameters, "run_inference must accept num_chains parameter"
    assert sig.parameters["num_chains"].default == 4


def test_robustness_inference_uses_num_chains() -> None:
    """F9: robustness inference must also accept and pass through num_chains."""
    from engine.model.robustness import run_robustness_inference

    sig = inspect.signature(run_robustness_inference)
    assert "num_chains" in sig.parameters
