"""Test that ESS gate uses total draws (num_samples * num_chains) as denominator."""
from __future__ import annotations

import pytest

from engine.model.inference import DiagnosticsFailure


def test_ess_gate_denominator_uses_total_draws() -> None:
    num_samples = 2000
    num_chains = 4
    ess_value = 2400.0
    ess_fraction_threshold = 0.4

    total_draws = num_samples * num_chains
    correct_ratio = ess_value / total_draws
    assert correct_ratio == pytest.approx(0.3)
    assert correct_ratio < ess_fraction_threshold

    buggy_ratio = ess_value / num_samples
    assert buggy_ratio == pytest.approx(1.2)
    assert buggy_ratio >= ess_fraction_threshold


def test_concentration_exempted_from_ess_gate() -> None:
    _AUX_PARAMS = {"concentration"}
    ess_dict = {
        "lambda[0]": 6000.0,
        "lambda[1]": 5500.0,
        "concentration": 800.0,
    }
    total_draws = 8000
    ess_fraction_threshold = 0.4

    all_ratios = min(v / total_draws for v in ess_dict.values())
    assert all_ratios < ess_fraction_threshold

    lambda_ess = {
        k: v for k, v in ess_dict.items() if k.split("[")[0] not in _AUX_PARAMS
    }
    min_ratio = min(v / total_draws for v in lambda_ess.values())
    assert min_ratio >= ess_fraction_threshold
