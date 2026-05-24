"""Test empirical precision prior for unmeasured entries."""
from __future__ import annotations

from engine.calibrate.beta import BetaPosterior
from engine.calibrate.calibrate import apply_empirical_precision_prior


def test_replaces_beta_1_1_with_measured_mean() -> None:
    measured = {
        ("LLM01", "security"): BetaPosterior(alpha=10.0, beta=3.0),
        ("LLM05", "security"): BetaPosterior(alpha=8.0, beta=4.0),
    }
    unmeasured_key = ("LLM06", "security")
    all_precision = {
        **measured,
        unmeasured_key: BetaPosterior(alpha=1.0, beta=1.0),
    }

    result = apply_empirical_precision_prior(all_precision, frame_blind_ids=set())

    updated = result[unmeasured_key]
    assert updated.alpha != 1.0
    assert updated.beta != 1.0
    mean_measured = sum(bp.mean for bp in measured.values()) / len(measured)
    assert abs(updated.mean - mean_measured) < 0.05


def test_skips_frame_blind_entries() -> None:
    all_precision = {
        ("LLM01", "security"): BetaPosterior(alpha=10.0, beta=3.0),
        ("LLM04", "security"): BetaPosterior(alpha=1.0, beta=1.0),
    }

    result = apply_empirical_precision_prior(
        all_precision, frame_blind_ids={"LLM04"},
    )

    assert result[("LLM04", "security")] == BetaPosterior(alpha=1.0, beta=1.0)


def test_no_change_when_all_measured() -> None:
    all_precision = {
        ("LLM01", "security"): BetaPosterior(alpha=10.0, beta=3.0),
        ("LLM05", "security"): BetaPosterior(alpha=8.0, beta=4.0),
    }

    result = apply_empirical_precision_prior(all_precision, frame_blind_ids=set())

    assert result == all_precision
