import pytest

from engine.model.inference import DiagnosticsFailure, _check_diagnostics


def test_gate_passes_clean() -> None:
    _check_diagnostics(
        r_hat={"lambda[0]": 1.001}, ess={"lambda[0]": 9000.0},
        divergences=0, ess_fraction=0.4, total_draws=16000,
    )  # must not raise


def test_gate_raises_on_divergence() -> None:
    with pytest.raises(DiagnosticsFailure, match="divergen"):
        _check_diagnostics(
            r_hat={"lambda[0]": 1.0}, ess={"lambda[0]": 9000.0},
            divergences=5, ess_fraction=0.4, total_draws=16000,
        )


def test_gate_raises_on_low_ess() -> None:
    with pytest.raises(DiagnosticsFailure, match="ESS"):
        _check_diagnostics(
            r_hat={"sigma_u": 1.0}, ess={"sigma_u": 100.0},  # 100/16000 << 0.4
            divergences=0, ess_fraction=0.4, total_draws=16000,
        )


def test_gate_raises_on_high_rhat() -> None:
    with pytest.raises(DiagnosticsFailure, match="R-hat"):
        _check_diagnostics(
            r_hat={"lambda[0]": 1.2}, ess={"lambda[0]": 9000.0},
            divergences=0, ess_fraction=0.4, total_draws=16000,
        )


def test_gate_raises_on_single_divergence() -> None:
    """J1 (U2-5): a single post-warmup divergence is sufficient to trigger the gate.

    This is the AUTHORITATIVE deterministic proof that _check_diagnostics gates on
    divergences=1 — no NUTS sampler involved, cross-platform stable. The shared
    _check_diagnostics function is called identically by run_inference (primary) and
    run_robustness_inference (robustness specs), so this test covers both paths.
    """
    with pytest.raises(DiagnosticsFailure, match="divergen"):
        _check_diagnostics(
            r_hat={"lambda[0]": 1.0}, ess={"lambda[0]": 9000.0},
            divergences=1, ess_fraction=0.4, total_draws=16000,
        )
