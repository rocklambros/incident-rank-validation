"""T14 — Power-math oracle spot-check.

Scope: formula math only.  No engine models, no corpus data.

Two independent verification paths for n_required:

1.  PRIMARY  (engine/decide/prospective_power.py):
        n = ceil( σ² · (z_{1-α/2} + z_{1-β})² / κ² )

2.  ORACLE   (this file, independent algebraic arrangement):
        Solve for the SE threshold that satisfies the power equation, then
        invert for n:
            SE_target = κ / (z_{1-α/2} + z_{1-β})
            n_oracle  = ceil( σ² / SE_target² )
        = ceil( σ² · (z_{1-α/2} + z_{1-β})² / κ² )   ← same algebra, distinct path

The two arrangements are equivalent by inspection; the test asserts agreement.

Additionally (F12/F21 cross-check): an independent Monte-Carlo estimate of σ²
is run here (different seed from T6) and checked against H1_SIGMA2, confirming
the closed-form surrogate variance is consistent with simulation.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from engine.decide.prospective_power import (
    H1_JOINT_3TIER_UNIFORM,
    H1_SIGMA2,
    kappa_sample_size_required,
)

# ---------------------------------------------------------------------------
# MC helper (independent seed / parameterisation from test_prospective_power)
# ---------------------------------------------------------------------------

_ORACLE_MC_N_ITEMS: int = 500
_ORACLE_MC_N_REPS: int = 3_000  # different count from T6 for independence
_ORACLE_MC_SEED: int = 42  # different seed from T6


def _kappa_from_counts_oracle(counts: npt.NDArray[np.float64]) -> float:
    """Quadratic-weighted kappa from a count matrix.
    Independent re-implementation using disagreement weights (i-j)².
    """
    n = float(counts.sum())
    if n < 1e-9:
        return float("nan")
    p = counts / n
    row_m = p.sum(axis=1)
    col_m = p.sum(axis=0)
    k = counts.shape[0]
    idx = np.arange(k, dtype=np.float64)
    d = (idx[:, None] - idx[None, :]) ** 2
    exp_d = float(np.sum(d * np.outer(row_m, col_m)))
    if exp_d < 1e-9:
        return 1.0
    return 1.0 - float(np.sum(d * p)) / exp_d


def _oracle_mc_sigma2(
    joint: npt.NDArray[np.float64],
    n_items: int,
    n_reps: int,
    seed: int,
) -> float:
    """Independent MC estimate of σ² for the oracle cross-check."""
    rng = np.random.default_rng(seed)
    k = joint.shape[0]
    flat_probs = joint.ravel()
    kappas = np.empty(n_reps, dtype=np.float64)
    for i in range(n_reps):
        cells = rng.choice(k * k, size=n_items, p=flat_probs)
        counts = np.bincount(cells, minlength=k * k).reshape(k, k).astype(np.float64)
        kappas[i] = _kappa_from_counts_oracle(counts)
    valid = kappas[~np.isnan(kappas)]
    return float(n_items * np.var(valid, ddof=0))


# ---------------------------------------------------------------------------
# Oracle formula (independent arrangement)
# ---------------------------------------------------------------------------


def _oracle_n_required(
    target_kappa: float,
    alpha: float,
    power: float,
    sigma2: float,
) -> int:
    """Oracle arrangement: solve for SE then invert.

    Standard power equation for a one-sided-equivalent test at level α/2
    with power 1-β:

        κ / SE = z_{1-α/2} + z_{1-β}
        ⟹ SE_target = κ / (z_{1-α/2} + z_{1-β})
        ⟹ n = ceil( σ² / SE_target² )
             = ceil( σ² · (z_{1-α/2} + z_{1-β})² / κ² )

    This is algebraically equivalent to the primary formula but expressed via
    an intermediate SE-threshold variable.
    """
    z_a = float(norm.ppf(1.0 - alpha / 2.0))
    z_b = float(norm.ppf(power))
    se_target = target_kappa / (z_a + z_b)
    return math.ceil(sigma2 / se_target**2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPowerOracleSpotCheck:
    """T14: independent algebraic arrangement agrees with primary formula."""

    def test_oracle_matches_primary_registered_inputs(self) -> None:
        """Primary and oracle formulas must agree for the registered defaults."""
        kappa, alpha, power = 0.40, 0.05, 0.80
        primary = kappa_sample_size_required(kappa, alpha, power, H1_SIGMA2)
        oracle = _oracle_n_required(kappa, alpha, power, H1_SIGMA2)
        assert primary == oracle, (
            f"Primary formula gives n={primary}, oracle gives n={oracle}. "
            f"They must agree — one arrangement is wrong."
        )

    def test_oracle_pin_is_46(self) -> None:
        """Oracle spot-check: n_required = 46 for the registered inputs."""
        n = _oracle_n_required(0.40, 0.05, 0.80, H1_SIGMA2)
        assert n == 46

    def test_oracle_matches_primary_varied_inputs(self) -> None:
        """Primary and oracle agree across a grid of (kappa, power) pairs."""
        for kappa in [0.20, 0.40, 0.60, 0.80]:
            for power in [0.70, 0.80, 0.90]:
                primary = kappa_sample_size_required(kappa, 0.05, power, H1_SIGMA2)
                oracle = _oracle_n_required(kappa, 0.05, power, H1_SIGMA2)
                assert primary == oracle, (
                    f"Disagreement at kappa={kappa}, power={power}: "
                    f"primary={primary}, oracle={oracle}"
                )

    def test_oracle_with_mc_sigma2(self) -> None:
        """Oracle n_required using σ²_mc agrees with primary using σ²_closed
        within 2 items (MC noise) — cross-checks both formulas against simulation.
        """
        sigma2_mc = _oracle_mc_sigma2(
            H1_JOINT_3TIER_UNIFORM,
            n_items=_ORACLE_MC_N_ITEMS,
            n_reps=_ORACLE_MC_N_REPS,
            seed=_ORACLE_MC_SEED,
        )
        n_mc = _oracle_n_required(0.40, 0.05, 0.80, sigma2_mc)
        n_closed = _oracle_n_required(0.40, 0.05, 0.80, H1_SIGMA2)

        # Verify the MC σ² is in a reasonable range (15 % RTol)
        rel_error = abs(sigma2_mc - H1_SIGMA2) / H1_SIGMA2
        assert rel_error <= 0.15, (
            f"Oracle MC σ²_mc = {sigma2_mc:.4f} deviates from H1_SIGMA2 = {H1_SIGMA2:.4f} "
            f"by {rel_error:.1%} (>15 %); closed-form surrogate may be wrong."
        )

        # n_required derived from MC σ² should be within ±3 of the closed-form n
        assert abs(n_mc - n_closed) <= 3, (
            f"n_mc={n_mc} vs n_closed={n_closed}; gap > 3 suggests formula error."
        )

    def test_beta_term_is_present(self) -> None:
        """Regression: omitting the β-term (v1 bug) gives a different, smaller n.

        Without z_{1-β}, the formula gives 50 % power, not 80 %.
        This test confirms the β-term contributes a meaningful increase to n.
        """
        kappa, alpha, power, sigma2 = 0.40, 0.05, 0.80, H1_SIGMA2
        n_correct = kappa_sample_size_required(kappa, alpha, power, sigma2)

        # v1-style (alpha-only, no beta term) — manually compute:
        z_a = float(norm.ppf(1.0 - alpha / 2.0))
        n_v1_bug = math.ceil(sigma2 * z_a**2 / kappa**2)

        assert n_correct > n_v1_bug, (
            f"n_correct={n_correct} should be > n_v1_bug={n_v1_bug}; "
            "the β-term is missing (v1 regression)."
        )
