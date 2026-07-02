"""Prospective power statement for quadratic-weighted kappa (U3 T5/T6).

REGISTERED: 2026-06-30
Scope: FUTURE CYCLES ONLY.  NOT a 2026 pre-registration (κ = 0.20 was
observed first; κ = 0.40 is a forward-looking target registered 2026-06-30).

Criterion: ``"design_n_ci_excludes_zero"`` (κ > 0) — honestly labelled.
NOT κ ≥ 0.40 non-inferiority.

Power formula (F5 — β-term mandatory):

    n_required = ceil( σ² · (z_{1-α/2} + z_{1-β})² / κ² )

The v1 formula (omitted the z_{1-β} term) delivered 50% power mislabelled
"0.95 confidence".  The β-term (z_{power}) is mandatory for correct power.

H1 joint probability model (registered, F16):
    3-tier system with uniform marginals (1/3 per tier for both raters).
    Joint matrix entries:
        diagonal     p_ii  = 1/5   (three cells × 1/5 = 3/5)
        off-diagonal p_ij  = 1/15  (six cells × 1/15 = 2/5)
        total = 1.0  ✓
    Agreement weights: w_ij = 1 − (i−j)²/4  (k=3, so (k−1)²=4)
        P_e = 2/3, P_o = 4/5
        κ = (P_o − P_e)/(1 − P_e) = (2/15)/(1/3) = 2/5 = 0.40  ✓
    Fleiss-Cohen-Everitt delta-method σ² (exact rational):
        g_ij  = w_ij − (1−κ)(ā_i + ā_j)
        Σ p_ij g_ij² = 13/125  (note: Σ p_ij g_ij = 0 for this model)
        σ² = (13/125) / (1−P_e)² = (13/125) · 9 = 117/125 = 0.936

DISCLOSURE (F6/F21/F25):
    The Fleiss-Cohen-Everitt asymptotic variance is a COARSE DESIGN-STAGE
    SURROGATE, DISTINCT from and NOT governing the reported paired-draw
    bootstrap CI.  Normal-approximation at n ≈ 20 with ranked/stratum
    dependence violates the iid assumption.  This power statement is a rough
    taxonomy-sizing guide, not a claim about the reported CI.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

__all__ = [
    "H1_JOINT_3TIER_UNIFORM",
    "H1_SIGMA2",
    "DISCLOSURE",
    "STRUCTURAL_NOTE",
    "REGISTERED_DATE",
    "kappa_sample_size_required",
    "prospective_power_statement",
    "_agreement_weights",
    "_fleiss_cohen_everitt_sigma2",
    "_excludes_zero_at_n",
]

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

REGISTERED_DATE: str = "2026-06-30"

DISCLOSURE: str = (
    "Coarse design-stage surrogate. The Fleiss-Cohen-Everitt asymptotic "
    "variance is DISTINCT from and does NOT govern the reported paired-draw "
    "bootstrap CI. Normal-approximation at n ≈ 20 with ranked/stratum "
    "dependence violates iid. Power statement is a rough sizing guide only, "
    "not a claim about the reported CI."
)

STRUCTURAL_NOTE: str = (
    "n is a fixed ~20-entry taxonomy, not growable — "
    "structural-adequacy verdict, not a 'collect more entries' instruction."
)

#: H1 joint probability matrix (3-tier, uniform marginals, target κ = 0.40).
#: Diagonal = 1/5; off-diagonal = 1/15.  All row/col marginals = 1/3.
H1_JOINT_3TIER_UNIFORM: npt.NDArray[np.float64] = np.array(
    [
        [1.0 / 5.0, 1.0 / 15.0, 1.0 / 15.0],
        [1.0 / 15.0, 1.0 / 5.0, 1.0 / 15.0],
        [1.0 / 15.0, 1.0 / 15.0, 1.0 / 5.0],
    ],
    dtype=np.float64,
)

#: Closed-form per-item asymptotic variance for H1_JOINT_3TIER_UNIFORM.
#: σ² = 117/125 = 0.936 exactly (Fleiss-Cohen-Everitt delta-method).
H1_SIGMA2: float = 117.0 / 125.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _agreement_weights(k: int) -> npt.NDArray[np.float64]:
    """Agreement weight matrix: w_ij = 1 − (i−j)² / (k−1)² for k categories."""
    idx = np.arange(k, dtype=np.float64)
    d_sq = (idx[:, None] - idx[None, :]) ** 2
    return 1.0 - d_sq / float((k - 1) ** 2)


def _fleiss_cohen_everitt_sigma2(joint: npt.NDArray[np.float64]) -> float:
    """Per-item asymptotic variance of quadratic-weighted kappa (Fleiss-Cohen-Everitt 1969).

    Variant: delta-method on the multinomial distribution with agreement
    weights  w_ij = 1 − (i−j)² / (k−1)²,  where k = joint.shape[0].

    Formula::

        σ² = [Σ_ij p_ij g_ij² − (Σ_ij p_ij g_ij)²] / (1 − P_e)²

    where::

        g_ij  = w_ij − (1−κ)(ā_i + ā_j)
        ā_i   = Σ_j w_ij · p_.j   (row expected agreement under marginals)
        ā_j   = Σ_i w_ij · p_i.   (col expected agreement under marginals)
        P_e   = Σ_ij w_ij · p_i. · p_.j
        P_o   = Σ_ij w_ij · p_ij
        κ     = (P_o − P_e) / (1 − P_e)

    Requires a square joint probability matrix (entries non-negative, sums to 1).

    Raises
    ------
    ValueError
        If the joint matrix is not square or if 1 − P_e is effectively zero.
    """
    k = joint.shape[0]
    if joint.ndim != 2 or joint.shape[1] != k:
        raise ValueError(f"joint must be a square 2-D matrix; got shape {joint.shape}")

    w = _agreement_weights(k)

    row_m: npt.NDArray[np.float64] = joint.sum(axis=1)  # p_i.
    col_m: npt.NDArray[np.float64] = joint.sum(axis=0)  # p_.j

    p_e = float(np.sum(w * np.outer(row_m, col_m)))
    p_o = float(np.sum(w * joint))

    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        raise ValueError(
            f"1 − P_e ≈ 0 (P_e = {p_e:.6f}): kappa is undefined."
        )

    kappa = (p_o - p_e) / denom

    # ā_i = Σ_j w_ij p_.j;  ā_j = Σ_i w_ij p_i.
    a_row: npt.NDArray[np.float64] = w @ col_m  # shape (k,)
    a_col: npt.NDArray[np.float64] = w.T @ row_m  # shape (k,)

    # g_ij = w_ij − (1 − κ)(ā_i + ā_j)
    g: npt.NDArray[np.float64] = w - (1.0 - kappa) * (a_row[:, None] + a_col[None, :])

    big_a = float(np.sum(joint * g**2))
    big_b = float(np.sum(joint * g)) ** 2

    return (big_a - big_b) / denom**2


def _excludes_zero_at_n(
    kappa: float,
    sigma2: float,
    n: int,
    alpha: float,
) -> bool:
    """Return True iff the DESIGN-n CI [κ ± z_{α/2}·σ/√n] excludes zero.

    Uses the design (target) κ, not any observed estimate.
    This is a rough sizing check distinct from the reported bootstrap CI.
    """
    if n <= 0:
        return False
    z_alpha_half = float(norm.ppf(1.0 - alpha / 2.0))
    se = math.sqrt(sigma2 / n)
    ci_lo = kappa - z_alpha_half * se
    return bool(ci_lo > 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def kappa_sample_size_required(
    target_kappa: float,
    alpha: float,
    power: float,
    variance_factor: float,
) -> int:
    """Required sample size for a weighted-kappa significance test.

    Applies the corrected (F5) formula with both the α-term and the β-term::

        n = ceil( σ² · (z_{1−α/2} + z_{1−β})² / κ² )

    The v1 formula used only z_{1−α/2}, which gives 50% power — not the
    stated "0.95 confidence".  The z_{1−β} term (= z_{power}) is mandatory.

    Parameters
    ----------
    target_kappa:
        Target κ for which to size the study.  Must be > 0.
    alpha:
        Two-sided significance level, e.g. 0.05.  Must be in (0, 1).
    power:
        Desired power 1−β, e.g. 0.80.  Must be in (0, 1).
    variance_factor:
        Per-item asymptotic variance σ² of the weighted-kappa estimator
        (e.g. ``H1_SIGMA2`` from the registered Fleiss-Cohen-Everitt model).
        Must be > 0.

    Returns
    -------
    int
        Minimum sample size n satisfying the stated power requirement.

    Raises
    ------
    ValueError
        If ``target_kappa ≤ 0``, ``alpha`` or ``power`` outside (0, 1),
        or ``variance_factor ≤ 0``.
    """
    if target_kappa <= 0.0:
        raise ValueError(f"target_kappa must be > 0; got {target_kappa!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1); got {power!r}")
    if variance_factor <= 0.0:
        raise ValueError(f"variance_factor must be > 0; got {variance_factor!r}")

    z_alpha_half = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    z_sum_sq = (z_alpha_half + z_beta) ** 2

    n_raw = variance_factor * z_sum_sq / target_kappa**2
    return math.ceil(n_raw)


def prospective_power_statement(
    *,
    target_kappa: float = 0.40,
    alpha: float = 0.05,
    power: float = 0.80,
    current_n: int = 20,
    registered_date: str = REGISTERED_DATE,
) -> dict[str, object]:
    """Build the ``prospective_power`` block for the baselines manifest (U3 T6).

    All parameters carry registered defaults; any change requires a dated
    amendment note (no U5 escape hatch).

    Returns
    -------
    dict[str, object]
        JSON-serialisable dict with keys:
        ``target_kappa``, ``alpha_two_sided``, ``power_1_minus_beta``,
        ``criterion``, ``registered_date``, ``scope``, ``method``,
        ``h1_joint_model``, ``sigma2``, ``n_required``, ``current_n``,
        ``excludes_zero_at_current_n``, ``stage``,
        ``disclosure``, ``structural_adequacy_verdict``.

    Notes
    -----
    ``excludes_zero_at_current_n`` is derived from the DESIGN κ, not any
    observed 2026 estimate.  At ``current_n = 20`` it is ``False`` because
    n_required > 20 (the 20-entry taxonomy is structurally inadequate for
    the design-level CI to clear zero at κ = 0.40).

    DISCLOSURE: the Fleiss-Cohen-Everitt σ² is a coarse design-stage
    surrogate, DISTINCT from and NOT governing the reported paired-draw
    bootstrap CI.
    """
    sigma2 = _fleiss_cohen_everitt_sigma2(H1_JOINT_3TIER_UNIFORM)
    n_required = kappa_sample_size_required(
        target_kappa=target_kappa,
        alpha=alpha,
        power=power,
        variance_factor=sigma2,
    )
    excludes_zero = _excludes_zero_at_n(target_kappa, sigma2, current_n, alpha)

    return {
        "target_kappa": target_kappa,
        "alpha_two_sided": alpha,
        "power_1_minus_beta": power,
        "criterion": "design_n_ci_excludes_zero",
        "registered_date": registered_date,
        "scope": "prospective_future_cycles",
        "method": "fleiss_cohen_everitt_wk_variance",
        "h1_joint_model": {
            "variant": "3tier_uniform_marginal",
            "description": (
                "3 tiers, uniform marginals (1/3 each). "
                "Diagonal=1/5, off-diagonal=1/15. "
                "Agreement weights w_ij=1-(i-j)^2/4. "
                "Achieves kappa=0.40 exactly."
            ),
            "p_diagonal": 1.0 / 5.0,
            "p_off_diagonal": 1.0 / 15.0,
            "p_e": 2.0 / 3.0,
            "p_o": 4.0 / 5.0,
        },
        "sigma2": sigma2,
        "n_required": n_required,
        "current_n": current_n,
        "excludes_zero_at_current_n": excludes_zero,
        "stage": "design",
        "disclosure": DISCLOSURE,
        "structural_adequacy_verdict": STRUCTURAL_NOTE,
    }
