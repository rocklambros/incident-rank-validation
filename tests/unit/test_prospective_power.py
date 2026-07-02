"""Unit tests for engine.decide.prospective_power — T5 (sample-size solver)
and T6 (power statement + Monte-Carlo variance cross-check).

T5 pins:
  - n_required vs hand-computed closed form (WITH the β-term, F5)
  - edge cases: κ ≤ 0 raises, α/power outside (0,1) raises
  - monotonicity: larger κ → smaller n; larger power → larger n

T6 pins:
  - prospective_power_statement returns the required keys
  - excludes_zero_at_current_n is False at n = 20 (structural inadequacy)
  - disclosure field is present and non-empty
  - prospective framing: registered_date, scope, stage
  - Monte-Carlo variance cross-check (F12/F21): σ²_mc agrees with the
    closed-form H1_SIGMA2 within 15 % relative tolerance.
    Output is PROVISIONAL until this gate passes.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import norm

from engine.decide.prospective_power import (
    H1_JOINT_3TIER_UNIFORM,
    H1_SIGMA2,
    REGISTERED_DATE,
    _excludes_zero_at_n,
    _fleiss_cohen_everitt_sigma2,
    kappa_sample_size_required,
    prospective_power_statement,
)

# ---------------------------------------------------------------------------
# Monte-Carlo helpers (T6 cross-check gate — F12/F21)
# ---------------------------------------------------------------------------

_MC_N_ITEMS: int = 500  # items per simulated dataset
_MC_N_REPS: int = 5_000  # number of replications
_MC_SEED: int = 20_260_630  # seed = registered date
_MC_RTOL: float = 0.15  # 15 % relative tolerance (3σ ≈ 6 % << 15 %)


def _kappa_from_counts(counts: npt.NDArray[np.float64]) -> float:
    """Quadratic-weighted kappa from a count matrix (disagreement weights (i-j)²).

    Mirrors engine/decide/kappa.py for independence.
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
    obs_d = float(np.sum(d * p))
    return 1.0 - obs_d / exp_d


def _mc_sigma2_estimate(
    joint: npt.NDArray[np.float64],
    n_items: int,
    n_reps: int,
    seed: int,
) -> float:
    """Estimate σ² = n · Var(κ̂) via Monte-Carlo draws from ``joint``.

    Draws ``n_reps`` simulated datasets of ``n_items`` paired (tier_A, tier_B)
    observations from the joint distribution, computes kappa for each, and
    returns ``n_items · sample_variance(kappas)``.
    """
    rng = np.random.default_rng(seed)
    k = joint.shape[0]
    flat_probs = joint.ravel()
    kappas = np.empty(n_reps, dtype=np.float64)
    for i in range(n_reps):
        cells = rng.choice(k * k, size=n_items, p=flat_probs)
        counts = np.bincount(cells, minlength=k * k).reshape(k, k).astype(np.float64)
        kappas[i] = _kappa_from_counts(counts)
    valid = kappas[~np.isnan(kappas)]
    return float(n_items * np.var(valid, ddof=0))


# ---------------------------------------------------------------------------
# T5 — kappa_sample_size_required
# ---------------------------------------------------------------------------


class TestKappaSampleSizeRequired:
    """T5: pin n_required vs hand-computed closed form; edge cases; monotonicity."""

    # Hand-computed n_required for the registered defaults (F5 β-term present):
    #   σ² = 117/125 = 0.936
    #   z_{0.975} = 1.9599639845400536, z_{0.80} = 0.8416212335729143
    #   z_sum = 2.8015852181129679,  z_sum² ≈ 7.84888
    #   n_raw = 0.936 × 7.84888 / 0.40² = 7.3465… / 0.16 ≈ 45.916
    #   n_required = ceil(45.916) = 46
    HAND_N: int = 46

    def test_pin_registered_defaults(self) -> None:
        """n_required matches hand-computed value 46 for the registered inputs."""
        n = kappa_sample_size_required(
            target_kappa=0.40,
            alpha=0.05,
            power=0.80,
            variance_factor=H1_SIGMA2,
        )
        assert n == self.HAND_N, (
            f"n_required = {n} but hand-computed value is {self.HAND_N}. "
            "Check that the β-term (z_{1-β}) is present in the formula."
        )

    def test_pin_matches_closed_form_directly(self) -> None:
        """Verify the formula n = ceil(σ²·(z_{α/2}+z_β)² / κ²) numerically."""
        alpha, power, kappa = 0.05, 0.80, 0.40
        sigma2 = H1_SIGMA2
        z_a = float(norm.ppf(1.0 - alpha / 2.0))
        z_b = float(norm.ppf(power))
        n_expected = math.ceil(sigma2 * (z_a + z_b) ** 2 / kappa**2)
        n_got = kappa_sample_size_required(kappa, alpha, power, sigma2)
        assert n_got == n_expected

    # Edge cases ----------------------------------------------------------

    def test_kappa_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="target_kappa"):
            kappa_sample_size_required(0.0, 0.05, 0.80, H1_SIGMA2)

    def test_kappa_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="target_kappa"):
            kappa_sample_size_required(-0.1, 0.05, 0.80, H1_SIGMA2)

    def test_alpha_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            kappa_sample_size_required(0.40, 0.0, 0.80, H1_SIGMA2)

    def test_alpha_one_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            kappa_sample_size_required(0.40, 1.0, 0.80, H1_SIGMA2)

    def test_power_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="power"):
            kappa_sample_size_required(0.40, 0.05, 0.0, H1_SIGMA2)

    def test_power_one_raises(self) -> None:
        with pytest.raises(ValueError, match="power"):
            kappa_sample_size_required(0.40, 0.05, 1.0, H1_SIGMA2)

    def test_variance_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="variance_factor"):
            kappa_sample_size_required(0.40, 0.05, 0.80, 0.0)

    # Monotonicity --------------------------------------------------------

    def test_monotone_in_kappa(self) -> None:
        """Larger κ → strictly smaller n (easier to detect)."""
        ns = [
            kappa_sample_size_required(k, 0.05, 0.80, H1_SIGMA2)
            for k in [0.20, 0.30, 0.40, 0.60, 0.80]
        ]
        assert ns == sorted(ns, reverse=True), f"n not monotonically decreasing: {ns}"

    def test_monotone_in_power(self) -> None:
        """Higher power → larger or equal n."""
        ns = [
            kappa_sample_size_required(0.40, 0.05, p, H1_SIGMA2)
            for p in [0.50, 0.70, 0.80, 0.90, 0.95]
        ]
        assert ns == sorted(ns), f"n not monotonically non-decreasing: {ns}"

    def test_monotone_in_alpha(self) -> None:
        """Stricter α (smaller value) → larger n."""
        ns = [
            kappa_sample_size_required(0.40, a, 0.80, H1_SIGMA2)
            for a in [0.10, 0.05, 0.02, 0.01]
        ]
        assert ns == sorted(ns), f"n not monotonically non-decreasing: {ns}"


# ---------------------------------------------------------------------------
# T6 — prospective_power_statement + MC variance cross-check
# ---------------------------------------------------------------------------


class TestProspectivePowerStatement:
    """T6: dict keys, prospective framing, excludes_zero=False at n=20,
    disclosure present, and MC variance cross-check gate."""

    REQUIRED_KEYS: frozenset[str] = frozenset(
        {
            "target_kappa",
            "alpha_two_sided",
            "power_1_minus_beta",
            "criterion",
            "registered_date",
            "scope",
            "method",
            "h1_joint_model",
            "sigma2",
            "n_required",
            "current_n",
            "excludes_zero_at_current_n",
            "stage",
            "disclosure",
            "structural_adequacy_verdict",
        }
    )

    def _default_stmt(self) -> dict[str, object]:
        return prospective_power_statement()

    def test_all_required_keys_present(self) -> None:
        stmt = self._default_stmt()
        missing = self.REQUIRED_KEYS - set(stmt)
        assert not missing, f"Missing keys: {sorted(missing)}"

    def test_n_required_is_46(self) -> None:
        stmt = self._default_stmt()
        assert stmt["n_required"] == 46

    def test_excludes_zero_at_current_n_is_false(self) -> None:
        """At n=20 the design CI does NOT exclude 0 — structural inadequacy."""
        stmt = self._default_stmt()
        assert stmt["excludes_zero_at_current_n"] is False, (
            "Expected excludes_zero_at_current_n=False at n=20 "
            "(n_required=46 > 20 implies the taxonomy is structurally inadequate)"
        )

    def test_excludes_zero_at_n_required(self) -> None:
        """At n=n_required the design CI should exclude 0."""
        # Use the underlying functions directly — avoids extracting typed values
        # from dict[str, object] and keeps the test readable.
        sigma2 = H1_SIGMA2
        n_req = kappa_sample_size_required(0.40, 0.05, 0.80, sigma2)
        assert _excludes_zero_at_n(0.40, sigma2, n_req, 0.05), (
            "CI should exclude 0 at n=n_required"
        )

    def test_disclosure_field_present_and_non_empty(self) -> None:
        stmt = self._default_stmt()
        disclosure = stmt.get("disclosure", "")
        assert isinstance(disclosure, str) and len(disclosure) > 20, (
            "disclosure field must be a non-empty string describing the surrogate caveat"
        )

    def test_disclosure_mentions_surrogate(self) -> None:
        stmt = self._default_stmt()
        text = str(stmt["disclosure"]).lower()
        assert "surrogate" in text or "distinct" in text, (
            "disclosure must mention 'surrogate' or 'distinct'"
        )

    def test_stage_is_design(self) -> None:
        stmt = self._default_stmt()
        assert stmt["stage"] == "design"

    def test_scope_is_prospective_future_cycles(self) -> None:
        stmt = self._default_stmt()
        assert stmt["scope"] == "prospective_future_cycles"

    def test_registered_date(self) -> None:
        stmt = self._default_stmt()
        assert stmt["registered_date"] == REGISTERED_DATE

    def test_criterion_is_honest(self) -> None:
        stmt = self._default_stmt()
        assert stmt["criterion"] == "design_n_ci_excludes_zero"

    def test_method_is_fleiss_cohen_everitt(self) -> None:
        stmt = self._default_stmt()
        assert "fleiss_cohen_everitt" in str(stmt["method"]).lower()

    def test_current_n_default(self) -> None:
        stmt = self._default_stmt()
        assert stmt["current_n"] == 20

    def test_sigma2_matches_constant(self) -> None:
        stmt = self._default_stmt()
        sigma2_raw = stmt["sigma2"]
        assert isinstance(sigma2_raw, float)
        assert abs(sigma2_raw - H1_SIGMA2) < 1e-12

    def test_custom_current_n_gt_n_required_excludes_zero(self) -> None:
        """When current_n = n_required + 10 the CI should exclude 0."""
        stmt = prospective_power_statement(current_n=56)
        assert stmt["excludes_zero_at_current_n"] is True

    def test_structural_note_present(self) -> None:
        stmt = self._default_stmt()
        note = str(stmt.get("structural_adequacy_verdict", ""))
        assert "taxonomy" in note.lower() or "fixed" in note.lower()

    def test_h1_joint_model_block_present(self) -> None:
        stmt = self._default_stmt()
        model = stmt.get("h1_joint_model")
        assert isinstance(model, dict), "h1_joint_model must be a dict"
        d: dict[str, object] = model
        assert "variant" in d


class TestFCESigma2:
    """Unit tests for _fleiss_cohen_everitt_sigma2."""

    def test_registered_model_exact(self) -> None:
        """Closed-form σ² = 117/125 for H1_JOINT_3TIER_UNIFORM."""
        sigma2 = _fleiss_cohen_everitt_sigma2(H1_JOINT_3TIER_UNIFORM)
        assert abs(sigma2 - 117.0 / 125.0) < 1e-10, (
            f"σ² = {sigma2} but expected 117/125 = 0.936"
        )

    def test_perfect_agreement_joint(self) -> None:
        """Identity joint (all mass on diagonal) gives σ² = 0."""
        joint = np.diag([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        sigma2 = _fleiss_cohen_everitt_sigma2(joint)
        # kappa=1, g_ij = w_ij - 0*(ā_i + ā_j) = w_ij, but off-diag p=0
        # A = sum p_ii w_ii^2 - 0, (1-P_e)^2 > 0 — just check it's finite and >= 0
        assert math.isfinite(sigma2) and sigma2 >= 0.0


# ---------------------------------------------------------------------------
# T6 Monte-Carlo variance cross-check (provisional gate — F12/F21)
# ---------------------------------------------------------------------------


class TestMCVarianceCrossCheck:
    """Monte-Carlo estimate of σ² must agree with the closed-form H1_SIGMA2
    within 15 % relative tolerance.  Output is PROVISIONAL until this passes.
    """

    def test_mc_sigma2_agrees_with_closed_form(self) -> None:
        """σ²_mc ≈ H1_SIGMA2 within 15 % (F12/F21 provisional gate)."""
        sigma2_mc = _mc_sigma2_estimate(
            H1_JOINT_3TIER_UNIFORM,
            n_items=_MC_N_ITEMS,
            n_reps=_MC_N_REPS,
            seed=_MC_SEED,
        )
        rel_error = abs(sigma2_mc - H1_SIGMA2) / H1_SIGMA2
        assert rel_error <= _MC_RTOL, (
            f"MC σ²_mc = {sigma2_mc:.4f} vs closed-form H1_SIGMA2 = {H1_SIGMA2:.4f}; "
            f"relative error {rel_error:.3f} exceeds tolerance {_MC_RTOL}. "
            f"PROVISIONAL: output is not publishable until this gate passes."
        )

    def test_mc_kappa_mean_near_target(self) -> None:
        """The MC mean kappa should be near 0.40 (sanity check for joint model)."""
        rng = np.random.default_rng(_MC_SEED + 1)
        k = H1_JOINT_3TIER_UNIFORM.shape[0]
        flat_probs = H1_JOINT_3TIER_UNIFORM.ravel()
        kappas = []
        for _ in range(500):
            cells = rng.choice(k * k, size=_MC_N_ITEMS, p=flat_probs)
            counts = np.bincount(cells, minlength=k * k).reshape(k, k).astype(np.float64)
            kv = _kappa_from_counts(counts)
            if math.isfinite(kv):
                kappas.append(kv)
        mean_k = float(np.mean(kappas))
        assert abs(mean_k - 0.40) < 0.05, (
            f"MC mean kappa = {mean_k:.3f}; expected ≈ 0.40 for the H1 joint model"
        )
