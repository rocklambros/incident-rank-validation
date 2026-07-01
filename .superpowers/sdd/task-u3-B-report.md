# U3 Cluster B Report — T5 / T6 / T14 (Prospective Power)

## Status: DONE — provisional gate GREEN

---

## Corrected formula (F5 — β-term mandatory)

    n_required = ceil( σ² · (z_{1-α/2} + z_{1-β})² / κ² )

Registered inputs: κ=0.40, α=0.05 (two-sided), power=0.80.
z_{0.975} = 1.9600, z_{0.80} = 0.8416, z_sum = 2.8016, z_sum² = 7.8489.
n_raw = 0.936 × 7.8489 / 0.16 = 45.916 → **n_required = 46**.

The v1 formula (omitted z_{1-β}) gave n=23 at 50% power — that was wrong.

---

## σ² variant + marginal chosen (F16)

**Variant:** Fleiss-Cohen-Everitt (1969) asymptotic delta-method with agreement
weights w_ij = 1 − (i−j)²/(k−1)², k=3.

**H1 joint model (registered 2026-06-30):**
3-tier system, uniform marginals (1/3 per tier).
Diagonal = 1/5 per cell; off-diagonal = 1/15 per cell.
Achieves κ = 0.40 exactly: P_e = 2/3, P_o = 4/5, κ = (2/15)/(1/3) = 2/5 ✓.

**Closed-form derivation (exact rational):**
g_ij = w_ij − (1−κ)(ā_i + ā_j); Σ p_ij g_ij = 0 for this model.
Σ p_ij g_ij² = 13/125; σ² = (13/125) / (1/9) = **117/125 = 0.936** exactly.

---

## MC cross-check result (T6 provisional gate)

Parameters: n_items=500, n_reps=5000, seed=20260630.
σ²_mc ≈ 0.936 ± ~2% (varies by run); relative error < 6% < 15% tolerance.
**Gate: GREEN — closed form and simulation agree.**

Independent T14 MC (n_items=500, n_reps=3000, seed=42):
σ²_mc within 15% of H1_SIGMA2; n_mc within ±3 of n_closed. **GREEN.**

---

## Disclosure / prospective framing

- `disclosure` field: "Coarse design-stage surrogate. The Fleiss-Cohen-Everitt
  asymptotic variance is DISTINCT from and does NOT govern the reported
  paired-draw bootstrap CI. Normal-approximation at n≈20 with ranked/stratum
  dependence violates iid. Power statement is a rough sizing guide only."
- `scope`: "prospective_future_cycles" — NOT a 2026 pre-registration.
- `registered_date`: "2026-06-30".
- `stage`: "design".
- `criterion`: "design_n_ci_excludes_zero" (κ>0, honestly labelled; NOT non-inferiority).
- `excludes_zero_at_current_n`: **False** at n=20 (n_required=46 > 20 — structural inadequacy).
- `structural_adequacy_verdict`: "n is a fixed ~20-entry taxonomy, not growable —
  structural-adequacy verdict, not a 'collect more entries' instruction."

---

## T14 oracle spot-check

Independent algebraic arrangement (solve for SE, then invert):
    SE_target = κ / (z_{1-α/2} + z_{1-β})
    n_oracle = ceil(σ² / SE_target²)
Gives n_oracle = 46. Agrees with primary formula across a (κ, power) grid.
Also verified: n_v1_bug (omitting β-term) = 23 < 46, confirming β-term contribution.

---

## Gate outputs

- `uv run pytest -q` (full suite): all pass, 0 failures (exit 0).
- `uv run ruff check .`: all checks passed.
- `uv run mypy engine tests`: success, no issues in 254 source files.
- F4 pin: 37 new tests all green (T5: 11, T6: 21, T14: 5 in two test classes).

---

## Files changed

- **Created:** `engine/decide/prospective_power.py` — `kappa_sample_size_required`,
  `prospective_power_statement`, `_fleiss_cohen_everitt_sigma2`, `_agreement_weights`,
  `_excludes_zero_at_n`, constants `H1_JOINT_3TIER_UNIFORM`, `H1_SIGMA2`, `DISCLOSURE`,
  `STRUCTURAL_NOTE`, `REGISTERED_DATE`.
- **Created:** `tests/unit/test_prospective_power.py` — T5 + T6 tests (RED→GREEN).
- **Created:** `tests/unit/test_power_oracle.py` — T14 oracle spot-check (RED→GREEN).

---

## Concerns

None. The 15%-relative-tolerance MC gate is deliberately generous (3σ ≈ 6%) to avoid
flakiness; a 2× error in the closed form would still fail it. The normal-approximation
caveat at n=20 is fully disclosed in the `disclosure` field — this is the design-stage
surrogate acknowledged in the plan.
