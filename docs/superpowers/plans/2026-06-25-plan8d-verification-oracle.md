# Plan 8d — Independent Python Verification Oracle (provisional gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent Python consistency-check oracle (`engine/verify/oracle.py`) that re-derives the engine's three headline deliverables by a *different method* and marks a cycle **PROVISIONAL** when any pre-declared tolerance is breached.

**Architecture:** The oracle reads a cycle's persisted artifacts (λ samples, vote ballots, hierarchical σ_u, the engine's incidence ranking) and re-derives — *without importing engine estimator code* — (D1) the incidence ranking from λ medians × strata sizes, (D2) the Plackett-Luce/Davidson vote ranking via a **Bradley-Terry MM/fixed-point** optimizer (a different family than the engine's `scipy.optimize` L-BFGS-B), and (D3) σ_u via a **DerSimonian-Laird moment surrogate** on the unpooled `poisson_flat` per-entry log-λ (not a matched NUTS posterior). It compares each to the engine's persisted output within pre-declared module-constant tolerances and emits `oracle_report.json`. `decide` runs the oracle after writing its artifacts; the report renders an Oracle Consistency Check section and a PROVISIONAL banner on any breach. No Merkle/signer gate (right-sized to an internal tool per the 8d decision).

**Tech Stack:** Python 3.12, numpy==2.1.3, scipy==1.15.0 (`scipy.stats.kendalltau` only), click. No new dependencies.

## Global Constraints

Every task's requirements implicitly include this section.

- **NO new dependencies.** numpy + scipy only. The oracle must use a *different optimizer/estimator family* than the engine on the same pinned stack — BT-MM (hand-rolled) for PL, a closed-form DSL moment estimator for σ_u. Do NOT add choix/pymc/statsmodels.
- **The oracle MUST NOT import engine estimator code.** `engine/verify/oracle.py` may import only `numpy`, `scipy.stats`, stdlib, and its own dataclasses. It must NOT import `engine.decide.concordance`, `engine.vote.plackett_luce`, `engine.model.*`, or `engine.calibrate.*`. It reads persisted artifacts as plain numpy/JSON. (The `run_oracle` orchestrator in `engine/verify/check.py` MAY import dataclasses from `oracle.py`, but not estimator code.)
- **Pre-declared tolerances are module constants** in `engine/verify/oracle.py` (committed = pre-registered). Do NOT add a `PreregManifest` field — it would rehash the frozen 2026 lock.
- **Provisional gate only — NO Merkle/signer machinery.** A tolerance breach flips a PROVISIONAL status in the report. Do NOT add a signer, `signed_at`, `read_and_verify_register` wiring, or a "refuse publishable report" gate. (8d decision, 2026-06-25.)
- **Deterministic & CPU-only.** The BT-MM fit starts from uniform worths and iterates deterministically. No RNG, no wall-clock.
- **CI gate (run the EXACT commands, whole-repo, before every commit):** `uv run ruff check .` → `uv run mypy engine tests` (engine AND tests) → before any push the FULL `uv run pytest -q` (NOT a `-k` subset). `isinstance` type-narrowing on JSON `object` values must use `int | float` union form (ruff UP038), not `(int, float)`.
- **mypy strict.** Every test function `-> None`; helpers fully typed; cast `np.median(...)`/`np.percentile(...)` results with `float(...)`.
- **No AI attribution in any commit message or GitHub-visible content.**
- **Branch:** all work on `plan7/engine-upgrade-recall-pl` (PR #22). Do not branch or merge.

## Deliverables, methods, and tolerances (pre-declared)

| ID | Deliverable | Engine method | Oracle method (different family) | Tolerance (module constant) |
|---|---|---|---|---|
| D1 | Incidence ranking (λ·size) | `_ranks_from_incidence` per-draw, persisted point ranking | re-derive from median λ × Σ strata sizes | `ORACLE_TAU_INCIDENCE = 0.95` Kendall-τ **and** exact top-tier set match; per-entry CI-overlap |
| D2 | PL/Davidson vote ranking | Davidson MLE via `scipy.optimize` L-BFGS-B (ν tie param) | Bradley-Terry **MM/fixed-point**, half-credit ties | `ORACLE_TAU_PL = 0.70` Kendall-τ **and** exact top-tier set match |
| D3 | σ_u (pooling SD) | NUTS HalfNormal posterior median (hierarchical) | **DerSimonian-Laird** moment estimator on unpooled `poisson_flat` log-λ | `ORACLE_SIGMA_U_BAND = 0.75` (|Δσ_u|) |

Tolerances are **coarse method-agreement bands for the build**; they are reviewed against real numbers when the cycle runs (Plan 8f). D3 is intentionally generous (moment estimator vs MCMC posterior is a method-bias comparison, not MCSE). A missing artifact ⇒ that deliverable is **SKIPPED**, not FAILED. Kappa itself is *not* separately gated: it is a deterministic function of the two rankings (both checked) and is MCSE-dominated over draws.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `engine/verify/__init__.py` | package marker | Create (Task 1) |
| `engine/verify/oracle.py` | Pure re-derivations (D1/D2/D3), tolerances, verdict dataclasses, comparison helpers. NO engine-estimator imports. | Create (Tasks 1–4) |
| `engine/verify/check.py` | `run_oracle(cycle_dir) -> OracleVerdict`: artifact loader + orchestration + `oracle_report.json` writer. | Create (Task 6) |
| `engine/cli/pipeline.py` | decide persists `incidence_ranking.json` + `vote_rankings.npy`/`vote_entry_ids.json` (T5); runs `run_oracle` after artifacts (T7); `verify-oracle` CLI (T6); `report_cmd` loads `oracle_report.json` (T7). | Modify (Tasks 5–7) |
| `engine/report/render.py` | `ReportInputs.oracle_verdict`; `_render_oracle_lines`; PROVISIONAL banner. | Modify (Task 7) |
| `tests/unit/test_oracle_*.py`, `tests/unit/test_oracle_run.py`, `tests/unit/test_render_oracle.py` | Tests. | Create (Tasks 1–7) |

---

### Task 1: Oracle incidence ranking re-derivation (D1)

**Files:**
- Create: `engine/verify/__init__.py` (empty), `engine/verify/oracle.py`
- Test: `tests/unit/test_oracle_incidence.py`

**Interfaces:**
- Produces:
  - `def oracle_incidence_ranking(lambda_samples: npt.NDArray[np.float64], entry_ids: tuple[str, ...], entry_strata: dict[str, tuple[str, ...]], stratum_sizes: dict[str, int]) -> tuple[str, ...]` — best→worst by `median(λ_e) × Σ_s stratum_sizes[s]`, ties broken by entry id.
  - `def oracle_incidence_intervals(lambda_samples: npt.NDArray[np.float64], entry_ids: tuple[str, ...], entry_strata: dict[str, tuple[str, ...]], stratum_sizes: dict[str, int]) -> dict[str, tuple[float, float]]` — per-entry (2.5, 97.5) percentile incidence CI.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_oracle_incidence.py`:

```python
"""Tests for the oracle incidence re-derivation (Plan 8d D1)."""
from __future__ import annotations

import numpy as np

from engine.verify.oracle import oracle_incidence_intervals, oracle_incidence_ranking


def test_incidence_ranking_orders_by_lambda_times_size() -> None:
    # Two entries, single stratum size 10. A has higher lambda -> A ranks first.
    lambda_samples = np.array([[0.5, 0.1], [0.6, 0.2], [0.4, 0.15]])
    entry_ids = ("A", "B")
    entry_strata = {"A": ("security",), "B": ("security",)}
    stratum_sizes = {"security": 10}
    ranking = oracle_incidence_ranking(
        lambda_samples, entry_ids, entry_strata, stratum_sizes
    )
    assert ranking == ("A", "B")


def test_incidence_ranking_uses_multi_stratum_sum() -> None:
    # B has lower lambda but spans two strata; its total exposure beats A.
    lambda_samples = np.array([[0.30, 0.20], [0.30, 0.20]])
    entry_ids = ("A", "B")
    entry_strata = {"A": ("security",), "B": ("security", "ai-harm")}
    stratum_sizes = {"security": 100, "ai-harm": 100}
    # A: 0.30*100 = 30 ; B: 0.20*200 = 40 -> B first
    ranking = oracle_incidence_ranking(
        lambda_samples, entry_ids, entry_strata, stratum_sizes
    )
    assert ranking == ("B", "A")


def test_incidence_intervals_are_ordered_pairs() -> None:
    lambda_samples = np.array([[0.5, 0.1], [0.6, 0.2], [0.4, 0.15], [0.55, 0.12]])
    entry_ids = ("A", "B")
    entry_strata = {"A": ("security",), "B": ("security",)}
    stratum_sizes = {"security": 10}
    ci = oracle_incidence_intervals(
        lambda_samples, entry_ids, entry_strata, stratum_sizes
    )
    assert set(ci.keys()) == {"A", "B"}
    for lo, hi in ci.values():
        assert lo <= hi


def test_incidence_ranking_tiebreak_by_entry_id() -> None:
    # Identical incidence -> deterministic order by entry id.
    lambda_samples = np.array([[0.2, 0.2], [0.2, 0.2]])
    entry_ids = ("B", "A")
    entry_strata = {"A": ("s",), "B": ("s",)}
    stratum_sizes = {"s": 5}
    ranking = oracle_incidence_ranking(
        lambda_samples, entry_ids, entry_strata, stratum_sizes
    )
    assert ranking == ("A", "B")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_oracle_incidence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.verify'`.

- [ ] **Step 3: Write the implementation**

Create `engine/verify/__init__.py` (empty file).

Create `engine/verify/oracle.py`:

```python
"""Independent consistency-check oracle (Plan 8d).

Re-derives the engine's headline deliverables by a DIFFERENT method on the
pinned numpy/scipy stack and compares within pre-declared tolerances.  This
module MUST NOT import engine estimator code (concordance / plackett_luce /
model / calibrate); it reads persisted artifacts as plain numpy/JSON so it is
a genuine cross-check, not a re-run.  It is a CONSISTENCY check, not
independent verification (shared author/conceptual source).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _incidence_value(
    lam: float,
    entry: str,
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> float:
    """lambda_e * sum of sizes of all strata entry e was observed in."""
    total_size = float(sum(stratum_sizes[s] for s in entry_strata[entry]))
    return lam * total_size


def oracle_incidence_ranking(
    lambda_samples: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> tuple[str, ...]:
    """Re-derive the incidence ranking (best->worst) from median lambda x size."""
    median_lambda = np.median(lambda_samples, axis=0)
    incidence = {
        e: _incidence_value(float(median_lambda[i]), e, entry_strata, stratum_sizes)
        for i, e in enumerate(entry_ids)
    }
    order = sorted(entry_ids, key=lambda e: (-incidence[e], e))
    return tuple(order)


def oracle_incidence_intervals(
    lambda_samples: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> dict[str, tuple[float, float]]:
    """Per-entry (2.5, 97.5) percentile incidence interval."""
    intervals: dict[str, tuple[float, float]] = {}
    for i, e in enumerate(entry_ids):
        total_size = float(sum(stratum_sizes[s] for s in entry_strata[e]))
        draws = lambda_samples[:, i] * total_size
        intervals[e] = (
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        )
    return intervals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_oracle_incidence.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`
Expected: no errors.

```bash
git add engine/verify/__init__.py engine/verify/oracle.py tests/unit/test_oracle_incidence.py
git commit -m "feat(verify): oracle incidence-ranking re-derivation (Plan 8d D1)"
```

---

### Task 2: Oracle PL ranking via Bradley-Terry MM (D2)

**Files:**
- Modify: `engine/verify/oracle.py`
- Test: `tests/unit/test_oracle_pl.py`

**Interfaces:**
- Consumes: a respondent-rankings array `(n_respondents, n_entries)` of ranks (1 = best, equal rank = tie) — the same shape the engine persists as `vote_rankings.npy` (Task 5).
- Produces:
  - `def _pairwise_wins_halfcredit(rankings: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]` — returns `(wins, comparisons)`: `wins[i]` = total credit of entry i (1 per strict win + 0.5 per tie), `comparisons[i, j]` = number of respondents comparing i and j.
  - `def oracle_pl_ranking_mm(rankings: npt.NDArray[np.float64], entry_ids: tuple[str, ...], max_iter: int = 1000, tol: float = 1e-9) -> tuple[str, ...]` — BT worths via MM (Hunter 2004), best→worst, ties broken by entry id.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_oracle_pl.py`:

```python
"""Tests for the oracle Bradley-Terry MM PL re-derivation (Plan 8d D2)."""
from __future__ import annotations

import numpy as np

from engine.verify.oracle import _pairwise_wins_halfcredit, oracle_pl_ranking_mm


def test_pairwise_wins_halfcredit() -> None:
    # 3 respondents, entries A,B,C. r0,r1: A>B>C ; r2: A=B>C
    rankings = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.5, 1.5, 3.0]])
    wins, comparisons = _pairwise_wins_halfcredit(rankings)
    # A: beats B twice (r0,r1) + 0.5 tie (r2) = 2.5 ; beats C 3 times = 3 -> 5.5
    assert wins[0] == 5.5
    # B: 0.5 tie with A (r2) + beats C 3 times = 3.5
    assert wins[1] == 3.5
    # C: never wins -> 0
    assert wins[2] == 0.0
    # every pair compared by all 3 respondents
    assert comparisons[0, 1] == 3
    assert comparisons[0, 2] == 3


def test_mm_recovers_strict_order() -> None:
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (6, 1))
    ranking = oracle_pl_ranking_mm(rankings, ("A", "B", "C"))
    assert ranking == ("A", "B", "C")


def test_mm_agrees_with_known_dominance() -> None:
    # A dominates; B and C close. A must be first.
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
        ]
    )
    ranking = oracle_pl_ranking_mm(rankings, ("A", "B", "C"))
    assert ranking[0] == "A"


def test_mm_is_deterministic() -> None:
    rankings = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0]])
    r1 = oracle_pl_ranking_mm(rankings, ("A", "B", "C"))
    r2 = oracle_pl_ranking_mm(rankings, ("A", "B", "C"))
    assert r1 == r2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_oracle_pl.py -v`
Expected: FAIL — `ImportError: cannot import name '_pairwise_wins_halfcredit'`.

- [ ] **Step 3: Write the implementation**

Append to `engine/verify/oracle.py`:

```python
def _pairwise_wins_halfcredit(
    rankings: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-entry win credit (1 strict / 0.5 tie) and pairwise comparison counts.

    Different tie handling than the engine's Davidson nu model: ties are split
    as half a win each.  This is intentional independence for the cross-check.
    """
    n_resp, n = rankings.shape
    wins = np.zeros(n, dtype=np.float64)
    comparisons = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = rankings[:, i] - rankings[:, j]
            i_wins = float(np.sum(diff < 0.0))  # lower rank = preferred
            j_wins = float(np.sum(diff > 0.0))
            ties = float(np.sum(diff == 0.0))
            wins[i] += i_wins + 0.5 * ties
            wins[j] += j_wins + 0.5 * ties
            comparisons[i, j] = float(n_resp)
            comparisons[j, i] = float(n_resp)
    return wins, comparisons


def oracle_pl_ranking_mm(
    rankings: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> tuple[str, ...]:
    """Bradley-Terry worths via MM/fixed-point (Hunter 2004), then rank.

    Update: pi_i <- w_i / sum_j!=i  n_ij / (pi_i + pi_j) ; renormalize to sum 1.
    A different optimizer family than the engine's scipy.optimize L-BFGS-B.
    """
    n = len(entry_ids)
    wins, comparisons = _pairwise_wins_halfcredit(rankings)
    pi = np.full(n, 1.0 / n, dtype=np.float64)
    eps = 1e-12
    for _ in range(max_iter):
        denom = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                denom[i] += comparisons[i, j] / (pi[i] + pi[j] + eps)
        new_pi = wins / (denom + eps)
        total = float(np.sum(new_pi))
        if total <= 0.0:
            break
        new_pi = new_pi / total
        if float(np.max(np.abs(new_pi - pi))) < tol:
            pi = new_pi
            break
        pi = new_pi
    worths = {e: float(pi[i]) for i, e in enumerate(entry_ids)}
    order = sorted(entry_ids, key=lambda e: (-worths[e], e))
    return tuple(order)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_oracle_pl.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`

```bash
git add engine/verify/oracle.py tests/unit/test_oracle_pl.py
git commit -m "feat(verify): oracle PL ranking via Bradley-Terry MM (Plan 8d D2)"
```

---

### Task 3: Oracle σ_u DerSimonian-Laird surrogate (D3)

**Files:**
- Modify: `engine/verify/oracle.py`
- Test: `tests/unit/test_oracle_sigma_u.py`

**Interfaces:**
- Consumes: `poisson_flat` unpooled λ samples `(n_samples, n_entries)`.
- Produces:
  - `def oracle_sigma_u_surrogate(lambda_samples: npt.NDArray[np.float64]) -> float` — DerSimonian-Laird between-entry SD of log-λ (random-effects moment estimator). Returns 0.0 when fewer than 2 entries or non-positive denominator.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_oracle_sigma_u.py`:

```python
"""Tests for the oracle DSL sigma_u surrogate (Plan 8d D3)."""
from __future__ import annotations

import numpy as np

from engine.verify.oracle import oracle_sigma_u_surrogate


def test_sigma_u_zero_when_entries_identical() -> None:
    # All entries share the same log-lambda distribution -> no between-entry SD.
    rng = np.random.default_rng(0)
    base = np.exp(rng.normal(0.0, 0.05, size=(500, 1)))
    lambda_samples = np.repeat(base, 4, axis=1)
    sigma = oracle_sigma_u_surrogate(lambda_samples)
    assert sigma < 0.1


def test_sigma_u_positive_when_entries_spread() -> None:
    # Entries centered at very different log-rates -> positive between-entry SD.
    rng = np.random.default_rng(1)
    cols = []
    for center in (-2.0, -1.0, 0.0, 1.0, 2.0):
        cols.append(np.exp(rng.normal(center, 0.05, size=600)))
    lambda_samples = np.column_stack(cols)
    sigma = oracle_sigma_u_surrogate(lambda_samples)
    assert sigma > 0.5


def test_sigma_u_single_entry_returns_zero() -> None:
    lambda_samples = np.array([[0.1], [0.2], [0.15]])
    assert oracle_sigma_u_surrogate(lambda_samples) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_oracle_sigma_u.py -v`
Expected: FAIL — `ImportError: cannot import name 'oracle_sigma_u_surrogate'`.

- [ ] **Step 3: Write the implementation**

Append to `engine/verify/oracle.py`:

```python
def oracle_sigma_u_surrogate(lambda_samples: npt.NDArray[np.float64]) -> float:
    """DerSimonian-Laird between-entry SD of log-lambda (random-effects moment).

    A closed-form surrogate for the engine's NUTS HalfNormal sigma_u posterior:
    y_e = median(log lambda_e), v_e = var(log lambda_e) (within-entry sampling
    variance).  tau^2 = max(0, (Q - (k-1)) / C) with DSL weights w_e = 1/v_e.
    No MCMC, no scipy.optimize.  Computed on the UNPOOLED poisson_flat samples
    so it is an independent estimate of the pooling SD, not a re-read of the
    hierarchical posterior.
    """
    k = lambda_samples.shape[1]
    if k < 2:
        return 0.0
    log_lambda = np.log(np.clip(lambda_samples, 1e-12, None))
    y = np.median(log_lambda, axis=0)
    v = np.var(log_lambda, axis=0, ddof=1)
    v = np.clip(v, 1e-12, None)
    w = 1.0 / v
    sum_w = float(np.sum(w))
    y_bar = float(np.sum(w * y) / sum_w)
    q = float(np.sum(w * (y - y_bar) ** 2))
    c = sum_w - float(np.sum(w**2)) / sum_w
    if c <= 0.0:
        return 0.0
    tau2 = max(0.0, (q - (k - 1)) / c)
    return float(np.sqrt(tau2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_oracle_sigma_u.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`

```bash
git add engine/verify/oracle.py tests/unit/test_oracle_sigma_u.py
git commit -m "feat(verify): oracle sigma_u DerSimonian-Laird surrogate (Plan 8d D3)"
```

---

### Task 4: Tolerances, verdict dataclasses, comparison helpers

**Files:**
- Modify: `engine/verify/oracle.py`
- Test: `tests/unit/test_oracle_verdict.py`

**Interfaces:**
- Produces:
  - Constants: `ORACLE_TAU_INCIDENCE: float = 0.95`, `ORACLE_TAU_PL: float = 0.70`, `ORACLE_SIGMA_U_BAND: float = 0.75`.
  - `def kendall_tau(rank_a: tuple[str, ...], rank_b: tuple[str, ...]) -> float` — Kendall-τ between two rankings over the same entry set (order-aligned by entry id). Returns 1.0 for identical, raises `ValueError` on mismatched entry sets.
  - `def top_tier_set(ranking: tuple[str, ...]) -> frozenset[str]` — the top third (matching the engine's tiering: `n<=3 -> top 1`, else `n//3`).
  - `@dataclass(frozen=True, slots=True) class OracleDeliverable` — `name: str`, `status: str` (`"PASS"`/`"FAIL"`/`"SKIP"`), `metric: str`, `detail: str`.
  - `@dataclass(frozen=True, slots=True) class OracleVerdict` — `deliverables: tuple[OracleDeliverable, ...]`, with a property `provisional: bool` (True iff any deliverable status == `"FAIL"`).
  - `def compare_ranking(name: str, engine_ranking: tuple[str, ...], oracle_ranking: tuple[str, ...], tau_floor: float) -> OracleDeliverable` — PASS iff Kendall-τ ≥ floor AND top-tier sets equal.
  - `def compare_sigma_u(engine_sigma_u: float, oracle_sigma_u: float, band: float) -> OracleDeliverable` — PASS iff `abs(engine - oracle) <= band`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_oracle_verdict.py`:

```python
"""Tests for oracle tolerances, comparisons, and verdict (Plan 8d Task 4)."""
from __future__ import annotations

import pytest

from engine.verify.oracle import (
    ORACLE_SIGMA_U_BAND,
    ORACLE_TAU_INCIDENCE,
    ORACLE_TAU_PL,
    OracleVerdict,
    compare_ranking,
    compare_sigma_u,
    kendall_tau,
    top_tier_set,
)


def test_tolerance_constants() -> None:
    assert ORACLE_TAU_INCIDENCE == 0.95
    assert ORACLE_TAU_PL == 0.70
    assert ORACLE_SIGMA_U_BAND == 0.75


def test_kendall_tau_identical_is_one() -> None:
    assert kendall_tau(("A", "B", "C"), ("A", "B", "C")) == 1.0


def test_kendall_tau_reversed_is_negative_one() -> None:
    assert kendall_tau(("A", "B", "C"), ("C", "B", "A")) == -1.0


def test_kendall_tau_mismatched_sets_raises() -> None:
    with pytest.raises(ValueError):
        kendall_tau(("A", "B"), ("A", "C"))


def test_top_tier_set() -> None:
    assert top_tier_set(("A", "B", "C", "D", "E", "F")) == {"A", "B"}
    assert top_tier_set(("A", "B")) == {"A"}


def test_compare_ranking_pass_when_identical() -> None:
    d = compare_ranking(
        "incidence", ("A", "B", "C", "D"), ("A", "B", "C", "D"), ORACLE_TAU_INCIDENCE
    )
    assert d.status == "PASS"


def test_compare_ranking_fail_when_reversed() -> None:
    d = compare_ranking(
        "incidence", ("A", "B", "C", "D"), ("D", "C", "B", "A"), ORACLE_TAU_INCIDENCE
    )
    assert d.status == "FAIL"


def test_compare_sigma_u_pass_and_fail() -> None:
    assert compare_sigma_u(1.0, 1.5, ORACLE_SIGMA_U_BAND).status == "PASS"
    assert compare_sigma_u(1.0, 2.0, ORACLE_SIGMA_U_BAND).status == "FAIL"


def test_verdict_provisional_iff_any_fail() -> None:
    ok = compare_sigma_u(1.0, 1.0, ORACLE_SIGMA_U_BAND)
    bad = compare_sigma_u(1.0, 9.0, ORACLE_SIGMA_U_BAND)
    assert OracleVerdict(deliverables=(ok,)).provisional is False
    assert OracleVerdict(deliverables=(ok, bad)).provisional is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_oracle_verdict.py -v`
Expected: FAIL — `ImportError: cannot import name 'OracleVerdict'`.

- [ ] **Step 3: Write the implementation**

Append to `engine/verify/oracle.py` (add `from dataclasses import dataclass` and `from scipy.stats import kendalltau` to the import block at the top):

```python
ORACLE_TAU_INCIDENCE: float = 0.95
ORACLE_TAU_PL: float = 0.70
ORACLE_SIGMA_U_BAND: float = 0.75


@dataclass(frozen=True, slots=True)
class OracleDeliverable:
    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    metric: str
    detail: str


@dataclass(frozen=True, slots=True)
class OracleVerdict:
    deliverables: tuple[OracleDeliverable, ...]

    @property
    def provisional(self) -> bool:
        return any(d.status == "FAIL" for d in self.deliverables)


def kendall_tau(rank_a: tuple[str, ...], rank_b: tuple[str, ...]) -> float:
    """Kendall tau between two rankings over the same entry set."""
    if set(rank_a) != set(rank_b):
        raise ValueError("kendall_tau: rankings cover different entry sets")
    pos_a = {e: i for i, e in enumerate(rank_a)}
    pos_b = {e: i for i, e in enumerate(rank_b)}
    entries = sorted(rank_a)
    va = [pos_a[e] for e in entries]
    vb = [pos_b[e] for e in entries]
    return float(kendalltau(va, vb)[0])


def top_tier_set(ranking: tuple[str, ...]) -> frozenset[str]:
    """Top tier (matches engine tiering: n<=3 -> 1, else n//3)."""
    n = len(ranking)
    size = 1 if n <= 3 else n // 3
    return frozenset(ranking[:size])


def compare_ranking(
    name: str,
    engine_ranking: tuple[str, ...],
    oracle_ranking: tuple[str, ...],
    tau_floor: float,
) -> OracleDeliverable:
    tau = kendall_tau(engine_ranking, oracle_ranking)
    tiers_match = top_tier_set(engine_ranking) == top_tier_set(oracle_ranking)
    status = "PASS" if (tau >= tau_floor and tiers_match) else "FAIL"
    return OracleDeliverable(
        name=name,
        status=status,
        metric=f"kendall_tau={tau:.3f} (floor {tau_floor:.2f})",
        detail=f"top_tier_match={tiers_match}",
    )


def compare_sigma_u(
    engine_sigma_u: float,
    oracle_sigma_u: float,
    band: float,
) -> OracleDeliverable:
    delta = abs(engine_sigma_u - oracle_sigma_u)
    status = "PASS" if delta <= band else "FAIL"
    return OracleDeliverable(
        name="sigma_u",
        status=status,
        metric=f"|delta|={delta:.3f} (band {band:.2f})",
        detail=f"engine={engine_sigma_u:.3f} oracle={oracle_sigma_u:.3f}",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_oracle_verdict.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`

```bash
git add engine/verify/oracle.py tests/unit/test_oracle_verdict.py
git commit -m "feat(verify): oracle tolerances, comparisons, and verdict (Plan 8d T4)"
```

---

### Task 5: Persist the engine deliverables the oracle compares against

**Files:**
- Modify: `engine/cli/pipeline.py` (`decide_real`)
- Test: `tests/unit/test_decide_oracle_artifacts.py`

The oracle's D1 needs the engine's *own* incidence ranking + per-entry CI persisted (currently only kappa is written), and D2 needs the ballot matrix persisted so re-verification is self-contained (no xlsx needed). This task adds a small helper and writes two artifacts in `decide_real`.

**Interfaces:**
- Consumes: `inference_result.lambda_samples`, `inference_result.entry_ids`, `entry_strata`, `stratum_sizes`, `vote_posterior.entries`, `vote_data.rankings` — all already in scope in `decide_real`. The incidence helper reuses the engine's own `_ranks_from_incidence` from `engine.decide.concordance`.
- Produces:
  - `def build_incidence_ranking_artifact(lambda_samples: npt.NDArray[np.float64], entry_ids: tuple[str, ...], common: list[str], entry_strata: dict[str, tuple[str, ...]], stratum_sizes: dict[str, int]) -> dict[str, object]` (in `pipeline.py`, near `build_vote_pl_summary`) — returns `{"ranking": [...best->worst...], "incidence_median": {e: float}, "incidence_ci": {e: [lo, hi]}}`.
  - Two new files in `cycle/"results"`: `incidence_ranking.json`, `vote_rankings.npy` (+ `vote_entry_ids.json`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_decide_oracle_artifacts.py`:

```python
"""Tests for build_incidence_ranking_artifact (Plan 8d Task 5)."""
from __future__ import annotations

import numpy as np

from engine.cli.pipeline import build_incidence_ranking_artifact


def test_incidence_artifact_shape_and_order() -> None:
    lambda_samples = np.array([[0.5, 0.1], [0.6, 0.2], [0.4, 0.15]])
    entry_ids = ("A", "B")
    common = ["A", "B"]
    entry_strata = {"A": ("security",), "B": ("security",)}
    stratum_sizes = {"security": 10}
    art = build_incidence_ranking_artifact(
        lambda_samples, entry_ids, common, entry_strata, stratum_sizes
    )
    assert art["ranking"] == ["A", "B"]
    assert set(art["incidence_median"].keys()) == {"A", "B"}  # type: ignore[union-attr]
    for lo, hi in art["incidence_ci"].values():  # type: ignore[union-attr]
        assert lo <= hi


def test_incidence_artifact_is_json_serializable() -> None:
    import json

    lambda_samples = np.array([[0.5, 0.1], [0.6, 0.2]])
    art = build_incidence_ranking_artifact(
        lambda_samples, ("A", "B"), ["A", "B"],
        {"A": ("s",), "B": ("s",)}, {"s": 5},
    )
    restored = json.loads(json.dumps(art))
    assert restored["ranking"] == ["A", "B"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_decide_oracle_artifacts.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_incidence_ranking_artifact'`.

- [ ] **Step 3: Add the helper to `engine/cli/pipeline.py`**

Add immediately after `build_vote_pl_summary`:

```python
def build_incidence_ranking_artifact(
    lambda_samples: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    common: list[str],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
) -> dict[str, object]:
    """Persist the engine's incidence deliverable for the oracle to check (8d).

    Uses the engine's OWN _ranks_from_incidence on the median lambda vector so
    the persisted ranking is exactly the engine's method; the oracle re-derives
    independently and compares.
    """
    from engine.decide.concordance import _ranks_from_incidence

    inf_idx = {e: i for i, e in enumerate(entry_ids)}
    median_lambda = np.median(lambda_samples, axis=0)
    point_ranks = _ranks_from_incidence(
        median_lambda, inf_idx, common, entry_strata, stratum_sizes
    )
    ranking = [e for _, e in sorted(zip(point_ranks, common, strict=True))]

    incidence_median: dict[str, float] = {}
    incidence_ci: dict[str, list[float]] = {}
    for e in common:
        total_size = float(sum(stratum_sizes[s] for s in entry_strata[e]))
        draws = lambda_samples[:, inf_idx[e]] * total_size
        incidence_median[e] = float(np.median(draws))
        incidence_ci[e] = [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ]
    return {
        "ranking": ranking,
        "incidence_median": incidence_median,
        "incidence_ci": incidence_ci,
    }
```

- [ ] **Step 4: Run the helper test to verify it passes**

Run: `uv run pytest tests/unit/test_decide_oracle_artifacts.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the artifacts in `decide_real`**

In `decide_real`, the common-entry set and `out_dir` are available where the vote PL summary is built. After the `write_decide_artifacts(...)` call and the existing `vote_plackett_luce.json` write, add:

```python
        # Plan 8d: persist the engine deliverables the oracle checks against,
        # and the ballot matrix so re-verification is self-contained.
        _common = [e for e in inference_result.entry_ids if e in set(vote_posterior.entries)]
        incidence_artifact = build_incidence_ranking_artifact(
            inference_result.lambda_samples,
            inference_result.entry_ids,
            _common,
            entry_strata,
            stratum_sizes,
        )
        (out_dir / "incidence_ranking.json").write_text(
            json.dumps(incidence_artifact, indent=2, sort_keys=True)
        )
        np.save(out_dir / "vote_rankings.npy", vote_data.rankings)
        (out_dir / "vote_entry_ids.json").write_text(
            json.dumps(list(vote_data.entry_ids), indent=2)
        )
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (no regressions). If a decide/synthetic test asserts exact `results/` directory contents, update it to allow the new files (a correction, not a weakening); if unsure, set DONE_WITH_CONCERNS and explain.

- [ ] **Step 7: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`

```bash
git add engine/cli/pipeline.py tests/unit/test_decide_oracle_artifacts.py
git commit -m "feat(decide): persist incidence ranking + ballot matrix for oracle (Plan 8d T5)"
```

---

### Task 6: `run_oracle` orchestrator + `verify-oracle` CLI

**Files:**
- Create: `engine/verify/check.py`
- Modify: `engine/cli/pipeline.py` (add `verify-oracle` click command)
- Test: `tests/unit/test_oracle_run.py`

**Interfaces:**
- Consumes: all oracle functions + dataclasses from `engine/verify/oracle.py`; artifacts under `<cycle>/results/` and `<cycle>/infer/`; `<cycle>/classify/labeled_incidents.json`.
- Produces:
  - `def run_oracle(cycle: Path) -> OracleVerdict` — loads artifacts, runs D1/D2/D3 (SKIP a deliverable whose inputs are absent), writes `<cycle>/results/oracle_report.json`, returns the verdict.
  - `def _verdict_to_dict(v: OracleVerdict) -> dict[str, object]` — JSON payload: `{"provisional": bool, "deliverables": [{"name","status","metric","detail"}, ...]}`.
  - A click command `verify-oracle` taking `--cycle` that calls `run_oracle` and echoes the verdict.

**Artifact map** (paths `run_oracle` reads):
- `<cycle>/infer/lambda_samples.npy` + `<cycle>/infer/inference_summary.json` (`entry_ids`)
- `<cycle>/infer/robustness_poisson_flat_lambda.npy` (D3 surrogate input; SKIP D3 if absent)
- `<cycle>/results/robustness_spread.json` → the hierarchical `sigma_u` (search `robustness` list for the spec whose `sigma_u` is not null; SKIP D3 if none)
- `<cycle>/results/incidence_ranking.json` (`ranking`) — engine D1
- `<cycle>/results/vote_rankings.npy` + `<cycle>/results/vote_entry_ids.json` — D2 ballots
- `<cycle>/results/vote_plackett_luce.json` (`ranking`) — engine D2
- `<cycle>/classify/labeled_incidents.json` — to rebuild `entry_strata`/`stratum_sizes` via the same logic decide uses

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_oracle_run.py`:

```python
"""Integration test for run_oracle over a synthetic cycle dir (Plan 8d Task 6)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.verify.check import run_oracle


def _write_cycle(tmp: Path) -> Path:
    cycle = tmp / "cycle"
    (cycle / "infer").mkdir(parents=True)
    (cycle / "results").mkdir(parents=True)
    (cycle / "classify").mkdir(parents=True)

    entry_ids = ["A", "B", "C", "D"]
    # A>B>C>D incidence: descending lambda, single stratum size 10
    rng = np.random.default_rng(0)
    centers = {"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}
    cols = [np.clip(rng.normal(centers[e], 0.01, 400), 1e-6, None) for e in entry_ids]
    lam = np.column_stack(cols)
    np.save(cycle / "infer" / "lambda_samples.npy", lam)
    np.save(cycle / "infer" / "robustness_poisson_flat_lambda.npy", lam)
    (cycle / "infer" / "inference_summary.json").write_text(
        json.dumps({"entry_ids": entry_ids})
    )

    # labeled incidents: every entry in stratum 'security', 10 docs
    labeled = [{"entry_id": e, "stratum": "security"} for e in entry_ids for _ in range(10)]
    (cycle / "classify" / "labeled_incidents.json").write_text(json.dumps(labeled))

    # engine incidence ranking (matches the lambda order)
    (cycle / "results" / "incidence_ranking.json").write_text(
        json.dumps({"ranking": ["A", "B", "C", "D"]})
    )
    # robustness spread carrying a hierarchical sigma_u
    (cycle / "results" / "robustness_spread.json").write_text(
        json.dumps(
            {
                "primary": {"spec_name": "kappa", "sigma_u": None},
                "robustness": [
                    {"spec_name": "hierarchical_pooling", "sigma_u": 0.9},
                ],
            }
        )
    )
    # ballots: all respondents rank A>B>C>D
    rankings = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (12, 1))
    np.save(cycle / "results" / "vote_rankings.npy", rankings)
    (cycle / "results" / "vote_entry_ids.json").write_text(json.dumps(entry_ids))
    (cycle / "results" / "vote_plackett_luce.json").write_text(
        json.dumps({"ranking": ["A", "B", "C", "D"]})
    )
    return cycle


def test_run_oracle_all_pass(tmp_path: Path) -> None:
    cycle = _write_cycle(tmp_path)
    verdict = run_oracle(cycle)
    names = {d.name: d.status for d in verdict.deliverables}
    assert names["incidence"] == "PASS"
    assert names["plackett_luce"] == "PASS"
    assert names["sigma_u"] == "PASS"
    assert verdict.provisional is False
    # report written
    report = json.loads((cycle / "results" / "oracle_report.json").read_text())
    assert report["provisional"] is False
    assert len(report["deliverables"]) == 3


def test_run_oracle_flags_provisional_on_bad_engine_ranking(tmp_path: Path) -> None:
    cycle = _write_cycle(tmp_path)
    # corrupt the engine incidence ranking -> oracle disagrees -> FAIL
    (cycle / "results" / "incidence_ranking.json").write_text(
        json.dumps({"ranking": ["D", "C", "B", "A"]})
    )
    verdict = run_oracle(cycle)
    names = {d.name: d.status for d in verdict.deliverables}
    assert names["incidence"] == "FAIL"
    assert verdict.provisional is True


def test_run_oracle_skips_missing_sigma_u(tmp_path: Path) -> None:
    cycle = _write_cycle(tmp_path)
    (cycle / "infer" / "robustness_poisson_flat_lambda.npy").unlink()
    verdict = run_oracle(cycle)
    names = {d.name: d.status for d in verdict.deliverables}
    assert names["sigma_u"] == "SKIP"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_oracle_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.verify.check'`.

- [ ] **Step 3: Write the implementation**

Create `engine/verify/check.py`:

```python
"""Oracle orchestration: load a cycle's artifacts, run D1/D2/D3, write report."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from engine.verify.oracle import (
    ORACLE_SIGMA_U_BAND,
    ORACLE_TAU_INCIDENCE,
    ORACLE_TAU_PL,
    OracleDeliverable,
    OracleVerdict,
    compare_ranking,
    compare_sigma_u,
    oracle_incidence_ranking,
    oracle_pl_ranking_mm,
    oracle_sigma_u_surrogate,
)


def _build_strata(
    labeled: list[dict[str, object]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    entry_strata_sets: dict[str, set[str]] = defaultdict(set)
    stratum_counts: dict[str, int] = defaultdict(int)
    for item in labeled:
        eid = str(item.get("entry_id", ""))
        stratum = str(item.get("stratum", "default"))
        entry_strata_sets[eid].add(stratum)
        stratum_counts[stratum] += 1
    entry_strata = {e: tuple(sorted(ss)) for e, ss in entry_strata_sets.items()}
    stratum_sizes = {s: max(c, 1) for s, c in stratum_counts.items()}
    return entry_strata, stratum_sizes


def _hierarchical_sigma_u(spread: dict[str, object]) -> float | None:
    robustness = spread.get("robustness", [])
    if not isinstance(robustness, list):
        return None
    for spec in robustness:
        if isinstance(spec, dict) and spec.get("sigma_u") is not None:
            return float(spec["sigma_u"])
    return None


def run_oracle(cycle: Path) -> OracleVerdict:
    """Re-derive D1/D2/D3 independently and compare to the engine's output."""
    infer = cycle / "infer"
    results = cycle / "results"
    deliverables: list[OracleDeliverable] = []

    # --- D1: incidence ranking ---
    inc_path = results / "incidence_ranking.json"
    lam_path = infer / "lambda_samples.npy"
    summ_path = infer / "inference_summary.json"
    labeled_path = cycle / "classify" / "labeled_incidents.json"
    if inc_path.exists() and lam_path.exists() and summ_path.exists() and labeled_path.exists():
        engine_ranking = tuple(json.loads(inc_path.read_text())["ranking"])
        lam = np.load(lam_path, allow_pickle=False)
        entry_ids = tuple(json.loads(summ_path.read_text())["entry_ids"])
        labeled = json.loads(labeled_path.read_text())
        entry_strata, stratum_sizes = _build_strata(labeled)
        common = tuple(e for e in entry_ids if e in set(engine_ranking))
        oracle_ranking = oracle_incidence_ranking(
            lam[:, [entry_ids.index(e) for e in common]],
            common,
            entry_strata,
            stratum_sizes,
        )
        deliverables.append(
            compare_ranking("incidence", engine_ranking, oracle_ranking, ORACLE_TAU_INCIDENCE)
        )
    else:
        deliverables.append(
            OracleDeliverable("incidence", "SKIP", "n/a", "missing inputs")
        )

    # --- D2: PL vote ranking ---
    pl_path = results / "vote_plackett_luce.json"
    ballots_path = results / "vote_rankings.npy"
    vote_ids_path = results / "vote_entry_ids.json"
    if pl_path.exists() and ballots_path.exists() and vote_ids_path.exists():
        engine_pl = tuple(json.loads(pl_path.read_text())["ranking"])
        ballots = np.load(ballots_path, allow_pickle=False)
        vote_ids = tuple(json.loads(vote_ids_path.read_text()))
        oracle_pl = oracle_pl_ranking_mm(ballots, vote_ids)
        deliverables.append(
            compare_ranking("plackett_luce", engine_pl, oracle_pl, ORACLE_TAU_PL)
        )
    else:
        deliverables.append(
            OracleDeliverable("plackett_luce", "SKIP", "n/a", "missing inputs")
        )

    # --- D3: sigma_u surrogate ---
    spread_path = results / "robustness_spread.json"
    flat_path = infer / "robustness_poisson_flat_lambda.npy"
    engine_sigma = (
        _hierarchical_sigma_u(json.loads(spread_path.read_text()))
        if spread_path.exists()
        else None
    )
    if engine_sigma is not None and flat_path.exists():
        flat_lam = np.load(flat_path, allow_pickle=False)
        oracle_sigma = oracle_sigma_u_surrogate(flat_lam)
        deliverables.append(
            compare_sigma_u(engine_sigma, oracle_sigma, ORACLE_SIGMA_U_BAND)
        )
    else:
        deliverables.append(
            OracleDeliverable("sigma_u", "SKIP", "n/a", "missing inputs")
        )

    verdict = OracleVerdict(deliverables=tuple(deliverables))
    (results / "oracle_report.json").write_text(
        json.dumps(_verdict_to_dict(verdict), indent=2, sort_keys=True)
    )
    return verdict


def _verdict_to_dict(verdict: OracleVerdict) -> dict[str, object]:
    return {
        "provisional": verdict.provisional,
        "deliverables": [
            {"name": d.name, "status": d.status, "metric": d.metric, "detail": d.detail}
            for d in verdict.deliverables
        ],
    }
```

> Note: `lam[:, [entry_ids.index(e) for e in common]]` re-slices λ columns to the `common` order so the oracle's `entry_ids` argument and the λ columns align. `entry_ids` here is from `inference_summary.json` and matches the λ column order (the engine writes them together).

Add the CLI command to `engine/cli/pipeline.py` (near the other click commands; reuse the module's `click` import):

```python
@click.command("verify-oracle")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
def verify_oracle_cmd(cycle: Path) -> None:
    """Run the independent consistency-check oracle over a completed cycle."""
    from engine.verify.check import run_oracle

    verdict = run_oracle(cycle)
    for d in verdict.deliverables:
        click.echo(f"[{d.status}] {d.name}: {d.metric} ; {d.detail}")
    click.echo(f"PROVISIONAL: {verdict.provisional}")
```

Register `verify_oracle_cmd` on the CLI group exactly as the sibling commands are registered (find the `cli.add_command(...)` block or the group decorator pattern in `pipeline.py` and follow it).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_oracle_run.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, type-check, full suite, commit**

Run: `uv run ruff check . && uv run mypy engine tests && uv run pytest -q`

```bash
git add engine/verify/check.py engine/cli/pipeline.py tests/unit/test_oracle_run.py
git commit -m "feat(verify): run_oracle orchestrator + verify-oracle CLI (Plan 8d T6)"
```

---

### Task 7: Wire the oracle into decide + render the verdict / PROVISIONAL banner

**Files:**
- Modify: `engine/cli/pipeline.py` (`decide_real` runs the oracle; `report_cmd` loads `oracle_report.json`)
- Modify: `engine/report/render.py` (`ReportInputs.oracle_verdict`, `_render_oracle_lines`, banner)
- Test: `tests/unit/test_render_oracle.py`

**Interfaces:**
- Consumes: `run_oracle` (Task 6); `oracle_report.json` schema (`{"provisional": bool, "deliverables": [...]}`).
- Produces: `ReportInputs.oracle_verdict: dict[str, object] | None = None`; `_render_oracle_lines(oracle: dict[str, object] | None) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_render_oracle.py`:

```python
"""Tests for oracle render section + PROVISIONAL banner (Plan 8d Task 7)."""
from __future__ import annotations

from engine.report.render import _render_oracle_lines


def test_render_oracle_none_is_empty() -> None:
    assert _render_oracle_lines(None) == []


def test_render_oracle_pass_shows_section_no_banner() -> None:
    oracle: dict[str, object] = {
        "provisional": False,
        "deliverables": [
            {"name": "incidence", "status": "PASS", "metric": "kendall_tau=1.000", "detail": ""},
            {"name": "sigma_u", "status": "PASS", "metric": "|delta|=0.10", "detail": ""},
        ],
    }
    text = "".join(_render_oracle_lines(oracle))
    assert "Oracle Consistency Check" in text
    assert "incidence" in text
    assert "PASS" in text
    assert "PROVISIONAL" not in text


def test_render_oracle_fail_shows_provisional_banner() -> None:
    oracle: dict[str, object] = {
        "provisional": True,
        "deliverables": [
            {"name": "incidence", "status": "FAIL", "metric": "kendall_tau=-1.0", "detail": ""},
            {"name": "sigma_u", "status": "SKIP", "metric": "n/a", "detail": "missing"},
        ],
    }
    text = "".join(_render_oracle_lines(oracle))
    assert "PROVISIONAL" in text
    assert "incidence" in text
    assert "FAIL" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_render_oracle.py -v`
Expected: FAIL — `ImportError: cannot import name '_render_oracle_lines'`.

- [ ] **Step 3: Add the renderer + field + wiring**

In `engine/report/render.py`, add the field to `ReportInputs` (last trailing optional field):

```python
    oracle_verdict: dict[str, object] | None = None
```

Add the renderer (after `_render_vote_pl_lines`):

```python
def _render_oracle_lines(oracle: dict[str, object] | None) -> list[str]:
    """Render the independent oracle consistency check + provisional banner."""
    if oracle is None:
        return []
    lines: list[str] = ["\n## Oracle Consistency Check\n"]
    if oracle.get("provisional") is True:
        lines.append(
            "**PROVISIONAL: the independent oracle disagrees with the engine on "
            "one or more deliverables (see FAIL rows). Treat results as "
            "un-cross-checked.**\n"
        )
    deliverables = oracle.get("deliverables", [])
    if isinstance(deliverables, list):
        for d in deliverables:
            if isinstance(d, dict):
                lines.append(
                    f"- [{d.get('status')}] {d.get('name')}: {d.get('metric')}\n"
                )
    return lines
```

In `render_report`, after the vote-PL block, add:

```python
    lines.extend(_render_oracle_lines(inputs.oracle_verdict))
```

In `engine/cli/pipeline.py` `report_cmd`, after loading `vote_plackett_luce.json`, load the oracle report:

```python
        oracle_path = results_dir / "oracle_report.json"
        oracle_verdict: dict[str, object] | None = None
        if oracle_path.exists():
            oracle_verdict = json.loads(oracle_path.read_text())
```

Add `oracle_verdict=oracle_verdict,` to the `ReportInputs(...)` construction.

In `decide_real`, after the Task 5 artifact writes, run the oracle so the cycle self-checks:

```python
        # Plan 8d: run the independent oracle and persist its verdict.
        from engine.verify.check import run_oracle
        oracle_verdict = run_oracle(cycle)
        if oracle_verdict.provisional:
            click.echo(
                "Oracle consistency check: PROVISIONAL (one or more deliverables "
                "disagree)"
            )
        else:
            click.echo("Oracle consistency check: PASS")
```

- [ ] **Step 4: Run the render tests, then the full suite**

Run: `uv run pytest tests/unit/test_render_oracle.py -v`
Expected: PASS (3 passed).

Run: `uv run pytest -q`
Expected: PASS. The synthetic decide path now runs `run_oracle`; if a synthetic cycle lacks the oracle inputs, deliverables SKIP (not FAIL) and `provisional` stays False — confirm no synthetic/decide test breaks. If one asserts exact `results/` contents, allow `oracle_report.json` (correction).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`

```bash
git add engine/report/render.py engine/cli/pipeline.py tests/unit/test_render_oracle.py
git commit -m "feat(report): wire oracle into decide + render verdict/PROVISIONAL banner (Plan 8d T7)"
```

---

## Lessons capture (after the final review, before finishing)

Append a `## Plan 8d` section to `docs/superpowers/plans/LESSONS-rarr.md`: the chosen oracle methods (BT-MM half-credit ties; DSL moment surrogate on poisson_flat), the right-sized gate (provisional flag, no Merkle/signer), the module-constant tolerances (and that 8f must review them against real numbers), the new persisted artifacts (`incidence_ranking.json`, `vote_rankings.npy`, `oracle_report.json`), and any oracle corrections. 8e/8f read this first.

---

## Self-Review (completed by plan author)

**1. Spec coverage (§5.7).** Independent re-derivation by a different optimizer family: D2 BT-MM (Task 2), D3 DSL surrogate (Task 3), D1 incidence (Task 1). Per-deliverable pre-declared tolerances: Task 4 constants. "no R, ever": pure numpy/scipy. Gate: right-sized to a provisional flag (8d decision) — Tasks 6–7; the Merkle/signer machinery is deliberately OUT of scope (documented in Global Constraints). Oracle does not import engine estimator code: enforced in Global Constraints + Task 1 docstring.

**2. Placeholder scan.** No TBD/TODO; every code step has complete code; tests assert concretely; constants concrete.

**3. Type consistency.** `OracleDeliverable`/`OracleVerdict` defined in Task 4, consumed in Task 6; `run_oracle(cycle: Path) -> OracleVerdict` and `_verdict_to_dict` keys (`provisional`, `deliverables` with `name/status/metric/detail`) match what `_render_oracle_lines` reads in Task 7; `incidence_ranking.json` `"ranking"` key (Task 5) matches what `run_oracle` reads (Task 6); `vote_rankings.npy` shape `(n_resp, n_entries)` matches `oracle_pl_ranking_mm` input (Task 2). `ReportInputs.oracle_verdict` default `None` mirrors `vote_plackett_luce`.

**4. Known scope notes.** Incidence (D1) shares λ inputs with the engine, so it is a *re-implementation* check (catches ranking/strata arithmetic bugs) rather than a different-method check — documented in the deliverables table. σ_u band and τ floors are coarse build-time constants reviewed in 8f. Kappa scalar is intentionally not gated (function of two checked rankings; MCSE-dominated).
