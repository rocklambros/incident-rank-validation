# Plan 8c — Tie-Aware Plackett-Luce (Davidson) Vote Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tie-aware paired-comparison vote-aggregation model (Davidson 1970) as a vote-side robustness lens, so the headline kappa-concordance result can be checked against a principled alternative to the bootstrap-mean-rank vote ranking.

**Architecture:** The 1–5 importance ballots (`VoteData.rankings`, averaged-rank ties) are reduced to per-respondent pairwise win/tie counts, then a Davidson tie-aware Bradley-Terry model is fit by penalized MLE on pinned scipy/numpy (no new deps). A respondent-level bootstrap (reusing the existing seed discipline) characterizes ranking stability. The worth ranking rides the existing `SpecResult.extra_rankings` channel (which already round-trips through `robustness_spread.json`) into the report; richer diagnostics persist to a new auditable `vote_plackett_luce.json` and render in a dedicated report section. The kappa-concordance result stays primary — this is a robustness lens, not a replacement.

**Tech Stack:** Python 3.12, numpy==2.1.3, scipy==1.15.0 (`scipy.optimize.minimize`, `scipy.stats.kendalltau`), click. No new dependencies.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the RARR spec, the LESSONS-rarr.md ledger, and the user's standing instructions.

- **NO new dependencies.** Only numpy==2.1.3 and scipy==1.15.0. Do NOT add choix, pymc, bradley-terry, or any ranking library. The Davidson MLE is hand-rolled on `scipy.optimize.minimize`.
- **kappa-concordance stays PRIMARY.** The Plackett-Luce/Davidson model is a vote-side robustness lens. Do not alter the primary concordance computation, tier boundaries, or flag logic.
- **NO new manifest field.** Reuse the existing `manifest.prng_seed`. Adding a `PreregManifest` field would perturb the frozen 2026 lock; 8c must keep it byte-stable. Module-level constants carry the ridge and bootstrap count.
- **CPU-deterministic & reproducible.** The point fit starts from a fixed `x0` (zeros) via L-BFGS-B (deterministic). The bootstrap is seeded from `manifest.prng_seed`. No `Math.random`, no wall-clock.
- **CI gate commands (run the EXACT commands, whole-repo, before every commit):** `uv run ruff check .` → `uv run mypy engine tests` (engine AND tests) → and before any push the FULL `uv run pytest -q` (NOT a `-k` subset — shared-code changes have repo-wide reach; a `-k` subset slipped a real failure to CI in 8b remediation).
- **mypy is strict.** Every test function needs `-> None`; helpers need typed args/returns; `res.x` from scipy is `Any` and MUST be cast (`np.asarray(res.x, dtype=np.float64)`) to satisfy `warn_return_any`. scipy is already in `[[tool.mypy.overrides]] ignore_missing_imports=true`, so the imports themselves are fine.
- **No AI attribution in any commit message or GitHub-visible content.** Commit messages read as if written entirely by the repository owner.
- **Branch:** all work lands on `plan7/engine-upgrade-recall-pl` (PR #22). Do not branch or merge.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `engine/vote/plackett_luce.py` | Davidson tie-aware paired-comparison model: pairwise-count extraction, penalized MLE fit, respondent bootstrap, stability scalars. Pure scipy/numpy, no I/O. | Create (Tasks 1–2) |
| `engine/cli/pipeline.py` (`decide_real`, `report_cmd`) | Compute the Davidson fit + drop-ties sensitivity + bootstrap from `vote_data`; attach the worth ranking to `primary_spec_result.extra_rankings`; build & write `vote_plackett_luce.json`; load it in `report_cmd` and pass to render. | Modify (Tasks 3–4) |
| `engine/report/render.py` (`ReportInputs`, `_render_robustness_lines`, `render_report`, new `_render_vote_pl_lines`) | Render the PL worth ranking in the robustness section and a dedicated Vote-Aggregation section (tie parameter, Kendall-τ concordance vs mean-rank, drop-ties sensitivity, bootstrap stability). | Modify (Task 4) |
| `tests/unit/test_plackett_luce.py` | Unit tests for the model (recovery, ties, separation, drop-ties, bootstrap determinism, stability). | Create (Tasks 1–2) |
| `tests/unit/test_vote_pl_summary.py` | Unit tests for the `build_vote_pl_summary` helper. | Create (Task 3) |
| `tests/unit/test_render_vote_pl.py` | Unit tests for the render extensions. | Create (Task 4) |

## Why Davidson, not tie-extended PL or Rao-Kupper

The spec names "tie-aware Plackett-Luce (Davidson/Rao-Kupper)." Plackett-Luce proper is a model of *strict total orderings*; the ballots here are 1–5 importance scores, i.e. partial orders saturated with ties (averaged ranks make tied items share a rank). The **Davidson (1970)** extension of Bradley-Terry is the standard tie-aware aggregation for exactly this data shape, it has a single interpretable tie parameter ν, and it **reduces to Bradley-Terry/Plackett-Luce worths as ν→0** — so the "drop-ties" sensitivity is the same estimator with the tie term removed. Rao-Kupper is an equivalent threshold-parametrized alternative; implementing one tie-aware model is sufficient for a robustness lens (YAGNI). This choice is documented in the module docstring (Task 1, Step 3).

---

### Task 1: Davidson tie-aware paired-comparison fit

**Files:**
- Create: `engine/vote/plackett_luce.py`
- Test: `tests/unit/test_plackett_luce.py`

**Interfaces:**
- Consumes: `VoteData.rankings` shape `(n_respondents, n_entries)` — float ranks, 1 = best, **averaged ranks for ties** (so `row[i] == row[j]` ⟺ respondent tied entries i and j). `entry_ids: tuple[str, ...]` aligned to the columns.
- Produces:
  - `DEFAULT_RIDGE: float = 1e-3`
  - `@dataclass(frozen=True, slots=True) class DavidsonFit` with fields: `entries: tuple[str, ...]`, `log_worths: dict[str, float]`, `worths: dict[str, float]`, `tie_param: float`, `ranking: tuple[str, ...]` (best→worst), `converged: bool`.
  - `def fit_davidson(rankings: npt.NDArray[np.float64], entry_ids: tuple[str, ...], ridge: float = DEFAULT_RIDGE, include_ties: bool = True) -> DavidsonFit`
  - `def _pairwise_counts(rankings: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]` (returns `(wins, ties)`; `wins[a, b]` = number of respondents preferring entry a over entry b; `ties[i, j]` for `i < j` = number who tied them).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_plackett_luce.py`:

```python
"""Tests for the Davidson tie-aware paired-comparison vote model (Plan 8c)."""
from __future__ import annotations

import numpy as np

from engine.vote.plackett_luce import (
    DEFAULT_RIDGE,
    DavidsonFit,
    _pairwise_counts,
    fit_davidson,
)


def test_pairwise_counts_wins_and_ties() -> None:
    # 3 respondents, 3 entries (A, B, C).
    # r0: A>B>C (ranks 1,2,3); r1: A>B>C; r2: A tied B, both above C (ranks 1.5,1.5,3)
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.5, 1.5, 3.0],
        ]
    )
    wins, ties = _pairwise_counts(rankings)
    # A preferred over B by r0, r1 (r2 tied) -> wins[0,1] == 2
    assert wins[0, 1] == 2.0
    assert wins[1, 0] == 0.0
    # A and B both preferred over C by all 3 -> wins[0,2] == 3, wins[1,2] == 3
    assert wins[0, 2] == 3.0
    assert wins[1, 2] == 3.0
    # A tied B once (r2) -> ties[0,1] == 1 (upper triangle)
    assert ties[0, 1] == 1.0
    # no C ties
    assert ties[0, 2] == 0.0


def test_fit_recovers_strict_order() -> None:
    # Everyone ranks A>B>C strictly -> worth(A) > worth(B) > worth(C).
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    fit = fit_davidson(rankings, ("A", "B", "C"))
    assert fit.ranking == ("A", "B", "C")
    assert fit.log_worths["A"] > fit.log_worths["B"] > fit.log_worths["C"]
    assert fit.converged is True


def test_fit_tie_parameter_positive_when_ties_present() -> None:
    # Mix of strict and tied ballots -> tie parameter is finite and > 0.
    rankings = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.5, 1.5],
            [1.5, 1.5],
        ]
    )
    fit = fit_davidson(rankings, ("A", "B"))
    assert np.isfinite(fit.tie_param)
    assert fit.tie_param > 0.0


def test_fit_handles_complete_separation_without_diverging() -> None:
    # A is ranked strictly above everything by all respondents (separation):
    # worth(A) stays finite because of the ridge penalty.
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
            [1.0, 2.0, 3.0],
        ]
    )
    fit = fit_davidson(rankings, ("A", "B", "C"))
    assert np.isfinite(fit.log_worths["A"])
    assert fit.worths["A"] > fit.worths["B"]
    assert fit.ranking[0] == "A"


def test_fit_drop_ties_reduces_to_bradley_terry() -> None:
    # include_ties=False ignores tied pairs; strict order still recovered,
    # tie_param reported as 0.0.
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.5, 1.5, 3.0],
        ]
    )
    fit = fit_davidson(rankings, ("A", "B", "C"), include_ties=False)
    assert fit.ranking == ("A", "B", "C")
    assert fit.tie_param == 0.0


def test_default_ridge_value() -> None:
    assert DEFAULT_RIDGE == 1e-3


def test_fit_returns_dataclass_with_aligned_entries() -> None:
    rankings = np.tile(np.array([1.0, 2.0]), (3, 1))
    fit = fit_davidson(rankings, ("X", "Y"))
    assert isinstance(fit, DavidsonFit)
    assert fit.entries == ("X", "Y")
    assert set(fit.worths.keys()) == {"X", "Y"}
    assert set(fit.ranking) == {"X", "Y"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_plackett_luce.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.vote.plackett_luce'`.

- [ ] **Step 3: Write the implementation**

Create `engine/vote/plackett_luce.py`:

```python
"""Davidson (1970) tie-aware paired-comparison vote model (Plan 8c).

The OWASP importance ballots are 1-5 scores over the candidate entries, which
average-rank into partial orders saturated with ties.  Plackett-Luce proper
models strict total orderings, so we use the Davidson tie-aware extension of
Bradley-Terry, the standard aggregation for tie-heavy paired-comparison data.

For an unordered pair (i, j) with worths pi_i = exp(theta_i), pi_j = exp(theta_j),
a tie parameter nu = exp(gamma) >= 0, and s = sqrt(pi_i * pi_j):

    P(i beats j) = pi_i / (pi_i + pi_j + nu * s)
    P(j beats i) = pi_j / (pi_i + pi_j + nu * s)
    P(i ties  j) = nu * s / (pi_i + pi_j + nu * s)

As nu -> 0 this reduces to Bradley-Terry / Plackett-Luce worths, so
``include_ties=False`` (drop tied pairs, fix nu = 0) is a free drop-ties
sensitivity lens.  Worths are identified only up to scale and complete
separation drives a dominant entry's worth to infinity; a single L2 ridge on
theta makes the penalized negative log-likelihood strictly convex AND bounds
separation.  Rao-Kupper is an equivalent threshold-parametrized tie model; one
tie-aware model is sufficient for a robustness lens (YAGNI).

CPU-deterministic: the point fit starts from theta = 0, gamma = 0 via L-BFGS-B.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

DEFAULT_RIDGE: float = 1e-3
# gamma = log(nu) is bounded so that all-strict ballots (which drive nu -> 0)
# do not stall the optimizer at gamma -> -inf.  nu in [~2e-9, ~4.9e8].
_GAMMA_BOUND: float = 20.0


@dataclass(frozen=True, slots=True)
class DavidsonFit:
    """Point fit of the Davidson tie-aware model."""

    entries: tuple[str, ...]
    log_worths: dict[str, float]
    worths: dict[str, float]
    tie_param: float
    ranking: tuple[str, ...]  # best -> worst by worth
    converged: bool


def _pairwise_counts(
    rankings: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reduce respondent rank vectors to pairwise win/tie counts.

    ``rankings`` is (n_respondents, n_entries); lower rank = more preferred,
    equal rank = tie.  Returns ``(wins, ties)`` where ``wins[a, b]`` counts
    respondents preferring entry a over entry b and ``ties[i, j]`` (for i < j)
    counts respondents who tied them.
    """
    _, n = rankings.shape
    wins = np.zeros((n, n), dtype=np.float64)
    ties = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = rankings[:, i] - rankings[:, j]
            wins[i, j] = float(np.sum(diff < 0.0))  # i preferred (lower rank)
            wins[j, i] = float(np.sum(diff > 0.0))  # j preferred
            ties[i, j] = float(np.sum(diff == 0.0))
    return wins, ties


def _neg_log_likelihood(
    params: npt.NDArray[np.float64],
    wins: npt.NDArray[np.float64],
    ties: npt.NDArray[np.float64],
    iu: npt.NDArray[np.intp],
    ju: npt.NDArray[np.intp],
    ridge: float,
    include_ties: bool,
) -> float:
    """Penalized negative log-likelihood of the Davidson model.

    ``iu``/``ju`` are the upper-triangle (i < j) pair indices.  When
    ``include_ties`` is True the last element of ``params`` is gamma = log(nu).
    """
    n = wins.shape[0]
    theta = params[:n]
    ti = theta[iu]
    tj = theta[ju]
    wins_ij = wins[iu, ju]
    wins_ji = wins[ju, iu]
    if include_ties:
        gamma = float(params[n])
        log_s = 0.5 * (ti + tj)
        denom = np.exp(ti) + np.exp(tj) + np.exp(gamma + log_s)
        log_denom = np.log(denom)
        ties_ij = ties[iu, ju]
        ll = (
            np.sum(wins_ij * (ti - log_denom))
            + np.sum(wins_ji * (tj - log_denom))
            + np.sum(ties_ij * (gamma + log_s - log_denom))
        )
    else:
        denom = np.exp(ti) + np.exp(tj)
        log_denom = np.log(denom)
        ll = np.sum(wins_ij * (ti - log_denom)) + np.sum(wins_ji * (tj - log_denom))
    penalty = ridge * float(np.sum(theta * theta))
    return float(-ll + penalty)


def fit_davidson(
    rankings: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    ridge: float = DEFAULT_RIDGE,
    include_ties: bool = True,
) -> DavidsonFit:
    """Fit the Davidson tie-aware model by ridge-penalized MLE (L-BFGS-B)."""
    n = len(entry_ids)
    if rankings.shape[1] != n:
        raise ValueError(
            f"entry_ids length {n} does not match rankings columns "
            f"{rankings.shape[1]}"
        )
    wins, ties = _pairwise_counts(rankings)
    iu, ju = np.triu_indices(n, k=1)

    if include_ties:
        x0 = np.zeros(n + 1, dtype=np.float64)
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * n + [
            (-_GAMMA_BOUND, _GAMMA_BOUND)
        ]
    else:
        x0 = np.zeros(n, dtype=np.float64)
        bounds = [(None, None)] * n

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(wins, ties, iu, ju, ridge, include_ties),
        method="L-BFGS-B",
        bounds=bounds,
    )
    x = np.asarray(result.x, dtype=np.float64)
    theta = x[:n]
    tie_param = float(np.exp(x[n])) if include_ties else 0.0

    log_worths = {entry_ids[k]: float(theta[k]) for k in range(n)}
    worths = {entry_ids[k]: float(np.exp(theta[k])) for k in range(n)}
    # Deterministic ranking: highest worth first, ties broken by entry id.
    order = sorted(range(n), key=lambda k: (-theta[k], entry_ids[k]))
    ranking = tuple(entry_ids[k] for k in order)

    return DavidsonFit(
        entries=entry_ids,
        log_worths=log_worths,
        worths=worths,
        tie_param=tie_param,
        ranking=ranking,
        converged=bool(result.success),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_plackett_luce.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`
Expected: no errors.

```bash
git add engine/vote/plackett_luce.py tests/unit/test_plackett_luce.py
git commit -m "feat(vote): add Davidson tie-aware paired-comparison fit (Plan 8c T1)"
```

---

### Task 2: Respondent bootstrap and ranking stability

**Files:**
- Modify: `engine/vote/plackett_luce.py`
- Test: `tests/unit/test_plackett_luce.py`

**Interfaces:**
- Consumes: `fit_davidson`, `DavidsonFit` (Task 1); `manifest.prng_seed` (an int) supplied by the caller in Task 3.
- Produces:
  - `N_BOOTSTRAP_DEFAULT: int = 2000`
  - `@dataclass(frozen=True, slots=True) class DavidsonPosterior` with fields: `entries: tuple[str, ...]`, `point_ranking: tuple[str, ...]`, `median_ranks: dict[str, float]`, `top5_frequency: dict[str, float]`, `mean_kendall_tau_vs_point: float`, `n_respondents: int`, `n_bootstrap: int`.
  - `def _ranking_to_rank_vector(ranking: tuple[str, ...], entry_ids: tuple[str, ...]) -> npt.NDArray[np.float64]`
  - `def bootstrap_davidson(rankings: npt.NDArray[np.float64], entry_ids: tuple[str, ...], n_bootstrap: int = N_BOOTSTRAP_DEFAULT, seed: int = 42, ridge: float = DEFAULT_RIDGE) -> DavidsonPosterior`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_plackett_luce.py`:

```python
from engine.vote.plackett_luce import (  # noqa: E402  (appended import block)
    N_BOOTSTRAP_DEFAULT,
    DavidsonPosterior,
    _ranking_to_rank_vector,
    bootstrap_davidson,
)


def test_ranking_to_rank_vector() -> None:
    # ranking best->worst is (B, A, C); rank vector is aligned to entry_ids order.
    vec = _ranking_to_rank_vector(("B", "A", "C"), ("A", "B", "C"))
    # A is 2nd, B is 1st, C is 3rd
    assert list(vec) == [2.0, 1.0, 3.0]


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (8, 1))
    p1 = bootstrap_davidson(rankings, ("A", "B", "C"), n_bootstrap=30, seed=42)
    p2 = bootstrap_davidson(rankings, ("A", "B", "C"), n_bootstrap=30, seed=42)
    assert p1.median_ranks == p2.median_ranks
    assert p1.top5_frequency == p2.top5_frequency
    assert p1.mean_kendall_tau_vs_point == p2.mean_kendall_tau_vs_point


def test_bootstrap_strong_signal_is_stable() -> None:
    # Everyone strictly ranks A>B>C: A is top in every resample.
    rankings = np.tile(np.array([1.0, 2.0, 3.0]), (12, 1))
    post = bootstrap_davidson(rankings, ("A", "B", "C"), n_bootstrap=50, seed=7)
    assert post.point_ranking == ("A", "B", "C")
    assert post.top5_frequency["A"] == 1.0
    assert post.median_ranks["A"] == 1.0
    # Kendall tau vs the point ranking is in [-1, 1] and high for clean data.
    assert -1.0 <= post.mean_kendall_tau_vs_point <= 1.0
    assert post.mean_kendall_tau_vs_point > 0.8


def test_bootstrap_reports_counts() -> None:
    rankings = np.tile(np.array([1.0, 2.0]), (6, 1))
    post = bootstrap_davidson(rankings, ("A", "B"), n_bootstrap=20, seed=1)
    assert isinstance(post, DavidsonPosterior)
    assert post.n_respondents == 6
    assert post.n_bootstrap == 20


def test_default_bootstrap_count() -> None:
    assert N_BOOTSTRAP_DEFAULT == 2000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_plackett_luce.py -k "bootstrap or rank_vector or bootstrap_count" -v`
Expected: FAIL — `ImportError: cannot import name 'bootstrap_davidson'`.

- [ ] **Step 3: Write the implementation**

Append to `engine/vote/plackett_luce.py` (add `from scipy.stats import kendalltau` to the imports at the top, next to `from scipy.optimize import minimize`):

```python
N_BOOTSTRAP_DEFAULT: int = 2000


@dataclass(frozen=True, slots=True)
class DavidsonPosterior:
    """Respondent-bootstrap stability of the Davidson worth ranking."""

    entries: tuple[str, ...]
    point_ranking: tuple[str, ...]
    median_ranks: dict[str, float]
    top5_frequency: dict[str, float]
    mean_kendall_tau_vs_point: float
    n_respondents: int
    n_bootstrap: int


def _ranking_to_rank_vector(
    ranking: tuple[str, ...],
    entry_ids: tuple[str, ...],
) -> npt.NDArray[np.float64]:
    """Map a best->worst ranking to a 1-based rank vector aligned to entry_ids."""
    pos = {entry: i + 1 for i, entry in enumerate(ranking)}
    return np.array([pos[entry] for entry in entry_ids], dtype=np.float64)


def bootstrap_davidson(
    rankings: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = 42,
    ridge: float = DEFAULT_RIDGE,
) -> DavidsonPosterior:
    """Bootstrap the Davidson worth ranking by resampling respondents.

    Each replicate draws n_respondents with replacement, refits the model, and
    records the worth ranking.  Stability is summarized by per-entry median
    bootstrap rank, top-5 membership frequency, and the mean Kendall tau between
    each replicate ranking and the full-sample (point) ranking.
    """
    n_resp = rankings.shape[0]
    point = fit_davidson(rankings, entry_ids, ridge=ridge, include_ties=True)
    point_vec = _ranking_to_rank_vector(point.ranking, entry_ids)

    rng = np.random.default_rng(seed)
    positions: dict[str, list[int]] = {entry: [] for entry in entry_ids}
    top5_count: dict[str, int] = {entry: 0 for entry in entry_ids}
    taus: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_resp, size=n_resp)
        sample = rankings[idx]
        fit = fit_davidson(sample, entry_ids, ridge=ridge, include_ties=True)
        for rank_pos, entry in enumerate(fit.ranking, start=1):
            positions[entry].append(rank_pos)
        for entry in fit.ranking[:5]:
            top5_count[entry] += 1
        tau = kendalltau(point_vec, _ranking_to_rank_vector(fit.ranking, entry_ids))[0]
        taus.append(float(tau))

    median_ranks = {
        entry: float(np.median(positions[entry])) for entry in entry_ids
    }
    top5_frequency = {
        entry: top5_count[entry] / n_bootstrap for entry in entry_ids
    }
    mean_tau = float(np.mean(taus)) if taus else 0.0

    return DavidsonPosterior(
        entries=entry_ids,
        point_ranking=point.ranking,
        median_ranks=median_ranks,
        top5_frequency=top5_frequency,
        mean_kendall_tau_vs_point=mean_tau,
        n_respondents=n_resp,
        n_bootstrap=n_bootstrap,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_plackett_luce.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`
Expected: no errors.

```bash
git add engine/vote/plackett_luce.py tests/unit/test_plackett_luce.py
git commit -m "feat(vote): add respondent bootstrap + ranking stability for Davidson model (Plan 8c T2)"
```

---

### Task 3: Wire the Davidson model into `decide_real`

**Files:**
- Modify: `engine/cli/pipeline.py` (add `build_vote_pl_summary` helper near `build_robustness_spread`; call it inside `decide_real`)
- Test: `tests/unit/test_vote_pl_summary.py`

**Interfaces:**
- Consumes: `fit_davidson`, `bootstrap_davidson`, `DavidsonPosterior` (Tasks 1–2); `from scipy.stats import kendalltau`; `vote_data.rankings`, `vote_data.entry_ids` (`load_vote_data`); `vote_posterior` (the existing `bootstrap_vote_ranks` result, whose ordinal mean-rank ranking is the comparison baseline); `manifest.prng_seed`; the existing `primary_spec_result` construction at `pipeline.py:553-558`.
- Produces:
  - `def build_vote_pl_summary(rankings: npt.NDArray[np.float64], entry_ids: tuple[str, ...], mean_rank_ranking: tuple[str, ...], seed: int, n_bootstrap: int = N_BOOTSTRAP_DEFAULT, ridge: float = DEFAULT_RIDGE) -> dict[str, object]` — the JSON-serializable diagnostics for `vote_plackett_luce.json`. Its `"ranking"` value is the Davidson worth ranking that Task 3 also attaches to `primary_spec_result.extra_rankings["plackett_luce"]`.

The summary dict has these keys (used by Task 4's render):
`"model"`, `"ridge"`, `"n_bootstrap"`, `"seed"`, `"n_respondents"`, `"entries"`, `"worths"`, `"tie_param"`, `"ranking"`, `"ranking_drop_ties"`, `"bootstrap_median_ranks"`, `"bootstrap_top5_frequency"`, `"mean_kendall_tau_vs_point"`, `"kendall_tau_vs_meanrank"`, `"kendall_tau_withties_vs_dropties"`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_vote_pl_summary.py`:

```python
"""Tests for build_vote_pl_summary (Plan 8c Task 3)."""
from __future__ import annotations

import numpy as np

from engine.cli.pipeline import build_vote_pl_summary


def test_summary_has_required_keys_and_valid_ranking() -> None:
    rankings = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.5, 1.5, 3.0],
        ]
    )
    entry_ids = ("A", "B", "C")
    summary = build_vote_pl_summary(
        rankings,
        entry_ids,
        mean_rank_ranking=("A", "B", "C"),
        seed=42,
        n_bootstrap=20,
    )
    required = {
        "model",
        "ridge",
        "n_bootstrap",
        "seed",
        "n_respondents",
        "entries",
        "worths",
        "tie_param",
        "ranking",
        "ranking_drop_ties",
        "bootstrap_median_ranks",
        "bootstrap_top5_frequency",
        "mean_kendall_tau_vs_point",
        "kendall_tau_vs_meanrank",
        "kendall_tau_withties_vs_dropties",
    }
    assert required.issubset(summary.keys())
    assert summary["model"] == "davidson_tie_aware"
    assert summary["n_respondents"] == 3
    assert summary["seed"] == 42
    # ranking is a permutation of the entries
    assert set(summary["ranking"]) == set(entry_ids)  # type: ignore[arg-type]
    # clean strict data: Davidson ranking matches the mean-rank ranking
    assert summary["kendall_tau_vs_meanrank"] == 1.0


def test_summary_is_json_serializable() -> None:
    import json

    rankings = np.tile(np.array([1.0, 2.0]), (5, 1))
    summary = build_vote_pl_summary(
        rankings, ("A", "B"), mean_rank_ranking=("A", "B"), seed=1, n_bootstrap=10
    )
    # Must round-trip through JSON without error.
    restored = json.loads(json.dumps(summary))
    assert restored["entries"] == ["A", "B"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_vote_pl_summary.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_vote_pl_summary'`.

- [ ] **Step 3: Add the helper to `engine/cli/pipeline.py`**

At the top of `engine/cli/pipeline.py`, add the imports (place beside the other `engine.vote` imports; if numpy/typing are not already module-level, add them):

```python
import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau

from engine.vote.plackett_luce import (
    DEFAULT_RIDGE,
    N_BOOTSTRAP_DEFAULT,
    bootstrap_davidson,
    fit_davidson,
)
```

> Note: `pipeline.py` already imports `np` (it calls `np.load`). Do NOT add a duplicate `import numpy as np`. Add only the names not already present.

Add the helper immediately after `build_robustness_spread` (after `pipeline.py:53`):

```python
def build_vote_pl_summary(
    rankings: npt.NDArray[np.float64],
    entry_ids: tuple[str, ...],
    mean_rank_ranking: tuple[str, ...],
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    ridge: float = DEFAULT_RIDGE,
) -> dict[str, object]:
    """Fit the Davidson tie-aware vote model and assemble JSON diagnostics.

    Computes the worth ranking (with ties), the drop-ties Bradley-Terry
    sensitivity, and the respondent bootstrap, plus Kendall-tau concordance
    against the bootstrap mean-rank vote ranking (the primary vote ranking) and
    against the drop-ties ranking.  The returned dict is the auditable
    ``vote_plackett_luce.json`` payload; its ``"ranking"`` is also attached to
    ``SpecResult.extra_rankings["plackett_luce"]`` by the caller.
    """
    from engine.vote.plackett_luce import _ranking_to_rank_vector

    fit = fit_davidson(rankings, entry_ids, ridge=ridge, include_ties=True)
    fit_drop = fit_davidson(rankings, entry_ids, ridge=ridge, include_ties=False)
    post = bootstrap_davidson(
        rankings, entry_ids, n_bootstrap=n_bootstrap, seed=seed, ridge=ridge
    )

    pl_vec = _ranking_to_rank_vector(fit.ranking, entry_ids)
    mean_vec = _ranking_to_rank_vector(mean_rank_ranking, entry_ids)
    drop_vec = _ranking_to_rank_vector(fit_drop.ranking, entry_ids)
    tau_meanrank = float(kendalltau(pl_vec, mean_vec)[0])
    tau_dropties = float(kendalltau(pl_vec, drop_vec)[0])

    return {
        "model": "davidson_tie_aware",
        "ridge": ridge,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_respondents": int(rankings.shape[0]),
        "entries": list(entry_ids),
        "worths": fit.worths,
        "tie_param": fit.tie_param,
        "ranking": list(fit.ranking),
        "ranking_drop_ties": list(fit_drop.ranking),
        "bootstrap_median_ranks": post.median_ranks,
        "bootstrap_top5_frequency": post.top5_frequency,
        "mean_kendall_tau_vs_point": post.mean_kendall_tau_vs_point,
        "kendall_tau_vs_meanrank": tau_meanrank,
        "kendall_tau_withties_vs_dropties": tau_dropties,
    }
```

- [ ] **Step 4: Run the helper test to verify it passes**

Run: `uv run pytest tests/unit/test_vote_pl_summary.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Call the helper inside `decide_real` and attach the ranking**

In `engine/cli/pipeline.py`, the primary spec result is built around `pipeline.py:553-558`:

```python
        primary_spec_result = SpecResult(
            spec_name=manifest.primary_spec,
            weighted_kappa_median=concordance.weighted_kappa_median,
            weighted_kappa_ci=concordance.weighted_kappa_ci,
            flags=concordance.flags,
        )
```

Immediately BEFORE that block, derive the mean-rank ranking from the existing `vote_posterior.median_ranks` and compute the PL summary:

```python
        # Plan 8c: Davidson tie-aware vote model as a vote-side robustness lens.
        # The bootstrap mean-rank ordering (ascending median rank, 1 = best) is
        # the primary vote ranking we compare the worth ranking against.
        mean_rank_ranking = tuple(
            sorted(
                vote_posterior.entries,
                key=lambda e: (vote_posterior.median_ranks[e], e),
            )
        )
        vote_pl_summary = build_vote_pl_summary(
            vote_data.rankings,
            vote_data.entry_ids,
            mean_rank_ranking=mean_rank_ranking,
            seed=manifest.prng_seed,
        )
        pl_ranking = tuple(str(e) for e in vote_pl_summary["ranking"])  # type: ignore[union-attr]
```

Then change the `primary_spec_result` construction to attach the ranking:

```python
        primary_spec_result = SpecResult(
            spec_name=manifest.primary_spec,
            weighted_kappa_median=concordance.weighted_kappa_median,
            weighted_kappa_ci=concordance.weighted_kappa_ci,
            flags=concordance.flags,
            extra_rankings={"plackett_luce": pl_ranking},
        )
```

After `write_decide_artifacts(...)` (the block at `pipeline.py:607-612`), persist the diagnostics artifact:

```python
        (out_dir / "vote_plackett_luce.json").write_text(
            json.dumps(vote_pl_summary, indent=2, sort_keys=True)
        )
```

- [ ] **Step 6: Run the full suite to verify decide still passes**

Run: `uv run pytest -q`
Expected: PASS (no regressions; synthetic-cycle decide tests still pass because they exercise the same `vote_data`/`vote_posterior` path).

> If any synthetic/decide test fails because the synthetic vote fixture has too few respondents/entries for a stable fit, the fit must still RUN (it is ridge-regularized and cannot diverge). If a test asserts on exact `report.md`/spread contents, update that oracle to include the new `extra_rankings` line rather than weakening it — escalate to the controller if unsure whether it is a correction or a weakening.

- [ ] **Step 7: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`
Expected: no errors.

```bash
git add engine/cli/pipeline.py tests/unit/test_vote_pl_summary.py
git commit -m "feat(decide): compute Davidson vote model + persist diagnostics in decide-real (Plan 8c T3)"
```

---

### Task 4: Render the PL worth ranking and the Vote-Aggregation section

**Files:**
- Modify: `engine/report/render.py` (`ReportInputs`, `_render_robustness_lines`, `render_report`, new `_render_vote_pl_lines`)
- Modify: `engine/cli/pipeline.py` (`report_cmd`: load `vote_plackett_luce.json`, pass into `ReportInputs`)
- Test: `tests/unit/test_render_vote_pl.py`

**Interfaces:**
- Consumes: `RobustnessSpread`, `SpecResult` with `extra_rankings` (Task 3); the `vote_plackett_luce.json` dict written in Task 3.
- Produces: `ReportInputs.vote_plackett_luce: dict[str, object] | None = None`; `_render_vote_pl_lines(vote_pl: dict[str, object] | None) -> list[str]`; an extended `_render_robustness_lines` that renders `extra_rankings`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_render_vote_pl.py`:

```python
"""Tests for Plan 8c render extensions (vote Plackett-Luce / Davidson)."""
from __future__ import annotations

from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult
from engine.report.render import _render_robustness_lines, _render_vote_pl_lines


def _spec_with_pl() -> SpecResult:
    return SpecResult(
        spec_name="kappa_concordance",
        weighted_kappa_median=0.20,
        weighted_kappa_ci=(-0.16, 0.57),
        flags=(),
        extra_rankings={"plackett_luce": ("A", "B", "C", "D", "E", "F")},
    )


def test_robustness_lines_render_extra_rankings() -> None:
    spread = RobustnessSpread(primary=_spec_with_pl(), robustness=())
    text = "".join(_render_robustness_lines(spread))
    assert "plackett_luce" in text
    # top entries are shown
    assert "A" in text and "B" in text


def test_render_vote_pl_lines_shows_diagnostics() -> None:
    vote_pl: dict[str, object] = {
        "model": "davidson_tie_aware",
        "ranking": ["A", "B", "C"],
        "tie_param": 1.37,
        "kendall_tau_vs_meanrank": 0.91,
        "kendall_tau_withties_vs_dropties": 0.86,
        "mean_kendall_tau_vs_point": 0.79,
        "n_respondents": 49,
        "n_bootstrap": 2000,
    }
    text = "".join(_render_vote_pl_lines(vote_pl))
    assert "Vote Aggregation" in text
    assert "Plackett-Luce" in text or "Davidson" in text
    assert "1.37" in text  # tie parameter
    assert "0.91" in text  # kendall tau vs mean-rank
    assert "0.79" in text  # bootstrap stability


def test_render_vote_pl_lines_none_is_empty() -> None:
    assert _render_vote_pl_lines(None) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_render_vote_pl.py -v`
Expected: FAIL — `ImportError: cannot import name '_render_vote_pl_lines'`.

- [ ] **Step 3: Extend `_render_robustness_lines` and add `_render_vote_pl_lines`**

In `engine/report/render.py`, the current `_render_robustness_lines` ends its per-spec loop with the `sigma_u` line (`render.py:48-50`). Inside the `for sr in all_specs:` loop, AFTER appending the kappa/sigma_u `line`, add rendering of `extra_rankings`:

```python
        lines.append(line + "\n")
        if sr.extra_rankings:
            for name, ranking in sorted(sr.extra_rankings.items()):
                top = " > ".join(ranking[:5])
                suffix = " > ..." if len(ranking) > 5 else ""
                lines.append(f"  - {name} ranking: {top}{suffix}\n")
```

> The existing `lines.append(line + "\n")` stays; the new `if sr.extra_rankings:` block goes directly after it, still inside the loop.

Add the new section renderer (place it after `_render_robustness_lines`):

```python
def _render_vote_pl_lines(vote_pl: dict[str, object] | None) -> list[str]:
    """Render the Davidson tie-aware vote-aggregation robustness section."""
    if vote_pl is None:
        return []
    lines: list[str] = ["\n## Vote Aggregation (Plackett-Luce / Davidson)\n"]
    ranking = vote_pl.get("ranking", [])
    if isinstance(ranking, list) and ranking:
        top = " > ".join(str(e) for e in ranking[:5])
        suffix = " > ..." if len(ranking) > 5 else ""
        lines.append(f"- Worth ranking: {top}{suffix}\n")
    tie_param = vote_pl.get("tie_param")
    if isinstance(tie_param, (int, float)):
        lines.append(f"- Tie parameter (nu): {tie_param:.2f}\n")
    tau_mean = vote_pl.get("kendall_tau_vs_meanrank")
    if isinstance(tau_mean, (int, float)):
        lines.append(f"- Kendall tau vs mean-rank vote ranking: {tau_mean:.2f}\n")
    tau_drop = vote_pl.get("kendall_tau_withties_vs_dropties")
    if isinstance(tau_drop, (int, float)):
        lines.append(f"- Kendall tau with-ties vs drop-ties: {tau_drop:.2f}\n")
    boot = vote_pl.get("mean_kendall_tau_vs_point")
    if isinstance(boot, (int, float)):
        lines.append(f"- Bootstrap stability (mean Kendall tau vs point): {boot:.2f}\n")
    return lines
```

- [ ] **Step 4: Wire the new field through `ReportInputs` and `render_report`**

In `engine/report/render.py`, add the field to `ReportInputs` (after `corpus_b_corroboration`, `render.py:31`):

```python
    vote_plackett_luce: dict[str, object] | None = None
```

In `render_report`, after the robustness block (`render.py:116-117`):

```python
    if inputs.robustness is not None:
        lines.extend(_render_robustness_lines(inputs.robustness))
    lines.extend(_render_vote_pl_lines(inputs.vote_plackett_luce))
```

- [ ] **Step 5: Load the artifact in `report_cmd` and pass it in**

In `engine/cli/pipeline.py` `report_cmd`, the robustness spread is loaded around `pipeline.py:752`. After that load (and before building `inputs`), load the PL artifact:

```python
        vote_pl_path = results_dir / "vote_plackett_luce.json"
        vote_pl_summary: dict[str, object] | None = None
        if vote_pl_path.exists():
            vote_pl_summary = json.loads(vote_pl_path.read_text())
```

Then add the field to the `ReportInputs(...)` construction (`pipeline.py:771-784`):

```python
            corpus_b_corroboration=corpus_b_corr,
            vote_plackett_luce=vote_pl_summary,
        )
```

- [ ] **Step 6: Run the render tests, then the full suite**

Run: `uv run pytest tests/unit/test_render_vote_pl.py -v`
Expected: PASS (3 passed).

Run: `uv run pytest -q`
Expected: PASS (full suite; no regressions).

- [ ] **Step 7: Lint, type-check, commit**

Run: `uv run ruff check . && uv run mypy engine tests`
Expected: no errors.

```bash
git add engine/report/render.py engine/cli/pipeline.py tests/unit/test_render_vote_pl.py
git commit -m "feat(report): render Davidson worth ranking + vote-aggregation section (Plan 8c T4)"
```

---

## Lessons capture (do this after the final review, before finishing)

Append a `## Plan 8c` section to `docs/superpowers/plans/LESSONS-rarr.md` recording: codebase realities that differed from this plan, the chosen Davidson-on-pairwise interpretation, the ridge/bootstrap constants, any oracle corrections, and carry-forwards for 8d (oracle) / 8e (cycle run). Per the standing rule, 8d/8e read this before their task lists.

---

## Self-Review (completed by plan author)

**1. Spec coverage.** RARR spec §5.5/§5.6 ("tie-aware Plackett-Luce as a vote-side robustness lens; respondent bootstrap; separation handling; report seed×tie-rule stability"): Task 1 (tie-aware Davidson fit + ridge separation handling), Task 2 (respondent bootstrap + stability scalar), Task 3 (wiring + drop-ties tie-rule sensitivity + Kendall-τ vs mean-rank + auditable artifact), Task 4 (render ranking + stability). kappa stays primary (Global Constraints; Task 3 leaves concordance untouched). No new deps / no new manifest field (Global Constraints).

**2. Placeholder scan.** No TBD/TODO; every code step shows full code; tests have real assertions; constants are concrete (`DEFAULT_RIDGE=1e-3`, `N_BOOTSTRAP_DEFAULT=2000`, `_GAMMA_BOUND=20.0`).

**3. Type consistency.** `DavidsonFit.ranking` / `DavidsonPosterior.point_ranking` are `tuple[str, ...]`; `_ranking_to_rank_vector(ranking, entry_ids)` defined in Task 2 and reused in Task 3; `build_vote_pl_summary` keys match exactly what `_render_vote_pl_lines` reads (`ranking`, `tie_param`, `kendall_tau_vs_meanrank`, `kendall_tau_withties_vs_dropties`, `mean_kendall_tau_vs_point`); `extra_rankings={"plackett_luce": ...}` key matches the render check in Task 4. `ReportInputs.vote_plackett_luce` default `None` mirrors `corpus_b_corroboration`.
