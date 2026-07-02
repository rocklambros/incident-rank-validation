# Plan 8b — Hierarchical Pooling Robustness Spec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a hierarchical partial-pooling model (`log λ_i = β0 + u_i`, `u_i ~ Normal(0, σ_u)`) as a declared robustness spec under the unchanged kappa primary, capturing and reporting σ_u with a prior-sensitivity sweep.

**Architecture:** A new `_run_hierarchical` NumPyro model in `engine/model/robustness.py` (non-centered parameterization, `λ` recorded as a deterministic site so the existing `lambda_samples` contract holds), dispatched by name from `run_robustness_inference`. `InferenceResult` and the Task-6 persist/reload plumbing gain a `sigma_u` field that flows into `SpecResult.sigma_u` and the report. A pure-function sensitivity sweep + prior-dominance decision rule support the σ_u robustness story. No new dependencies; CPU-only.

**Tech Stack:** Python 3.12, NumPyro/JAX (CPU only), NumPy, pytest, frozen dataclasses.

## Global Constraints

- **CPU only** — `assert jax.default_backend() == "cpu"` (mirror `_run_poisson_flat`). Never GPU.
- **kappa-concordance stays the PRIMARY** — hierarchical pooling is a *robustness spec only*; do not change `manifest.primary_spec`/`statistic` or the primary `run_inference`.
- **`λ` must be a recorded site** — use `numpyro.deterministic("lambda", ...)` so `samples["lambda"]` and the `(num_samples, num_entries)` shape contract hold for `lambda_samples`, `concordance`, and persistence.
- **CI gate commands (run the EXACT commands before every commit):** `uv sync --frozen --extra narrative`, `uv run ruff check .` (whole repo), **`uv run mypy engine tests`** (engine AND tests — test funcs need `-> None`, typed args; reuse `_make_manifest` from `tests/unit/test_prereg.py` for typed manifest construction), `uv run pytest -v`, `uv run semgrep --config .semgrep.yml --error engine/`.
- **Adding a manifest field requires schema_version ≥ 2 + updating the coverage-gate test** `test_verify_lock_raises_on_mutation` in `tests/unit/test_prereg.py` (add the field to the `mutations` dict; it is hash-affecting under v2, so NOT in `lock_invariant_fields`).
- **No new dependencies.** No AI attribution in commits. Frozen `cycles/2026/` byte-immutable.
- **Robustness specs do not run the primary's ESS gate** (`_run_poisson_flat` extracts diagnostics but does not raise `DiagnosticsFailure`); `_run_hierarchical` follows that pattern — capture σ_u/r-hat/ess diagnostics, do not hard-gate. (The primary's `_AUX_PARAMS` gate is untouched; parameterizing it is deferred to whenever hierarchical becomes primary, which this plan does NOT do.)

---

### Task 1: Add `sigma_u_hyperprior_scale` manifest field (schema v2)

**Files:**
- Modify: `engine/prereg/manifest.py:55-66` (add field after `goldset_hash`)
- Modify: `tests/unit/test_prereg.py` (coverage-gate `mutations` dict)
- Test: `tests/unit/test_manifest_sigma_u.py`

**Interfaces:**
- Consumes: existing `PreregManifest` (schema_version 1/2 from Plan 8a Task 2).
- Produces: `PreregManifest.sigma_u_hyperprior_scale: float | None = None` — the HalfNormal scale for the σ_u prior; only meaningful at schema_version ≥ 2; dropped from the v1 canonical form (so the frozen 2026 v1 lock stays byte-stable).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_manifest_sigma_u.py
from engine.prereg.lock import compute_lock_hash
from tests.unit.test_prereg import _make_manifest


def test_sigma_u_field_excluded_from_v1_canonical_form() -> None:
    m = _make_manifest()  # schema_version defaults to 1
    assert "sigma_u_hyperprior_scale" not in m.to_dict()


def test_sigma_u_under_v1_does_not_change_hash() -> None:
    base = _make_manifest()
    with_sigma = _make_manifest(sigma_u_hyperprior_scale=2.0)
    assert compute_lock_hash(base) == compute_lock_hash(with_sigma)


def test_sigma_u_included_and_hash_changes_at_v2() -> None:
    base = _make_manifest()
    v2 = _make_manifest(schema_version=2, sigma_u_hyperprior_scale=2.0)
    assert v2.to_dict()["sigma_u_hyperprior_scale"] == 2.0
    assert compute_lock_hash(base) != compute_lock_hash(v2)
```

(Confirm `_make_manifest` in `tests/unit/test_prereg.py` accepts `**overrides` / keyword passthrough; if it does not, extend it minimally to forward `schema_version` and `sigma_u_hyperprior_scale`, mirroring how it already forwards other fields.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_manifest_sigma_u.py -v`
Expected: FAIL — `PreregManifest` has no `sigma_u_hyperprior_scale`.

- [ ] **Step 3: Write minimal implementation**

In `engine/prereg/manifest.py`, add the field after `goldset_hash`:
```python
    schema_version: int = 1  # 1 = original field set; 2 = adds goldset_hash
    goldset_hash: str | None = None  # bound only when schema_version >= 2
    sigma_u_hyperprior_scale: float | None = None  # HalfNormal scale for sigma_u prior (schema >= 2)
```
Extend the v1-exclusion in `to_dict()`:
```python
        if self.schema_version == 1:
            result.pop("schema_version", None)
            result.pop("goldset_hash", None)
            result.pop("sigma_u_hyperprior_scale", None)
        return result
```
In `tests/unit/test_prereg.py`, add to the `mutations` dict in `test_verify_lock_raises_on_mutation` (it is hash-affecting, NOT lock-invariant):
```python
            "sigma_u_hyperprior_scale": 3.0,
```
(Place it among the hash-affecting entries; do NOT add it to `lock_invariant_fields`. The `manifest_fields == set(mutations.keys())` assertion then stays exhaustive. Note: the mutation runs on a default-schema_version=1 manifest, so mutating `sigma_u_hyperprior_scale` alone would NOT change a v1 hash — to keep this entry a real mismatch, also bump `schema_version` to 2 in the same `replace(...)` for this field, OR move `sigma_u_hyperprior_scale` into `lock_invariant_fields` alongside `goldset_hash` since both are v2-only. Choose the latter for consistency with `goldset_hash` and document it: at schema_version 1 these v2-only fields are lock-invariant by design.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_manifest_sigma_u.py tests/unit/test_prereg.py -v`
Expected: PASS.
Regression — the 2026 v1 lock canonical form is unchanged (sigma_u excluded at v1):
Run: `uv run pytest tests/unit -k "manifest or lock or prereg" -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/prereg/manifest.py tests/unit/test_prereg.py tests/unit/test_manifest_sigma_u.py
git commit -m "feat(prereg): add sigma_u_hyperprior_scale manifest field (schema v2)"
```

---

### Task 2: Add `sigma_u` to `InferenceResult` + persist/reload plumbing

**Files:**
- Modify: `engine/model/inference.py:35-42` (`InferenceResult` dataclass)
- Modify: `engine/cli/pipeline_executor.py` (`write_robustness_artifacts`)
- Modify: `engine/cli/pipeline.py` (decide-phase robustness reload + `SpecResult` build)
- Test: `tests/unit/test_inference_result_sigma_u.py`

**Interfaces:**
- Consumes: `InferenceResult` (Plan 8a fields), `write_robustness_artifacts(result, out_dir, spec_name)`, the decide-phase reload loop that builds `SpecResult`.
- Produces: `InferenceResult.sigma_u: float | None = None`; `write_robustness_artifacts` writes `"sigma_u": result.sigma_u` into the spec summary; the decide reload reads it and sets `SpecResult(sigma_u=...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_inference_result_sigma_u.py
import json
from pathlib import Path

import numpy as np

from engine.cli.pipeline_executor import write_robustness_artifacts
from engine.model.inference import InferenceResult


def test_inference_result_carries_sigma_u() -> None:
    r = InferenceResult(
        lambda_samples=np.zeros((4, 3)), entry_ids=("A", "B", "C"),
        r_hat={}, ess={}, divergences=0, num_warmup=1, num_samples=4,
        sigma_u=2.19,
    )
    assert r.sigma_u == 2.19


def test_write_robustness_artifacts_persists_sigma_u(tmp_path: Path) -> None:
    r = InferenceResult(
        lambda_samples=np.zeros((4, 3)), entry_ids=("A", "B", "C"),
        r_hat={}, ess={}, divergences=0, num_warmup=1, num_samples=4,
        sigma_u=2.19,
    )
    write_robustness_artifacts(r, tmp_path, "hierarchical_pooling")
    summary = json.loads((tmp_path / "robustness_hierarchical_pooling_summary.json").read_text())
    assert summary["sigma_u"] == 2.19
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_inference_result_sigma_u.py -v`
Expected: FAIL — `InferenceResult.__init__` has no `sigma_u`.

- [ ] **Step 3: Write minimal implementation**

In `engine/model/inference.py`, add the field to `InferenceResult` (after `num_samples`, with a default so all existing constructors keep working):
```python
@dataclass(frozen=True, slots=True)
class InferenceResult:
    lambda_samples: npt.NDArray[np.float64]  # shape (num_samples, num_entries)
    entry_ids: tuple[str, ...]
    r_hat: dict[str, float]
    ess: dict[str, float]
    divergences: int
    num_warmup: int
    num_samples: int
    sigma_u: float | None = None  # hierarchical pooling scale posterior median (Plan 8b)
```
In `engine/cli/pipeline_executor.py` `write_robustness_artifacts`, add `sigma_u` to the summary dict:
```python
    summary = {
        "spec_name": spec_name,
        "sigma_u": result.sigma_u,
        "entry_ids": list(result.entry_ids),
        "r_hat": result.r_hat,
        "ess": result.ess,
        "divergences": result.divergences,
        "num_warmup": result.num_warmup,
        "num_samples": result.num_samples,
    }
```
In `engine/cli/pipeline.py`, in the decide-phase robustness reload loop, carry `sigma_u` into the reconstructed `InferenceResult` and into the `SpecResult`:
```python
            r_inference = InferenceResult(
                lambda_samples=np.load(r_lambda_path),
                entry_ids=tuple(r_summary.get("entry_ids", [])),
                r_hat=r_summary.get("r_hat", {}),
                ess=r_summary.get("ess", {}),
                divergences=r_summary.get("divergences", 0),
                num_warmup=r_summary.get("num_warmup", 1000),
                num_samples=r_summary.get("num_samples", 2000),
                sigma_u=r_summary.get("sigma_u"),
            )
            ...
            robustness_results.append(SpecResult(
                spec_name=spec_name,
                weighted_kappa_median=r_concordance.weighted_kappa_median,
                weighted_kappa_ci=r_concordance.weighted_kappa_ci,
                flags=r_concordance.flags,
                sigma_u=r_inference.sigma_u,
            ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_inference_result_sigma_u.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k "inference or robustness or pipeline" -v` — Expected: PASS (sigma_u defaults to None everywhere else).

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/model/inference.py engine/cli/pipeline_executor.py engine/cli/pipeline.py tests/unit/test_inference_result_sigma_u.py
git commit -m "feat(model): carry sigma_u through InferenceResult + robustness persist/reload"
```

---

### Task 3: `_run_hierarchical` model + dispatch + σ_u capture

**Files:**
- Modify: `engine/model/robustness.py` (add `_run_hierarchical`, dispatch `"hierarchical_pooling"`)
- Test: `tests/unit/test_run_hierarchical.py`

**Interfaces:**
- Consumes: `run_robustness_inference(manifest, spec_name, measurable_entries, strata, observed_counts, stratum_sizes, calibration, overlap, ...)`; `_build_observation_arrays`, `_build_overlap_matrix`; `manifest.sigma_u_hyperprior_scale`, `manifest.prior_scale`, `manifest.prng_seed`.
- Produces: `run_robustness_inference(..., spec_name="hierarchical_pooling")` returns an `InferenceResult` whose `lambda_samples` come from a non-centered hierarchical model and whose `sigma_u` is the posterior median of the pooling scale.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_run_hierarchical.py
import numpy as np

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.overlap import OverlapWeights
from engine.model.robustness import run_robustness_inference
from tests.unit.test_prereg import _make_manifest


def _tiny_calibration(entries, stratum):
    recall = {(e, stratum): BetaPosterior(8.0, 2.0) for e in entries}
    precision = {(e, stratum): BetaPosterior(9.0, 1.0) for e in entries}
    return Calibration(recall=recall, precision=precision)


def test_hierarchical_returns_sigma_u_and_lambda_shape() -> None:
    entries = ("LLM01", "LLM02", "LLM03", "LLM04")
    stratum = "security"
    manifest = _make_manifest(schema_version=2, sigma_u_hyperprior_scale=1.0)
    observed = {(e, stratum): n for e, n in zip(entries, [50, 30, 10, 1])}
    result = run_robustness_inference(
        manifest=manifest, spec_name="hierarchical_pooling",
        measurable_entries=entries, strata=(stratum,),
        observed_counts=observed, stratum_sizes={stratum: 1000},
        calibration=_tiny_calibration(entries, stratum),
        overlap=OverlapWeights(weights={}),
        num_warmup=200, num_samples=200, num_chains=2,
    )
    # lambda contract preserved: (num_samples*num_chains, n_entries)
    assert result.lambda_samples.shape[1] == len(entries)
    # sigma_u captured as a positive scalar
    assert result.sigma_u is not None and result.sigma_u > 0.0
    assert np.all(result.lambda_samples > 0.0)  # exp(...) is strictly positive


def test_unknown_spec_still_raises() -> None:
    import pytest
    manifest = _make_manifest()
    with pytest.raises(ValueError, match="Unknown robustness spec"):
        run_robustness_inference(
            manifest=manifest, spec_name="nonexistent",
            measurable_entries=("A",), strata=("s",),
            observed_counts={("A", "s"): 1}, stratum_sizes={"s": 10},
            calibration=Calibration(recall={}, precision={}),
            overlap=OverlapWeights(weights={}),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_run_hierarchical.py -v`
Expected: FAIL — `Unknown robustness spec: hierarchical_pooling`.

- [ ] **Step 3: Write minimal implementation**

In `engine/model/robustness.py`, add the dispatch branch and the model. Add to `run_robustness_inference` (before the final `raise`):
```python
    if spec_name == "hierarchical_pooling":
        return _run_hierarchical(
            manifest, measurable_entries, strata, observed_counts,
            stratum_sizes, calibration, overlap, num_warmup, num_samples,
            num_chains,
        )
```
Add the function (mirroring `_run_poisson_flat`'s structure, diagnostics extraction, and return):
```python
def _run_hierarchical(
    manifest: PreregManifest,
    measurable_entries: tuple[str, ...],
    strata: tuple[str, ...],
    observed_counts: dict[tuple[str, str], int],
    stratum_sizes: dict[str, int],
    calibration: Calibration,
    overlap: OverlapWeights,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
) -> InferenceResult:
    assert jax.default_backend() == "cpu"

    n_entries = len(measurable_entries)
    obs, sizes, recall_a, recall_b, prec_a, prec_b = _build_observation_arrays(
        measurable_entries, strata, observed_counts, stratum_sizes, calibration,
    )
    W = _build_overlap_matrix(measurable_entries, overlap)

    beta0_loc = float(np.log(manifest.prior_scale))
    sigma_u_scale = float(manifest.sigma_u_hyperprior_scale or 1.0)
    conc_shape = manifest.concentration_shape
    conc_rate = manifest.concentration_rate

    def model(
        obs_data: npt.NDArray[np.float64],
        sizes_data: npt.NDArray[np.float64],
        recall_alpha: npt.NDArray[np.float64],
        recall_beta: npt.NDArray[np.float64],
        precision_alpha: npt.NDArray[np.float64],
        precision_beta: npt.NDArray[np.float64],
        W_data: npt.NDArray[np.float64],
    ) -> None:
        # Non-centered hierarchical: log lambda_i = beta0 + sigma_u * u_raw_i
        beta0 = numpyro.sample("beta0", dist.Normal(beta0_loc, 1.0))
        sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(sigma_u_scale))
        u_raw = numpyro.sample(
            "u_raw", dist.Normal(0.0, 1.0).expand([n_entries]).to_event(1),
        )
        lam = numpyro.deterministic("lambda", jnp.exp(beta0 + sigma_u * u_raw))

        recall = numpyro.sample(
            "recall", dist.Beta(jnp.array(recall_alpha), jnp.array(recall_beta)),
        )
        precision = numpyro.sample(
            "precision", dist.Beta(jnp.array(precision_alpha), jnp.array(precision_beta)),
        )
        concentration = numpyro.sample("concentration", dist.Gamma(conc_shape, conc_rate))

        true_rate = lam[:, None] * sizes_data[None, :]
        tp = true_rate * recall
        fp_rate = jnp.einsum("ij,js->is", jnp.array(W_data), true_rate * (1.0 - precision))
        expected = jnp.clip(tp + fp_rate, 1e-6, None)
        numpyro.sample(
            "obs", dist.NegativeBinomial2(mean=expected, concentration=concentration),
            obs=jnp.array(obs_data),
        )

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains, progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(manifest.prng_seed + 2000),
        obs, sizes, recall_a, recall_b, prec_a, prec_b, W,
    )

    samples = mcmc.get_samples()
    lambda_samples = np.asarray(samples["lambda"], dtype=np.float64)
    sigma_u_median = float(np.median(np.asarray(samples["sigma_u"], dtype=np.float64)))

    chain_samples = mcmc.get_samples(group_by_chain=True)
    summary: dict[str, Any] = numpyro.diagnostics.summary(chain_samples)
    r_hat_dict: dict[str, float] = {}
    ess_dict: dict[str, float] = {}
    for param_name, stats in summary.items():
        if "r_hat" in stats:
            vals = np.atleast_1d(stats["r_hat"])
            for idx, val in enumerate(vals.flat):
                key = f"{param_name}[{idx}]" if vals.size > 1 else param_name
                r_hat_dict[key] = float(val)
        if "n_eff" in stats:
            vals = np.atleast_1d(stats["n_eff"])
            for idx, val in enumerate(vals.flat):
                key = f"{param_name}[{idx}]" if vals.size > 1 else param_name
                ess_dict[key] = float(val)

    extra = mcmc.get_extra_fields()
    diverging = extra.get("diverging", np.array([]))
    divergences = int(np.asarray(diverging).sum())

    return InferenceResult(
        lambda_samples=lambda_samples,
        entry_ids=measurable_entries,
        r_hat=r_hat_dict,
        ess=ess_dict,
        divergences=divergences,
        num_warmup=num_warmup,
        num_samples=num_samples,
        sigma_u=sigma_u_median,
    )
```
(Imports `jnp`, `numpyro`, `dist`, `NUTS`, `MCMC`, `np`, `Any`, `_build_observation_arrays`, `_build_overlap_matrix`, `InferenceResult`, `Calibration`, `OverlapWeights`, `PreregManifest` are already present at the top of `robustness.py` from `_run_poisson_flat`; add none beyond what exists — verify and add only if missing.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_run_hierarchical.py -v`
Expected: PASS (small NUTS run; ~seconds on CPU).
Regression: `uv run pytest tests/unit -k robustness -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/model/robustness.py tests/unit/test_run_hierarchical.py
git commit -m "feat(model): hierarchical-pooling robustness spec with sigma_u capture"
```

---

### Task 4: σ_u sensitivity sweep + prior-dominance decision rule

**Files:**
- Create: `engine/model/sigma_u_sensitivity.py`
- Test: `tests/unit/test_sigma_u_sensitivity.py`

**Interfaces:**
- Consumes: `manifest.sigma_u_hyperprior_scale`, a callable that runs the hierarchical spec for a given scale and returns its `sigma_u` posterior median (so the sweep is testable without re-running NUTS).
- Produces: `sweep_sigma_u(scales: tuple[float, ...], run_fn) -> dict[float, float]` (scale → σ_u median); `is_prior_dominated(scales, sigma_u_by_scale, rel_tol: float = 0.25) -> bool` — True if σ_u tracks the prior scale (posterior ≈ prior, i.e. the data does not identify σ_u), which is the signal to *abandon pooling* and report independent rates.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sigma_u_sensitivity.py
from engine.model.sigma_u_sensitivity import is_prior_dominated, sweep_sigma_u


def test_sweep_collects_sigma_u_per_scale() -> None:
    calls = []

    def run_fn(scale: float) -> float:
        calls.append(scale)
        return 2.0  # data-dominated: same posterior regardless of prior

    out = sweep_sigma_u((0.5, 1.0, 2.0), run_fn)
    assert out == {0.5: 2.0, 1.0: 2.0, 2.0: 2.0}
    assert calls == [0.5, 1.0, 2.0]


def test_data_dominated_is_not_prior_dominated() -> None:
    # sigma_u stays ~2.0 across very different priors -> data identifies it.
    by_scale = {0.5: 2.0, 1.0: 2.0, 2.0: 2.05}
    assert is_prior_dominated((0.5, 1.0, 2.0), by_scale) is False


def test_prior_dominated_when_sigma_u_tracks_prior() -> None:
    # sigma_u ~ scale*const across priors -> posterior follows prior -> prior-dominated.
    by_scale = {0.5: 0.4, 1.0: 0.8, 2.0: 1.6}
    assert is_prior_dominated((0.5, 1.0, 2.0), by_scale) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sigma_u_sensitivity.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# engine/model/sigma_u_sensitivity.py
"""sigma_u prior-sensitivity sweep + prior-dominance decision rule (Plan 8b).

With ~20 groups sigma_u is weakly identified; if its posterior tracks the prior
scale, pooling is prior-driven and should be abandoned in favor of independent
per-entry rates (RARR design Sec 5.3).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sweep_sigma_u(
    scales: tuple[float, ...],
    run_fn: Callable[[float], float],
) -> dict[float, float]:
    """Run the hierarchical fit at each prior scale; return scale -> sigma_u median."""
    return {scale: run_fn(scale) for scale in scales}


def is_prior_dominated(
    scales: tuple[float, ...],
    sigma_u_by_scale: dict[float, float],
    rel_tol: float = 0.25,
) -> bool:
    """True if the sigma_u posterior tracks the prior scale (prior-dominated).

    Heuristic: if the ratio sigma_u/scale is roughly constant across scales
    (coefficient of variation of the ratios below rel_tol), the posterior is
    following the prior rather than the data, so sigma_u is not identified.
    """
    ratios = np.array([sigma_u_by_scale[s] / s for s in scales if s > 0])
    if ratios.size < 2:
        return False
    mean = float(ratios.mean())
    if mean == 0.0:
        return False
    cv = float(ratios.std() / mean)
    return cv < rel_tol
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sigma_u_sensitivity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/model/sigma_u_sensitivity.py tests/unit/test_sigma_u_sensitivity.py
git commit -m "feat(model): sigma_u prior-sensitivity sweep + prior-dominance rule"
```

---

### Task 5: Render σ_u in the report robustness section

**Files:**
- Modify: `engine/report/render.py:94-110` (robustness rendering block)
- Test: `tests/unit/test_render_sigma_u.py`

**Interfaces:**
- Consumes: `RobustnessSpread` with `SpecResult` entries that may carry `sigma_u`.
- Produces: `render_report` includes a `σ_u = <value>` annotation for any spec whose `sigma_u is not None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_render_sigma_u.py
from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult


def _spec(name: str, sigma_u: float | None = None) -> SpecResult:
    return SpecResult(
        spec_name=name, weighted_kappa_median=0.2,
        weighted_kappa_ci=(0.0, 0.4), flags=(), sigma_u=sigma_u,
    )


def test_render_shows_sigma_u_when_present() -> None:
    from engine.report.render import _render_robustness_lines

    spread = RobustnessSpread(
        primary=_spec("negative_binomial_per_stratum"),
        robustness=(_spec("hierarchical_pooling", sigma_u=2.19),),
    )
    text = "".join(_render_robustness_lines(spread))
    assert "hierarchical_pooling" in text
    assert "2.19" in text
    assert "σ_u" in text or "sigma_u" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_render_sigma_u.py -v`
Expected: FAIL — `_render_robustness_lines` does not exist.

- [ ] **Step 3: Write minimal implementation**

In `engine/report/render.py`, extract the robustness rendering into a helper and add σ_u. Replace the inline `if inputs.robustness is not None:` block with a call to a new module-level function, and define:
```python
def _render_robustness_lines(spread: "RobustnessSpread") -> list[str]:
    lines: list[str] = ["\n## Robustness\n"]
    all_specs = [spread.primary, *spread.robustness]
    for sr in all_specs:
        if sr.weighted_kappa_median is not None and sr.weighted_kappa_ci is not None:
            line = (
                f"- {sr.spec_name}: kappa = {sr.weighted_kappa_median:.2f} "
                f"[{sr.weighted_kappa_ci[0]:.2f}, {sr.weighted_kappa_ci[1]:.2f}]"
            )
        else:
            line = f"- {sr.spec_name}: kappa = N/A"
        if sr.sigma_u is not None:
            line += f"  (σ_u = {sr.sigma_u:.2f})"
        lines.append(line + "\n")
    spread_val = spread.spread
    if spread_val is not None:
        lines.append(f"Spread: {spread_val:.3f}\n")
    if not spread.is_consistent_in_direction():
        lines.append("**WARNING: Specs disagree on flag direction.**\n")
    return lines
```
and in `render_report`:
```python
    if inputs.robustness is not None:
        lines.extend(_render_robustness_lines(inputs.robustness))
```
(Import `RobustnessSpread` for the annotation if not already imported; it is already imported in `render.py` for `ReportInputs.robustness`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_render_sigma_u.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k "render or report" -v` — Expected: PASS (existing robustness output preserved, plus the σ_u annotation).

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/report/render.py tests/unit/test_render_sigma_u.py
git commit -m "feat(report): render sigma_u for the hierarchical robustness spec"
```

---

## Self-Review

**1. Spec coverage (8b's slice of RARR §5.3):** hierarchical model (Task 3) ✓; σ_u prior pre-registered in manifest (Task 1) ✓; σ_u persisted/surfaced (Task 2) ✓; sensitivity sweep + prior-dominance decision rule (Task 4) ✓; σ_u rendered (Task 5) ✓; non-centered + `λ` deterministic (Task 3, Global Constraints) ✓. **Deferred (logged, not gaps):** ESS-gate `_AUX_PARAMS` parameterization is unnecessary because hierarchical is a robustness spec that does not run the primary gate; the multi-prior *cycle-level* sweep run is wired in Plan 8f (cycle run); the decision rule's *consumption* (actually choosing to abandon pooling in a real cycle report) is a Plan 8f reporting concern.

**2. Placeholder scan:** none — every step has real code. Two implementer judgment points are explicitly bounded: Task 1 (whether `_make_manifest` forwards new kwargs — extend minimally if not; and treat v2-only fields as `lock_invariant_fields` at v1), and Task 3 (verify imports already present in `robustness.py`).

**3. Type consistency:** `InferenceResult.sigma_u: float | None` (Task 2) is read by `write_robustness_artifacts` (Task 2), `_run_hierarchical` return (Task 3), the decide reload + `SpecResult.sigma_u` (Task 2, field exists from Plan 8a Task 6), and `_render_robustness_lines` (Task 5). `sigma_u_hyperprior_scale: float | None` (Task 1) is read by `_run_hierarchical` (Task 3). `sweep_sigma_u`/`is_prior_dominated` (Task 4) are self-contained. Consistent.

---

## Execution

Autonomous subagent-driven (per the roadmap decision). On completion, push to PR #22 and confirm CI green (run `uv run mypy engine tests` + `uv run ruff check .` before every commit — the Plan 8a CI lesson). Then Plan 8c (tie-aware Plackett-Luce) with 8b's lessons folded in.
