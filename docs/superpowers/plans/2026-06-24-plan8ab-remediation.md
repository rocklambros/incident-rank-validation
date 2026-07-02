# Plan 8ab-Remediation — fix premortem-surfaced 8a/8b code defects

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fix the unambiguous code defects the 8a/8b implementation premortem surfaced — starting with the Critical goldset-hash enforcement — before Plan 8c.

**Architecture:** Five independent, TDD-tested fixes to existing engine code on branch `plan7/engine-upgrade-recall-pl` (PR #22). No new models. Methodology-affecting choices (F5 floor threshold, F2 gating) are pre-registered via manifest fields / documented thresholds.

**Tech Stack:** Python 3.12, NumPyro/JAX (CPU), NumPy, pytest, frozen dataclasses.

## Global Constraints

- **Run the EXACT CI commands before every commit:** `uv sync --frozen --extra narrative`; `uv run ruff check .` (whole repo); **`uv run mypy engine tests`** (engine AND tests — test funcs need `-> None`); `uv run pytest -v`; `uv run semgrep --config .semgrep.yml --error engine/`.
- **Adding a manifest field ⇒ schema_version v2 + update `test_verify_lock_raises_on_mutation`** in `tests/unit/test_prereg.py` (v2-only fields go in `lock_invariant_fields` alongside `goldset_hash`).
- **CPU only**; **frozen `cycles/2026/` byte-immutable**; **no AI attribution** in commits; reuse `_make_manifest` from `tests/unit/test_prereg.py` for typed manifest construction in tests.
- **Out of scope (deferred):** F4 (multi-label recall contract — methodology decision), F3/F6/F7/F8 + §14 annotation (Plan 8f debts).

---

### Task 1 (F1, CRITICAL): Enforce goldset_hash at infer time + thread it into the repro bundle

**Files:**
- Modify: `engine/cli/pipeline_executor.py` (add `_verify_goldset_hash`, call it after load; add `goldset_hash` param to `write_reproduction_bundle`)
- Modify: `engine/cli/pipeline.py` (pass `manifest.goldset_hash` into `write_reproduction_bundle`)
- Test: `tests/unit/test_goldset_hash_enforcement.py`

**Interfaces:**
- Consumes: `manifest.goldset_hash: str | None`, `GoldCalibration.provenance_hash: str`.
- Produces: `_verify_goldset_hash(manifest, gold) -> None` (raises `RuntimeError` on mismatch; no-op when `manifest.goldset_hash` is None); `write_reproduction_bundle(..., goldset_hash: str = "none")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_goldset_hash_enforcement.py
import pytest

from engine.calibrate.gold_schema import GoldCalibration
from engine.cli.pipeline_executor import _verify_goldset_hash
from tests.unit.test_prereg import _make_manifest


def _gold(h: str) -> GoldCalibration:
    return GoldCalibration(
        recall_labels=[], precision_labels=[], provenance_hash=h,
        rubric_hash="r", adjudicator_id="t", session_count=1,
    )


def test_mismatch_raises() -> None:
    m = _make_manifest(schema_version=2, goldset_hash="expected")
    with pytest.raises(RuntimeError, match="goldset"):
        _verify_goldset_hash(m, _gold("ACTUAL_DIFFERENT"))


def test_match_passes() -> None:
    m = _make_manifest(schema_version=2, goldset_hash="abc123")
    _verify_goldset_hash(m, _gold("abc123"))  # must not raise


def test_unbound_goldset_hash_is_noop() -> None:
    m = _make_manifest()  # v1, goldset_hash None
    _verify_goldset_hash(m, _gold("anything"))  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_goldset_hash_enforcement.py -v`
Expected: FAIL — `_verify_goldset_hash` does not exist.

- [ ] **Step 3: Write minimal implementation**

In `engine/cli/pipeline_executor.py`, add the helper (module level):
```python
def _verify_goldset_hash(manifest: "PreregManifest", gold: "GoldCalibration") -> None:
    """Fail loud if a pre-registered goldset_hash doesn't match the loaded goldset.

    No-op when goldset_hash is unbound (schema_version 1 / None). When bound, the
    goldset drives BOTH recall posteriors AND the overlap matrix W, so a silent
    mismatch corrupts the result (premortem F1)."""
    expected = manifest.goldset_hash
    if expected is None or expected == "none":
        return
    if gold.provenance_hash != expected:
        raise RuntimeError(
            f"goldset hash mismatch: manifest pre-registered {expected!r} but the "
            f"loaded goldset hashes to {gold.provenance_hash!r}. The goldset changed "
            f"after pre-registration; refusing to run inference on unverified inputs."
        )
```
Call it inside the `if _has_gold_files:` block, immediately after `_gold = load_gold_calibration(...)` and before `build_overlap_from_confusion`:
```python
            _gold = load_gold_calibration(...)
            _verify_goldset_hash(manifest, _gold)
            overlap = build_overlap_from_confusion(_gold, measurable_entries)
```
Add a `goldset_hash` parameter to `write_reproduction_bundle` and use it instead of the hardcoded `"none"`:
```python
def write_reproduction_bundle(
    out_dir: Path,
    cycle_id: str,
    engine_version: str,
    snapshot_hash: str,
    manifest_hash: str,
    lockfile_hash: str,
    stage2_manifest_hash: str = "",
    calibration_hash: str = "",
    vote_data_hash: str = "",
    goldset_hash: str = "none",
) -> None:
    ...
        goldset_hash=goldset_hash,
    ...
```
In `engine/cli/pipeline.py`, find the `write_reproduction_bundle(...)` call site(s) and pass `goldset_hash=manifest.goldset_hash or "none"` (the manifest's verified hash). (`grep -n "write_reproduction_bundle(" engine/cli/pipeline.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_goldset_hash_enforcement.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k "repro or bundle or provenance or pipeline" -v` — Expected: PASS (existing `write_reproduction_bundle` callers default `goldset_hash="none"`).

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/cli/pipeline_executor.py engine/cli/pipeline.py tests/unit/test_goldset_hash_enforcement.py
git commit -m "fix(infer): enforce pre-registered goldset_hash before building W (premortem F1)"
```

---

### Task 2 (F2): Explicit `extra_fields` + gate robustness diagnostics + forcing test

**Files:**
- Modify: `engine/model/inference.py` (extract `_check_diagnostics`; add `extra_fields=("diverging",)`)
- Modify: `engine/model/robustness.py` (add `extra_fields`; call `_check_diagnostics` in both spec fns)
- Test: `tests/unit/test_diagnostics_gate.py`

**Interfaces:**
- Produces: `_check_diagnostics(r_hat: dict[str,float], ess: dict[str,float], divergences: int, ess_fraction: float, total_draws: int) -> None` in `inference.py` (raises `DiagnosticsFailure`); used by `run_inference`, `_run_poisson_flat`, `_run_hierarchical`. All three `mcmc.run` calls pass `extra_fields=("diverging",)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_diagnostics_gate.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_diagnostics_gate.py -v`
Expected: FAIL — `_check_diagnostics` not importable.

- [ ] **Step 3: Write minimal implementation**

In `engine/model/inference.py`, extract the existing gate block (lines ~260-287) into a module-level helper, preserving the `_AUX_PARAMS={"concentration"}` exclusion:
```python
_AUX_PARAMS = {"concentration"}


def _check_diagnostics(
    r_hat: dict[str, float],
    ess: dict[str, float],
    divergences: int,
    ess_fraction: float,
    total_draws: int,
) -> None:
    """Raise DiagnosticsFailure if NUTS diagnostics are out of bounds.

    Shared by the primary and robustness specs so a divergent / poorly-mixed
    robustness fit cannot silently feed the spread (premortem F2). The ESS gate
    covers all sampled sites except `concentration` (so it covers lambda AND
    sigma_u for the hierarchical spec)."""
    max_rhat = max(r_hat.values()) if r_hat else 1.0
    if max_rhat > 1.01:
        raise DiagnosticsFailure(f"R-hat exceeded threshold: max R-hat = {max_rhat:.4f} > 1.01")
    if divergences > 0:
        raise DiagnosticsFailure(f"Post-warmup divergences detected: {divergences}")
    gated_ess = {k: v for k, v in ess.items() if k.split("[")[0] not in _AUX_PARAMS}
    min_ess_fraction = min((v / total_draws for v in gated_ess.values()), default=1.0)
    if min_ess_fraction < ess_fraction:
        raise DiagnosticsFailure(
            f"ESS below threshold: min ESS fraction = {min_ess_fraction:.4f} < {ess_fraction}"
        )
```
Replace the inline gate in `run_inference` with a call: `_check_diagnostics(r_hat_dict, ess_dict, divergences, ess_fraction, num_samples * num_chains)`. Add `extra_fields=("diverging",)` to the `mcmc.run(...)` call in `run_inference` (line ~218).
In `engine/model/robustness.py`: add `extra_fields=("diverging",)` to BOTH `mcmc.run(...)` calls (`_run_poisson_flat`, `_run_hierarchical`); import `_check_diagnostics` from `inference` and call it at the end of both functions before the `return` (using `manifest.ess_fraction` and `num_samples * num_chains`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_diagnostics_gate.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k "inference or robustness or hierarchical" -v` and `uv run pytest tests/proofs/test_two_cycle_parity.py -v` — Expected: PASS (clean synthetic fits pass the gate; robustness_specs=() means no robustness gating in synthetic).

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/model/inference.py engine/model/robustness.py tests/unit/test_diagnostics_gate.py
git commit -m "fix(model): explicit extra_fields + shared diagnostics gate for robustness specs (premortem F2)"
```

---

### Task 3 (F5): Count-floor for the overlap matrix W (kill n=1 → W=1.0)

**Files:**
- Modify: `engine/prereg/manifest.py` (add `overlap_min_fp` field, v2)
- Modify: `tests/unit/test_prereg.py` (coverage-gate)
- Modify: `engine/calibrate/confusion.py` (`min_fp_count` param)
- Modify: `engine/cli/pipeline_executor.py` (pass `manifest.overlap_min_fp`)
- Test: `tests/unit/test_confusion_floor.py`

**Interfaces:**
- Produces: `PreregManifest.overlap_min_fp: int = 1` (v2; default 1 = no floor for back-compat); `build_overlap_from_confusion(gold, measurable_entries, min_fp_count: int = 1)` — source columns with total FP `< min_fp_count` contribute no leakage.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_confusion_floor.py
from engine.calibrate.confusion import build_overlap_from_confusion
from engine.calibrate.gold_schema import GoldCalibration, GoldRecallLabel


def _gold(*triples) -> GoldCalibration:
    return GoldCalibration(
        recall_labels=[
            GoldRecallLabel(incident_id=f"i{n}", true_entry_ids=[t],
                            classifier_entry_id=p, source="g")
            for n, (t, p) in enumerate(triples)
        ],
        precision_labels=[], provenance_hash="h", rubric_hash="r",
        adjudicator_id="t", session_count=1,
    )


def test_floor_drops_single_fp_column() -> None:
    # LLM01 has exactly 1 FP (truly LLM02) -> with floor 2, no leakage column.
    gold = _gold(("LLM02", "LLM01"))
    W = build_overlap_from_confusion(gold, ("LLM01", "LLM02"), min_fp_count=2)
    assert "LLM02" not in W.weights or "LLM01" not in W.weights.get("LLM02", {})


def test_floor_keeps_sufficient_column() -> None:
    # LLM01 has 2 FPs (both truly LLM02) -> meets floor 2, W=1.0 retained.
    gold = _gold(("LLM02", "LLM01"), ("LLM02", "LLM01"))
    W = build_overlap_from_confusion(gold, ("LLM01", "LLM02"), min_fp_count=2)
    assert abs(W.weights["LLM02"]["LLM01"] - 1.0) < 1e-9


def test_default_floor_is_backcompat() -> None:
    gold = _gold(("LLM02", "LLM01"))
    W = build_overlap_from_confusion(gold, ("LLM01", "LLM02"))  # default min_fp_count=1
    assert abs(W.weights["LLM02"]["LLM01"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_confusion_floor.py -v`
Expected: FAIL — `build_overlap_from_confusion` has no `min_fp_count`.

- [ ] **Step 3: Write minimal implementation**

In `engine/calibrate/confusion.py`, add the parameter and the floor:
```python
def build_overlap_from_confusion(
    gold: GoldCalibration,
    measurable_entries: tuple[str, ...],
    min_fp_count: int = 1,
) -> OverlapWeights:
    ...
    weights: dict[str, dict[str, float]] = {}
    for source, targets in fp_counts.items():
        total = sum(targets.values())
        if total < min_fp_count:
            continue  # insufficient evidence of leakage from this source (premortem F5)
        for target, n in targets.items():
            weights.setdefault(target, {})[source] = n / total
    return OverlapWeights(weights=weights)
```
In `engine/prereg/manifest.py`, add after `sigma_u_hyperprior_scale`:
```python
    overlap_min_fp: int = 1  # min false-positive count to form a W leakage column (schema >= 2)
```
and extend the v1-exclusion in `to_dict()`:
```python
            result.pop("overlap_min_fp", None)
```
In `tests/unit/test_prereg.py`, add `"overlap_min_fp": 5` to the `mutations` dict and to `lock_invariant_fields`.
In `engine/cli/pipeline_executor.py`, pass the manifest value at the W-build call:
```python
            overlap = build_overlap_from_confusion(
                _gold, measurable_entries, min_fp_count=manifest.overlap_min_fp,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_confusion_floor.py tests/unit/test_prereg.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k "confusion or overlap or manifest or prereg" -v` — Expected: PASS; the 2026 v1 lock stays byte-stable (overlap_min_fp excluded at v1).

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/calibrate/confusion.py engine/prereg/manifest.py engine/cli/pipeline_executor.py tests/unit/test_confusion_floor.py tests/unit/test_prereg.py
git commit -m "fix(calibrate): count-floor for overlap W to avoid n=1 -> W=1.0 (premortem F5)"
```

---

### Task 4 (F10): Hardening — manifest validation, np.load, spec_name sanitize

**Files:**
- Modify: `engine/prereg/manifest.py` (`__post_init__` check)
- Modify: `engine/cli/pipeline.py` (two `np.load` → `allow_pickle=False`)
- Modify: `engine/cli/pipeline_executor.py` (sanitize `spec_name` in `write_robustness_artifacts`)
- Test: `tests/unit/test_remediation_hardening.py`

**Interfaces:**
- Produces: `PreregManifest.__post_init__` raises `ValueError` if `"hierarchical_pooling" in robustness_specs` and `sigma_u_hyperprior_scale is None`; `write_robustness_artifacts` rejects a `spec_name` containing path separators.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_remediation_hardening.py
from pathlib import Path

import numpy as np
import pytest

from engine.model.inference import InferenceResult
from tests.unit.test_prereg import _make_manifest


def test_manifest_rejects_hierarchical_without_sigma_u() -> None:
    with pytest.raises(ValueError, match="sigma_u_hyperprior_scale"):
        _make_manifest(
            schema_version=2,
            robustness_specs=("hierarchical_pooling",),
            sigma_u_hyperprior_scale=None,
        )


def test_manifest_accepts_hierarchical_with_sigma_u() -> None:
    _make_manifest(
        schema_version=2, robustness_specs=("hierarchical_pooling",),
        sigma_u_hyperprior_scale=1.0,
    )  # must not raise


def test_write_robustness_rejects_path_traversal(tmp_path: Path) -> None:
    from engine.cli.pipeline_executor import write_robustness_artifacts
    r = InferenceResult(
        lambda_samples=np.zeros((2, 2)), entry_ids=("A", "B"),
        r_hat={}, ess={}, divergences=0, num_warmup=1, num_samples=2,
    )
    with pytest.raises(ValueError, match="spec_name"):
        write_robustness_artifacts(r, tmp_path, "../evil")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_remediation_hardening.py -v`
Expected: FAIL — no validation yet.

- [ ] **Step 3: Write minimal implementation**

In `engine/prereg/manifest.py` `__post_init__`, append:
```python
        if "hierarchical_pooling" in self.robustness_specs and self.sigma_u_hyperprior_scale is None:
            raise ValueError(
                "robustness_specs declares 'hierarchical_pooling' but "
                "sigma_u_hyperprior_scale is None; set it (schema_version >= 2)."
            )
```
In `engine/cli/pipeline.py`, change both `np.load(...)` calls (primary lambda + robustness lambda) to pass `allow_pickle=False`.
In `engine/cli/pipeline_executor.py` `write_robustness_artifacts`, before building paths:
```python
    if Path(spec_name).name != spec_name or not spec_name:
        raise ValueError(f"unsafe spec_name for artifact path: {spec_name!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_remediation_hardening.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k "manifest or prereg or robustness or pipeline" -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/prereg/manifest.py engine/cli/pipeline.py engine/cli/pipeline_executor.py tests/unit/test_remediation_hardening.py
git commit -m "fix(hardening): manifest hierarchical validation, allow_pickle=False, spec_name sanitize (premortem F10)"
```

---

### Task 5 (F9): Per-spec robustness failure artifact + fail-fast CPU check

**Files:**
- Modify: `engine/cli/pipeline_executor.py` (per-spec try/except in the robustness loop; early CPU check)
- Test: `tests/unit/test_robustness_failure_artifact.py`

**Interfaces:**
- Produces: `write_robustness_failure(out_dir, spec_name, message)` writing `robustness_{spec}_failure.txt`; the infer-phase robustness loop wraps each spec and writes the failure artifact before re-raising. An early `jax.default_backend() == "cpu"` check fails fast before the primary run, not mid-loop.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_robustness_failure_artifact.py
from pathlib import Path

from engine.cli.pipeline_executor import write_robustness_failure


def test_failure_artifact_records_spec_and_message(tmp_path: Path) -> None:
    write_robustness_failure(tmp_path, "hierarchical_pooling", "ESS below threshold")
    p = tmp_path / "robustness_hierarchical_pooling_failure.txt"
    assert p.exists()
    text = p.read_text()
    assert "hierarchical_pooling" in text
    assert "ESS below threshold" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_robustness_failure_artifact.py -v`
Expected: FAIL — `write_robustness_failure` not defined.

- [ ] **Step 3: Write minimal implementation**

In `engine/cli/pipeline_executor.py`, add:
```python
def write_robustness_failure(out_dir: Path, spec_name: str, message: str) -> None:
    """Record which robustness spec failed and why (premortem F9)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"robustness_{spec_name}_failure.txt").write_text(
        f"Robustness spec '{spec_name}' failed:\n{message}\n"
    )
```
Wrap the robustness loop body so a failing spec records an artifact before propagating:
```python
            for spec_name in manifest.robustness_specs:
                try:
                    r_result = run_robustness_inference(...)
                    write_robustness_artifacts(r_result, out_dir, spec_name)
                except Exception as e:
                    write_robustness_failure(out_dir, spec_name, f"{type(e).__name__}: {e}")
                    raise
```
Add an early fail-fast CPU check at the top of `execute_infer_phase` (before the primary `run_inference`), so a GPU pod fails before doing work rather than mid-robustness-loop:
```python
    import jax
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"JAX backend is {jax.default_backend()!r}, expected 'cpu'. "
            "Set JAX_PLATFORM_NAME=cpu before launch (reproducibility)."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_robustness_failure_artifact.py -v`
Expected: PASS.
Regression: `uv run pytest tests/proofs/test_two_cycle_parity.py -v` and `uv run pytest tests/unit -k "pipeline or executor" -v` — Expected: PASS (CPU backend in CI).

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run mypy engine tests
git add engine/cli/pipeline_executor.py tests/unit/test_robustness_failure_artifact.py
git commit -m "fix(infer): per-spec robustness failure artifact + fail-fast CPU check (premortem F9)"
```

---

## Self-Review

**1. Coverage:** F1→T1, F2→T2, F5→T3, F10→T4, F9→T5. F4 deferred (methodology Q, documented). F3/F6/F7/F8/§14 = 8f debts (tracked in LESSONS-rarr.md). No gaps in the in-scope set.

**2. Placeholder scan:** every step has real code; the only judgment points are explicit (locating `write_reproduction_bundle` callers in T1; the two `np.load` sites in T4 — both grep-able).

**3. Type consistency:** `_verify_goldset_hash(manifest, gold)` (T1) uses `GoldCalibration.provenance_hash` + `manifest.goldset_hash` (both exist). `_check_diagnostics` signature (T2) is reused by all three NUTS fns. `overlap_min_fp` (T3) and `min_fp_count` param are threaded consistently. `write_robustness_failure` (T5) mirrors `write_robustness_artifacts` naming.

---

## Execution
Autonomous subagent-driven, T1 (Critical) first. Push to PR #22; CI green (run `mypy engine tests` + `ruff check .` before each commit). Then Plan 8c.
