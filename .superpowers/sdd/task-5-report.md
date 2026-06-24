# Task 5 (F9) Report — Per-spec robustness failure artifact + fail-fast CPU check

## TDD Evidence

| Step | Result |
|------|--------|
| Write test (Step 1) | `tests/unit/test_robustness_failure_artifact.py` created |
| Failing run (Step 2) | `ImportError: cannot import name 'write_robustness_failure'` (expected) |
| Implement (Step 3) | `write_robustness_failure` added; early CPU check + per-spec try/except in loop |
| Passing run (Step 4) | `1 passed in 0.21s` |

## CPU Check Behavior on CI (CPU path)

The early check added to `execute_infer_phase`:
```python
import jax
if jax.default_backend() != "cpu":
    raise RuntimeError(...)
```
`os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")` runs immediately before it. On CI (no GPU), JAX initializes to `"cpu"` so the check passes. On a GPU pod where JAX has already initialized to a GPU backend, `setdefault` is a no-op and the check fires before any NUTS work. This is the F9 fix: fail fast before the primary run, not mid-robustness-loop.

All 30 pipeline/executor unit tests pass under CPU. `test_two_cycle_parity.py` (2 passed) also clean.

## Per-spec Wrap Re-raises (does not swallow)

```python
except Exception as e:
    write_robustness_failure(out_dir, spec_name, f"{type(e).__name__}: {e}")
    raise
```
The bare `raise` re-raises the original exception after writing the artifact. The outer `DiagnosticsFailure` handler for the primary run is unchanged.

## mypy output

`uv run mypy engine tests` → `Success: no issues found in 204 source files`

## ruff output

`uv run ruff check .` → `All checks passed!`

## Files Changed

- `engine/cli/pipeline_executor.py` — added `write_robustness_failure`, early CPU check in `execute_infer_phase`, per-spec try/except wrap in robustness loop
- `tests/unit/test_robustness_failure_artifact.py` — new test (verbatim from brief)

## Lessons

1. The `os.environ.setdefault` pattern is racy on GPU pods: JAX eagerly initializes on first import in some runtimes, making the env var a no-op. An explicit backend assertion immediately after is the correct defensive pattern — it turns a silent mismatch into a loud early failure.
2. Per-spec failure artifacts fill the observability gap: without them, a robustness loop failure only surfaces as a generic decide-phase error with no record of which spec triggered it or what the error was.
3. The `write_robustness_failure` naming mirrors `write_robustness_artifacts` and `write_nuts_failure` — consistent artifact-writer naming makes grepping for all artifact-write sites straightforward.

---

## Review Fix Report (F9 — round 2)

### Issues addressed

| Finding | Fix |
|---------|-----|
| Robustness loop inside primary `try/except DiagnosticsFailure` — double-write | Moved robustness loop OUTSIDE the primary try/except block |
| `num_chains` hardcoded to `4` in `write_infer_artifacts` | Added `num_chains: int = 4` parameter; call site passes `num_chains=num_chains` |
| `write_robustness_failure` lacked path-safety guard | Added `if Path(spec_name).name != spec_name or not spec_name: raise ValueError(...)` |
| Re-raise test gap | Added 3 new tests: re-raise propagates, unsafe spec_name rejected, empty spec_name rejected |

### No-double-write confirmation

A robustness `DiagnosticsFailure` now hits only the inner `except Exception` in the robustness loop, writes `robustness_<spec>_failure.txt`, and re-raises. The outer `except DiagnosticsFailure` handler is only reachable if the PRIMARY `run_inference(...)` raises — it is now structurally impossible for a robustness failure to trigger `write_nuts_failure` / write `diagnostics_failure.txt`.

### Verification output

```
uv run ruff check .        → All checks passed!
uv run mypy engine tests   → Success: no issues found in 204 source files
uv run pytest tests/unit/test_robustness_failure_artifact.py tests/unit -k "pipeline or executor or robustness" -v
  → 56 passed in 104.99s
uv run pytest tests/proofs/test_two_cycle_parity.py -v
  → 2 passed in 42.89s
```

### Files changed

- `engine/cli/pipeline_executor.py` — robustness loop moved outside primary try/except; `write_infer_artifacts` gains `num_chains` param; `write_robustness_failure` gains path-safety guard
- `tests/unit/test_robustness_failure_artifact.py` — 3 new tests (re-raise, unsafe spec_name, empty spec_name)
