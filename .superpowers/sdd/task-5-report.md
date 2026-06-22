# Task 5 Report — Build overlap matrix W from goldset confusion

## TDD evidence

| Step | Command | Result |
|------|---------|--------|
| Red  | `uv run pytest tests/unit/test_confusion_overlap.py -v` | 1 error — `ModuleNotFoundError: No module named 'engine.calibrate.confusion'` |
| Green | same after writing `engine/calibrate/confusion.py` | 1 passed |
| Regression | `uv run pytest tests/unit -k "overlap or confusion or inference" -v` | 47 passed, 0 failed |
| Synthetic | `uv run pytest tests/proofs/test_two_cycle_parity.py -k parity -v` | 2 passed |
| Types | `uv run mypy engine` | Success: no issues found in 97 source files |

## How `gold` was sourced in the executor

`execute_infer_phase` did not previously load a `GoldCalibration` at all.  The gold files
for a real cycle live in `cycle / "calibration"` (the same dir that holds `posteriors.json`).
The new code checks whether `manual_curated_incidents.json` or `adjudicated_goldset.jsonl`
exists in that dir before attempting to load:

```python
_gold_dir = cycle / "calibration"
_has_gold_files = (
    (_gold_dir / "manual_curated_incidents.json").exists()
    or (_gold_dir / "adjudicated_goldset.jsonl").exists()
)
if _has_gold_files:
    _gold = load_gold_calibration(
        gold_dir=_gold_dir,
        valid_entry_ids=set(measurable_entries),
        rubric_hash=manifest.rubric_hash or "",
        adjudicator_id="executor",
    )
    overlap = build_overlap_from_confusion(_gold, measurable_entries)
```

`valid_entry_ids` is derived from `measurable_entries` (already built from
`labeled_incidents.json`).  `rubric_hash` comes from the already-loaded `PreregManifest`
(`manifest.rubric_hash`, which may be `None` for older schema-v1 manifests — coerced to
`""`).

### Fallback for no-gold cycles

If neither goldset file is present (synthetic cycles, early cycles without a curated
goldset, or any cycle that hasn't run the calibrate stage) the executor silently keeps
`overlap = OverlapWeights(weights={})`.  Additionally, a bare `except Exception` guard
around the load-and-build block prevents any unexpected load error (e.g. an entry-ID
mismatch after a rubric change) from aborting the infer phase — the fallback is logged
implicitly by keeping `overlap` unchanged.

The `test_two_cycle_parity.py` proof confirms the synthetic path (`execute_synthetic_pipeline`)
is unaffected — it never calls `execute_infer_phase`.

## Files changed

| File | Change |
|------|--------|
| `engine/calibrate/confusion.py` | New — `build_overlap_from_confusion()` |
| `engine/cli/pipeline_executor.py` | Replace `OverlapWeights(weights={})` at line 253 with gold-sourced W |
| `tests/unit/test_confusion_overlap.py` | New — unit test per TDD brief |

---

## Code-review fix — 2026-06-22

### Finding (Important): swallowed errors in gold-load branch

**Problem:** The `except Exception` in the `if _has_gold_files:` branch silently re-introduced
the inert-W bug. When goldset files are present, any load failure means a pathological state
(malformed JSON, entry-ID mismatch, OSError) — not a legitimate "no goldset" case. Swallowing
the exception caused the infer phase to silently degrade to zeroed precision/FP correction,
identical to the inert-W state.  The comment also falsely claimed errors were "logged
implicitly" (no logging exists in the function).

**Fix applied (`engine/cli/pipeline_executor.py`):**
- Narrowed `except Exception` → `except (ValueError, OSError, json.JSONDecodeError) as exc:`
  so genuine programmer bugs (AttributeError/TypeError) propagate naturally.
- Replaced the silent fallback with a loud `RuntimeError`:
  `raise RuntimeError(f"gold calibration present but failed to load: {type(exc).__name__}: {exc}") from exc`
- Removed the false "logged implicitly" comment.
- Empty-W fallback kept only in the `else` / `_has_gold_files is False` path (no change needed there).

**Edge-case tests added (`tests/unit/test_confusion_overlap.py`):**
- `test_empty_gold_returns_empty_weights` — no recall labels → `OverlapWeights(weights={})`.
- `test_true_entry_id_not_in_measurable_entries_is_skipped` — a `GoldRecallLabel` whose
  `true_entry_ids` contains an entry not in `measurable_entries` → that pair is skipped.

### Test outputs

| Suite | Result |
|-------|--------|
| `uv run pytest tests/unit/test_confusion_overlap.py -v` | **3 passed** |
| `uv run pytest tests/unit -k "overlap or confusion or inference" -v` | **49 passed** |
| `uv run pytest tests/proofs/test_two_cycle_parity.py -v` | **2 passed** (synthetic empty-W else-branch confirmed green) |
| `uv run mypy engine` | **Success: no issues found in 97 source files** |

---

## Lessons for later plans

### What plan 8e must wire

8e (or whichever plan moves to multi-model inference) must ensure `W` is built from the
**chosen classifier's** confusion, not all recall labels indiscriminately.  Currently
`build_overlap_from_confusion` uses every `GoldRecallLabel` regardless of which classifier
produced the `classifier_entry_id`.  When multiple classifiers are in play the caller
should pre-filter `gold.recall_labels` to only those labels where `source` or
`classifier_entry_id` corresponds to the active classifier before passing to
`build_overlap_from_confusion` — or add a `classifier_id` filter parameter to the
function.

### Does the synthetic path now exercise a non-empty W?

No.  `execute_synthetic_pipeline` uses the `SyntheticCorpusAdapter`, which returns a
hard-coded `_OVERLAP_WEIGHTS` (`{"LLM09": {"LLM07": 0.15}}`) via its `overlap_weights()`
method.  That adapter path is not touched by this task.  The `execute_infer_phase` function
is the real-data path only; synthetic cycles never reach it.  So the non-empty W path in
`execute_infer_phase` is exercised only when a real cycle's calibration dir holds goldset
files.  A dedicated integration test against a fixture cycle with a small goldset would be
the right way to prove the live W end-to-end — that can be deferred to plan 8e or a
dedicated testing task.
