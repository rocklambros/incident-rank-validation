# Task 5 Report: Infer Goldset/Snapshot Provenance-Guard Raise (F-C T5)

## Status: DONE

## Commit
`b77202d` — test(integration): infer goldset/snapshot provenance-guard raise (F-C T5)

## File Changed
- **Modified:** `tests/integration/test_fc_real_cycle.py` — added `test_infer_goldset_snapshot_provenance_guard_raises` (52 lines)

No engine files were changed; this task is test-only.

## Ghost Row + Why It Triggers Check #6 (Not Check #1 or a ValueError-Rewrap)

The builder appends this row to `calibration/adjudicated_goldset.jsonl` when
`ghost_recall_id="INC-GHOST"` is passed:

```json
{"incident_id":"INC-GHOST","llm_consensus":"LLM01","labels":["LLM01"],"adjudicated":"accept"}
```

The three constraints that guarantee check #6 fires cleanly:

1. **`labels=["LLM01"]` is in `valid_entry_ids`** (`measurable_entries` = {LLM01,LLM02}).
   If the label were outside that set, `_load_recall_from_adjudicated` would raise a
   `ValueError` that the `except (ValueError, OSError, JSONDecodeError)` block in
   `execute_infer_phase` re-wraps as `RuntimeError` — wrong error.

2. **INC-GHOST is NOT in `labeled_incidents.json`**.
   `classifier_labels.get("INC-GHOST", OUT_OF_SCOPE)` returns the sentinel string
   `"out-of-scope"` (non-None) → `classifier_entry_id is not None` → INC-GHOST is
   included in the `goldset_recall_ids` set passed to `verify_labeled_completeness`.
   If INC-GHOST had been added to labeled, the FIRST `verify_labeled_completeness`
   call (check #1, labeled⊆snapshot) would fire instead; the `match="goldset"` assertion
   would fail because check #1's message says "references … absent from the pinned corpus
   snapshot", not "goldset".

3. **INC-GHOST is NOT in the corpus snapshot (`incidents.json`)**.
   Check #6 computes `missing = goldset_recall_ids - corpus_ids`; INC-GHOST is in
   `goldset_recall_ids` (from point 2) and absent from `corpus_ids` → non-empty `missing`
   → raises `LabeledIncidentsIncompleteError` with message containing "goldset recall
   incident(s) are absent".

4. **`LabeledIncidentsIncompleteError(RuntimeError)` bypasses the catch block**.
   The `except (ValueError, OSError, json.JSONDecodeError)` block does NOT catch
   `RuntimeError`, so the error propagates uncaught. `match="goldset"` is unique to
   check #6's message and confirms which check fired.

## Spy-Not-Called Confirmation

The spy is patched onto `engine.model.inference.run_inference` before the call.
`run_inference` is imported at line ~328 in `pipeline_executor.py`, AFTER the goldset
guard fires at line ~309. The spy `_spy` raises `AssertionError` if called.
After catching the expected `LabeledIncidentsIncompleteError`, `spy_called` (a list
used as a mutable flag) is asserted to be empty — confirming run_inference was never
reached.

## TDD Evidence

- **Red phase:** builder already supported `ghost_recall_id` (from T0 skeleton) but no
  test existed; writing the test first would have produced a FAIL before the
  `ghost_recall_id` path was in the builder.
- **Green phase:** builder + test together → 7 passed (T0–T5) in 1.36s.

## Gate Outputs

```
uv run pytest tests/integration/test_fc_real_cycle.py -v
  7 passed in 1.36s   (T0 builder, T1 adapter OOS, T2 recall-flip, T3 overlap W,
                        T4a batch pre-flight, T4b truncated-labeled raise, T5 ghost-guard)

uv run pytest -q  (FULL suite)
  Exit code 0; 10 XFAIL (pre-existing Plan 5 markers), 0 failures, 0 errors

uv run pytest tests/unit/test_recall_single_label_semantics.py -q
  3 passed  (F4 pin green)

uv run ruff check .
  All checks passed!

uv run mypy engine tests
  Success: no issues found in 235 source files
```

## Concerns

None. The ghost row mechanics are straightforward; the three premortem constraints
(valid label, absent from labeled, absent from corpus) each map to a single
builder decision that was already in the T0 skeleton. The `match="goldset"` substring
uniquely identifies check #6 and was verified against coverage.py line 213.
