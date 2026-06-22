# Plan 8a Final Fix Report

## Finding 1 — Ruff gate (7 errors cleared)

`uv run ruff check . --fix` auto-fixed all 7 errors:
- `engine/cli/pipeline.py` — 5× UP037 (redundant quotes removed; `from __future__ import annotations` already present)
- `tests/unit/test_calibrate_with_gold_recall.py:4` — F401 unused import `PrecisionTally` removed
- `tests/unit/test_concordance_incidence_ranking.py:10` — F401 unused `pytest` removed (restored in same commit when the new test added `pytest.raises`)

## Finding 2 — Fail-loud precondition in `compute_concordance`

Added two precondition checks at the top of `compute_concordance` (after `common` is built):
1. If any entry in `common` is absent from `entry_strata` → `ValueError` naming the missing entries with message containing "entry_strata" and "Plan 8e".
2. If any stratum referenced by those entries is absent from `stratum_sizes` → analogous `ValueError`.

New test `test_compute_concordance_raises_named_valueerror_on_missing_entry_strata` in `tests/unit/test_concordance_incidence_ranking.py` calls `compute_concordance` with E3 absent from `entry_strata`, asserts `ValueError` (not `KeyError`) with message matching "E3" and "entry_strata".

## Verification outputs

```
uv run ruff check .
→ All checks passed!

uv run mypy engine
→ Success: no issues found in 97 source files

uv run pytest tests/unit/test_concordance_incidence_ranking.py -v
→ 5 passed in 1.24s

uv run pytest tests/unit -k concordance -v
→ 24 passed, 1 error (pre-existing ModuleNotFoundError: matplotlib in test_narrative.py — unrelated to these changes)

uv run pytest tests/proofs/test_two_cycle_parity.py -v
→ 2 passed in 42.30s  ✓ parity still green
```

## Note for Plan 8e

`engine/decide/concordance.py::compute_concordance` now enforces that `entry_strata` covers every entry in `common` (the intersection of `inference_result.entry_ids` and `vote_posterior.entries`). Plan 8e must build `entry_strata` to include ALL measurable entries returned by inference. If any entry has zero observed counts across all strata (sparse corpus), Plan 8e must decide whether to assign an empty-stratum sentinel (yielding incidence = 0) or to pre-filter such entries from `measurable_entries` before calling `compute_concordance`. The current posture is FAIL LOUD — no silent default to size 0.
