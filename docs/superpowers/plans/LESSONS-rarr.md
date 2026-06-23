# RARR Lessons Ledger (8a → 8e learning loop)

Each plan's execution appends lessons here; the next plan reads this before its task list.
Capture: codebase realities that differed from the plan's assumptions, test gotchas,
reusable patterns, and anything that should change how later plans are written.

## Plan 8a

### Task 1 (recall from gold) — lessons
- **`calibrate_with_gold` is now gold-only for recall.** It discards `base_tally.recall_counts` entirely. Callers must NOT pre-populate base recall expecting survival, and must merge multiple gold batches *before* a single call (iterative calls overwrite). → **8e cycle run** must source recall through one gold merge.
- **A `classifier_entry_id` not in `true_entry_ids` creates a precision FP on the claimed entry** (same misclassification signal). → **Task 5** (`W` from confusion) and **8b** can reuse this.
- Import path `engine.calibrate.gold_schema` confirmed (gold dataclasses live there, not `gold_loader`).
- **Process:** existing tests may lock in current (buggy) behavior; semantic-change tasks will need oracle corrections — reviewers must judge correction vs weakening (happened here with `test_merges_with_existing_tally`). Expect more of this in later tasks.

### Task 2 (manifest schema-versioning) — lessons
- **CRITICAL (RM14): the real 2026 manifest lock already FAILS to verify.** `lambda_min` was added to the dataclass after the lock was last written (at d7fb1d2): stored `11831dee...` vs actual `6af2069e...`. Task 2 is byte-stable (no contribution to the drift). → **8e MUST resolve this deliberately** (re-write the lock or document the drift) before anchoring; the spec §14 "2026 lock still verifies" criterion is currently FALSE independent of our work.
- **Adding manifest fields requires updating `test_verify_lock_raises_on_mutation`** — it enumerates ALL manifest fields (`manifest_fields == set(mutations.keys())`) and splits hash-affecting vs lock-invariant fields. → **8b/8c add `sigma_u_hyperprior` + oracle tolerances to the manifest: use schema_version >= 2 AND update this coverage-gate test.**
- `schema_version` has no range guard (v3 would silently include fields) — acceptable for now.

### Task 3 (provenance binding) — lessons
- **`hash_json` lives in `engine.calibrate.provenance`** (NOT `engine.snapshot.provenance` — the spec-extraction was wrong). Import canonical JSON hashing from there in later plans.
- **`write_reproduction_bundle` (pipeline_executor) has no manifest in scope** → it emits `goldset_hash="none"`. The other bundle path (`pipeline.py repro_bundle_cmd`) sources `manifest_data.get("goldset_hash","none")`. → **8e cycle-run must add a `goldset_hash` param to `write_reproduction_bundle` and thread the real value, AND normalize the sentinel** (existing optional hashes use `""`; new code used `"none"`).

### Task 4 (incidence ranking) — lessons
- **9/20 entries span BOTH strata** → incidence = λ_e × Σ_s size_s over observed strata; `entry_strata` is `dict[str, tuple[str,...]]`. **8b hierarchical + 8e baseline-recompute MUST use this multi-stratum incidence** (single-stratum undercounts).
- **`compute_concordance` is called from `pipeline.py` (`decide_real`) and `synthetic.py`, NOT `pipeline_executor.py`** (plan file-map was wrong). 8e's decide wiring lives in `pipeline.py`.
- There are **4** ranking call sites inside `compute_concordance` (kappa, flags-mismatch, flags-direction, comparisons), not 3.
- **`_ranks_from_lambda` is now dead code but retained** — useful for **8e** to reproduce the as-published bare-λ 0.20 baseline alongside the new λ·size result; annotate it when 8e uses it.

### Task 5 (overlap W from confusion) — lessons
- **`build_overlap_from_confusion` mixes ALL classifiers' labels.** When multi-model paths are active, **8e must filter gold by the chosen classifier** before building W.
- **A present-but-broken goldset now RAISES** in the executor (narrowed except + RuntimeError) instead of silently using empty W. → 8e must ensure the chosen cycle's goldset loads cleanly (rubric/entry-id consistency) or infer aborts loudly.
- **No integration test exercises a live non-empty W end-to-end** against a real-cycle fixture (synthetic bypasses the gold path). → 8e should add one.
- Review rubric reminder: a bare `except Exception` + silent fallback is an Important finding (swallowed errors). Avoid in later plans.

### Task 6 (robustness wiring) — lessons
- **Robustness inference runs in the INFER phase** (`execute_infer_phase` runs+persists each `robustness_<spec>` result); `decide_real` reloads + assembles the `RobustnessSpread` + gates; `report_cmd` renders. The brief's "loop in report assembly" was impossible (decide/report lack inference inputs). → **8e** populates `manifest.robustness_specs` to actually run hierarchical/PL, and must decide the **DiagnosticsFailure policy** for robustness NUTS (currently failed/missing spec → gate raises — safe default).
- **`SpecResult.sigma_u` / `extra_rankings` round-trip through JSON but stay None** until 8b (hierarchical → sigma_u) / 8b-8c (PL → extra_rankings) populate them; the report render path is None-safe but must be EXTENDED in 8b/8c to display them.
- **`report_cmd` robustness gate trusts file-presence** — 8e should make report raise if `manifest.robustness_specs` is non-empty but the spread file is missing.
- Pre-existing **`test_narrative.py` matplotlib import error** flagged — verify whether it's a CI/test-deps gap (see check below).

## Plan 8a — SUMMARY
All 6 tasks complete + reviewed clean (Task 5 needed 1 Important fix; rest clean first pass). Range c71b1c0..c1b6b4f. Key cross-plan carry-forwards for 8e: resolve the pre-existing 2026 lock drift (RM14); thread real goldset_hash into write_reproduction_bundle + normalize ""/"none"; filter gold by chosen classifier for W; multi-stratum incidence; populate robustness_specs + DiagnosticsFailure policy.

### Plan 8a — final-review lessons (PROCESS — apply to 8b-8e)
- **Per-task reviews ran ruff SCOPED to changed files and missed the repo-wide gate.** A whole-branch review caught 7 `ruff check .` errors (UP037 redundant quoted annotations under `from __future__ import annotations`; F401 unused test imports). → **Every 8b-8e implementer must run `uv run ruff check .` (whole repo) + `uv run mypy engine` before committing**, not just scoped checks.
- **`compute_concordance` now requires `entry_strata` to cover ALL measurable entries** and fails loud (named ValueError) otherwise. → **8e must build `entry_strata` from the full measurable-entry set** (not just counted entries), or pre-filter before calling.
- **Pre-existing env gap:** `test_narrative.py` errors because plotting deps (matplotlib/plotly/seaborn) are in `uv.lock` but NOT the default `uv sync` group (your earlier `uv sync` uninstalled them). Verify whether CI installs them; 8b's charting/8e's report rendering will need them.
