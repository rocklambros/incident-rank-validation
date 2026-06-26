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

### Plan 8a — CI-gate correction (CRITICAL PROCESS, apply to 8b-8e)
- **The authoritative CI commands (from `.github/workflows/ci.yml`) are:** `uv sync --frozen --extra narrative` → `uv run ruff check .` → **`uv run mypy engine tests`** (engine AND tests!) → `uv run pytest -v` → `uv run semgrep --config .semgrep.yml --error engine/` → cyclonedx → cosign → synthetic run.
- **`mypy engine` (engine-only) is NOT enough** — CI runs `mypy engine tests`, and mypy-strict flags test files: every test function needs `-> None`, helper functions need typed args, and `PreregManifest(**some_dict)` fails arg-type unless the dict is typed or a typed factory is used (reuse `_make_manifest` from `tests/unit/test_prereg.py`).
- → **Every 8b-8e implementer/fixer MUST run `uv run mypy engine tests` and `uv run ruff check .` (whole repo) before committing** — the EXACT CI commands, not engine-only approximations. PR #22 CI caught 25+ mypy-on-tests errors that engine-only checks missed.

### Plan 8b divergences claim — CORRECTED (the earlier claim was WRONG)
- **CORRECTION (verified by executing numpyro 0.16.1):** the prior claim that `divergences` is a placeholder 0 (because `mcmc.run` omits `extra_fields`) is **FALSE**. `NUTS.default_fields = ('z','diverging')`, so `get_extra_fields()` returns `['diverging']` WITHOUT the opt-in — divergences ARE measured and the primary `if divergences>0: raise DiagnosticsFailure` gate IS live. **Lesson: execute, don't assert, on library-behavior claims** (two reviewers asserted "always 0" from docs; the one who ran the code refuted them; I had propagated the wrong claim into this ledger).
- **The REAL residual (8a/8b impl premortem):** robustness specs (`_run_poisson_flat`, `_run_hierarchical`) MEASURE diagnostics but do NOT gate on them (no `DiagnosticsFailure` raise), and σ_u is never ESS-gated — yet spec §14 says "ESS gate covers λ and σ_u." Carve-out documented in plan8b:19 but NOT in spec §14 (undisclosed). Remediation: add explicit `extra_fields=("diverging",)` to all 3 sites + a forcing test (funnel→DiagnosticsFailure) for upgrade-proofing; AND either gate the robustness specs (looser threshold) or amend spec §14 to disclose the carve-out.

## Plan 8b — SUMMARY
All 5 tasks complete + reviewed clean (no fix loops needed — implementers ran the exact CI commands `mypy engine tests`/`ruff check .` per the 8a lesson). Range 2aa609f..fc47cac. Hierarchical pooling robustness spec: manifest sigma_u_hyperprior_scale (v2) → _run_hierarchical (non-centered) → sigma_u captured/persisted/reloaded/rendered; sensitivity sweep + prior-dominance rule. Carry to 8c (PL): same SpecResult.extra_rankings channel, same CI discipline, divergences-fix ticket pending.

### Remediation — CI-gate process lesson (apply to 8c-8f)
- **A `-k` test SUBSET is NOT a CI proxy.** The remediation local pre-check used `uv run pytest -k "...robustness..."` which did NOT match `test_run_hierarchical.py` ("hierarchical" ≠ "robustness"), so a real failure (the F2 diagnostics gate tripping the hierarchical test's tiny-fit R-hat=1.0218>1.01) slipped to CI. → **Before pushing, run the FULL `uv run pytest -q` (no `-k`), not a keyword subset** — changes to shared code (manifest `__post_init__`, the diagnostics gate) have repo-wide reach.
- **Gating robustness specs at the strict primary R-hat (1.01) requires test fits to actually converge** — tiny NUTS fits (200 samples) flirt with the threshold. Strengthen test fits (≥1000 warmup/samples) rather than loosening the gate; a real hierarchical fit that can't hit 1.01 should fail loud (it's the "pooling unreliable" signal).

## Plan 8c (tie-aware Plackett-Luce / Davidson vote model) — SUMMARY
All 4 tasks + final-review fix complete + reviewed clean. Range 5230cac..5369059, CI green on PR #22. Davidson paired-comparison tie model (NOT ranking-PL: ballots are tie-saturated partial orders) on pinned scipy L-BFGS-B with ridge (resolves scale identifiability AND bounds separation); reduces to BT/PL as ν→0 so `include_ties=False` is a free drop-ties sensitivity. Vote-side robustness lens — **kappa stays primary**; PL ranking rides `SpecResult.extra_rankings["plackett_luce"]` (already round-trips through robustness_spread.json), richer diagnostics in `vote_plackett_luce.json`. Final review caught `DavidsonFit.converged` computed-but-dropped → now surfaced + non-finite taus guarded to null.
- **Lesson (ruff UP038):** `isinstance(x, (int, float))` fails ruff UP038 in this repo; use `isinstance(x, int | float)` union form. Applies to all JSON-`object` value guards in render code.

## Plan 8d (independent verification oracle + provisional gate) — SUMMARY
All 7 tasks + 2 fix waves complete + reviewed clean (final whole-increment review opus: Ready-to-merge YES, no Critical). Range 3a628a7..bbe7640. `engine/verify/oracle.py` (pure, NO engine.* imports) re-derives 3 deliverables by a DIFFERENT method: D1 incidence (re-impl from λ medians × strata), D2 PL via Bradley-Terry MM/fixed-point (Hunter 2004, half-credit ties — different optimizer than the engine's L-BFGS-B Davidson), D3 σ_u via DerSimonian-Laird moment surrogate on unpooled poisson_flat log-λ. `engine/verify/check.py run_oracle(cycle)` loads persisted artifacts, compares within module-constant tolerances (τ_incidence=0.95, τ_PL=0.70, σ_u band=0.75), writes `oracle_report.json`. **Gate = provisional flag only** (reuses non_publishable-style banner); NO Merkle/signer (right-sized to internal tool, 8d decision). decide_real runs the oracle after artifacts (defensively wrapped — a verification crash never invalidates a completed decide); report renders verdict + PROVISIONAL banner on any FAIL.
- **CLI commands register in `engine/cli/main.py`** (`cli.add_command(...)`), NOT in pipeline.py — pipeline.py defines the command functions, main.py wires them.
- **decide persists oracle-input deliverables** (`incidence_ranking.json`, `vote_rankings.npy`, `vote_entry_ids.json`) so re-verification is self-contained (no xlsx needed). `incidence_ranking.json` "ranking" is best→worst via the engine's own `_ranks_from_incidence` on median λ.
- **Lesson (mypy dict invariance):** oracle signatures use `Mapping[str, tuple[str, ...]]` not `dict[...]` — dict values are invariant so test literals `{"A": ("security",)}` (inferred `tuple[str]`) fail against `dict[str, tuple[str, ...]]`; `Mapping` is covariant and dict callers still satisfy it.
- **Lesson (verification robustness):** an entry-set mismatch between engine and oracle rankings is FLAGGED as FAIL (via `_ranking_deliverable`), never silently filtered (would mask the inconsistency) nor allowed to raise.

### Plan 8d — CARRY-FORWARD to 8e/8f (tracked, NOT done in 8d)
- **⛔ D3 σ_u is INERT for the real 2026 config.** The 2026 manifest declares `primary_spec=negative_binomial_per_stratum`, `robustness_specs=["poisson_flat"]` — neither produces a hierarchical σ_u, so `_hierarchical_sigma_u` returns None and **D3 always SKIPs** (the oracle gate is 2-of-3, not 3-of-3). The SKIP is now loud/self-explaining, but to make D3 LIVE, **8f must add `hierarchical_pooling` to the cycle's `robustness_specs`** (or accept the documented 2-of-3 gate). Decide deliberately at 8f. (NB: this intersects RM14 — the 2026 lock already drifts; changing robustness_specs is a lock decision.)
- **I3 — no executable test exercises `decide_real → run_oracle`** (synthetic pipeline stops at infer-real). The writer↔reader contract (filenames + JSON keys) is guarded only by static review. **8e/8f: add one integration test that runs `decide-real` on the synthetic cycle to completion and asserts `oracle_report.json` has 3 deliverables, none erroring** — pins the contract cheaply.
- **M1/M2 — tolerance calibration against real numbers.** τ_PL=0.70 with HARD top-tier-set agreement (per spec §5.7 "exact headline-tier agreement") may FAIL a legitimate MM-vs-Davidson top-tier swap; tie-break differs (engine: `common` order; oracle: entry-id) so exact-incidence ties could diverge. The constants are coarse BUILD-TIME bands — **8f reviews them against the cycle's actual τ/Δσ_u values** and adjusts (keep tier-match hard for D1 re-impl check; reconsider for D2 different-method only if real data shows spurious swaps — but spec wants exact tier agreement, so this is a deliberate spec-vs-practice call).
- **Still-open RARR debts (from 8a/8b, unchanged):** RM14 (2026 lock drift from lambda_min), thread real goldset_hash into write_reproduction_bundle + normalize ""/"none", filter gold by chosen classifier for W, populate robustness_specs to actually run hierarchical/PL, F3 wire σ_u sweep/is_prior_dominated (dead code), F6 recall min-denominator gate, F7 bare-λ baseline + power statement, F8 verify strata disjoint.

## Plan 8e (bake-off scoring/selection harness) — SUMMARY

### Metric and selection policy decisions
- **OOS-inclusive macro-recall** (`balanced_accuracy_oos`): mean per-class recall over all selection classes including `OOS_CLASS`; aligns with RARR spec §5.2.
- **`LOCKBOX_FRACTION = 0.3`**: stratified, seeded (seed=42) held-back split; once the split runs it is never re-drawn (immutable lockbox).
- **`BAKEOFF_ALPHA = 0.05`**: Benjamini-Hochberg FDR control across the full grid × class entry table; two-sided z-test per (config, class) pair with a direction filter (improvement required, no regression allowed at BH threshold).
- **`min_cell = 5`**: sparse truth cells (< 5 incidents) are excluded from the SELECTION metric only — they still appear in predictions but do not count toward balanced accuracy or BH tests. Guards against spurious z-test significance on single-digit cell sizes.
- **Floor = status-quo labels** (`floor_predictions` arg to `run_bakeoff`): reproducible, auditable, no data leakage; a majority-class fallback is a Phase-3 option if status-quo labels are unavailable.
- **Winner selection**: highest `balanced_accuracy_oos` among configs that (a) beat the floor in overall score, (b) have ≥1 BH-rejected improvement and (c) have zero BH-rejected regressions; ties broken by config name sort (deterministic).

### What is DEFERRED to Phase 3 (Plan 8f)
- **Live RunPod `predict_fn`**: the `bakeoff_cmd` click command body intentionally raises `NotImplementedError("live RunPod predict_fn is wired in Phase 3...")` — the tested harness logic lives entirely in `run_bakeoff(predict_fn=...)`.
- **The manifest lock with the full grid + 4th model choice**: grid config names and `ModelConfig` objects are Phase-3 inputs to `run_bakeoff`.
- **The live injection gate against the new model**: Phase-3 verifies the chosen winner against the injection oracle before accepting output.
- **Output to a NEW cycle directory**: Phase-3 must NOT write into the byte-immutable 2026 cycle dir; it uses a fresh cycle dir (still-open RM14 interacts here).

### Review corrections / process notes
- **Import sort (ruff I001):** inserting a new `from engine.cli.*` import must go in alphabetical order with siblings; use `uv run ruff check --fix` rather than manual placement. Happened here — bakeoff < calibration < pipeline < reclassify; ruff --fix corrected it.
- **`test_cli.py` audit**: the existing CLI tests check individual command behavior, not an exhaustive registered-command set, so adding `bakeoff_cmd` required no test correction.
- **CLI registration pattern (confirmed from 8d)**: `from engine.cli.bakeoff import bakeoff_cmd` + `cli.add_command(bakeoff_cmd)` in `engine/cli/main.py` — command functions live in submodules, `main.py` wires them.

### Plan 8e — FINAL whole-increment review (opus): Ready-to-merge YES, no Critical
- **I1 [FIXED] — zero-lockbox-cell BA-drag.** `select_winner` excludes a class from the SELECTION metric if it is non-sparse on the full goldset but has ZERO lockbox truth cell (else `per_class_recall` returns 0/0→0.0 and silently drags every config + the floor). Root cause is a structural mismatch: the metric counts a class by multi-label **cell size**, but `lockbox_split` stratifies by the **alphabetical-first** label only (`_primary_class`), so a class appearing mostly as a *secondary* label can get little/no guaranteed lockbox allocation. NOT reachable on the current 1200-row goldset (verified by a 200-seed sweep: every non-sparse class is its own primary stratum, smallest LLM07=6→2 guaranteed) but **live for the Phase-3 goldset (~5,972, more multi-label spread) and any non-default fraction/min_cell**. The guard now prevents it; **8f should still sanity-check the lockbox cell sizes table** the run emits.
- **M2 [FIXED] — provenance now records `seed` + `lockbox_fraction`** so the "touched-once" lockbox is reproducible from `classify_provenance.json` alone (lock-before-numbers needs the split frozen + recorded).
- **Coverage guard [ADDED] — `run_bakeoff` raises** if any config's predictions or the floor do not cover the lockbox ids (else `_restrict` silently shrinks the denominator). **Phase-3 `predict_fn` must return a prediction for every lockbox incident.**
- **I2 [DOC, by design] — BH policy is conservative.** A strictly-better config can be rejected (winner=None) when lockbox per-class cells are too small for any single class to reach BH significance (simulated: perfect config vs 0.6 floor → None at n=5/class, accepted at n=40/class). The dangerous direction (a *worse* config winning) is well-guarded. **8f: ensure adequate per-class lockbox power, and read a `None` winner as "no config significantly beats the floor at this power," NOT "all configs are worse."**
- **Phase-3 readiness gaps the harness silently assumes** (track): real `predict_fn` over the live goldset (keys ⊇ lockbox — now guarded); floor = status-quo labels over the same id space; seed/fraction frozen before any numbers (now recorded); goldset OOS convention matches the loader (`llm_consensus=="out-of-scope"` OR `labels==[]`); 4th-model + injection-gating; NEW cycle output dir (RM14 interacts).
