# U2: Phase-2 Integrity Hardening Implementation Plan (premortem-hardened)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).
> **Hardened by adversarial-premortem-complete (6 perspectives → adjudication, Round-1). The F6 design was REVISED in response (see "F6 redesign" below). Every task carries its premortem remediations.**

**Goal:** Close pre-Phase-3 integrity debt: F6 recall thin-denominator handling (CO-BLOCKER with F-C), goldset_hash threading + sentinel normalization, robustness divergence-gate forcing-test + spec §14 disclosure, F8 strata-disjoint guard, report_cmd robustness file-presence gate.

**Architecture:** Additive guards + a byte-safe, schema-versioned, **non-point-estimate-biasing** F6 treatment of thin recall cells. New manifest fields ride `schema_version >= 3` and are excluded from the `< 3` canonical `to_dict()` so existing locked manifests hash identically. F6 is OFF for schema<3 cycles, so the 2026 baseline rankings stay byte-identical.

## ⚑ F6 REDESIGN (premortem finding B/C/D/F — supersedes the original "soft Beta(5,5) widen")
The original plan silently widened thin recall cells toward a prior center. **Rejected** — every center biases λ=observed/recall and HIDES under-detected entries (B), fakes confidence + erases the 'wide' adequacy caveat (C), and is an unfalsifiable ranking lever (F). **New design (does LESS to the numbers):**
- **DETECT + FLAG, do NOT silently regularize the point estimate.** A recall cell with denominator `< K` (thin) or `true_positives == 0` (under-detected / "caught none of N known") gets a persisted flag; its adequacy label is NEVER `adequate` (new `regularized-thin`/`wide` state computed from UN-widened counts).
- **No center-biasing prior.** Keep `Beta(1,1)+counts` (honest, wide). The widening-toward-a-center is removed.
- **Uniform recall floor ε (numerical only).** To stop `λ=observed/recall` exploding when recall≈0 (which can break NUTS), apply a small, disclosed, manifest-set floor `recall >= ε` UNIFORMLY to all cells (not just thin) so relative rankings are unbiased. (If the diagnostics gate already catches the divergence in practice, the floor is the documented fallback — the implementer verifies which is needed.)
- **Disclosure is mandatory and cross-unit:** U2 PERSISTS the flags (diagnostic.json/coverage.json); **U9 renders** a "Recall regularization / under-detected entries" report section + a per-entry λ sensitivity panel; **U5 lock** requires the live RARR manifest to set F6 deliberately. "F6 done" for the pre-Phase-3 gate = U2 persistence DONE + the U9/U5 obligations tracked.
- **K** is a transparent power-flag threshold (it does NOT move λ, only flags), so the "unfalsifiable lever" concern is largely moot; still document K's derivation (align with `MIN_CELL=5` or a stated measurability target) and require a committed ranking sensitivity grid (K∈{6,8,10}) as a report artifact at U9.
- **`recall_min_denominator_gate`** (default False = keep-but-flag) lets the U5 lock OPTIONALLY exclude flagged cells from the headline — a DISCLOSED lock decision, never silent.

**[FOR THE HUMAN, on return]** This pivots my earlier firm F6 call (you authorized "go with your recommendation"). The pivot is strictly safer (no silent regularization; under-detected entries stay prominent; previous rankings byte-identical). The residual human-calls the premortem flagged — the exact K value and whether the U5 lock should hard-exclude TP=0 cells — are deferred to U5 with the sensitivity grid as the evidence base.

## Global Constraints
- No AI/Claude/Anthropic attribution. CI gate: `uv run ruff check .` + `uv run mypy engine tests` before every commit; FULL `uv run pytest -q` before any push. F4 pin green. Two-stage review + opus whole-increment; push PR #22; CI green.
- **Byte-immutability:** new manifest fields excluded from `to_dict()` below schema_version 3 (via an independent `if schema_version < 3:` block, NOT by extending the `==1` block); the 2026 v1 lock AND v2 RARR locks must re-verify (golden-hash tests). F6 OFF for schema<3.
- **Preserve previous rankings:** F6 changes NO point estimate for thick cells and is OFF for schema<3 → 2026 baseline + thick-cell published rankings byte-identical.
- No R; independent Python oracle; provisional gate.

## Tasks (F6 first; remediation IDs from the premortem in brackets)

### U2-1 (F6 core): thin/under-detected recall-cell DETECTION + adequacy-flag fix + uniform floor [CO-BLOCKER]
**Files:** `engine/calibrate/beta.py`, `engine/calibrate/calibrate.py` (adequacy flag 123-130), `tests/unit/test_beta.py`, `tests/unit/test_calibrate.py`.
- [ ] **[C]** A thin/under-detected cell (denom<K or TP==0) must NEVER report `flag=='adequate'`: add a non-collapsible `regularized-thin` (or keep `wide`) state at calibrate.py:123-130, with the adequate/wide decision + the `n=` computed from UN-widened counts. Test: a thin cell never `=='adequate'`.
- [ ] **[B]** NO center-biasing prior. Keep `from_counts` default Beta(1,1)+counts (byte-identical). Add a uniform, manifest-set recall floor ε in the recall→λ path ONLY (read inference.py to place it; uniform across cells; default ε such that thick cells unaffected). Test: floor bounds λ tail for a TP=0 cell; thick cells byte-identical with floor at default.
- [ ] **[L]** Widening/flag logic touches RECALL only (calibrate.py:55). Regression test: precision + rollup posteriors byte-identical with F6 on vs off, AND `apply_empirical_precision_prior` still fires on its (1.0,1.0) cells.
- [ ] Commit `feat(calibrate): flag thin/under-detected recall cells + uniform λ floor (F6 core, U2-1)`.

### U2-2 (F6 wiring): manifest schema-3 (lock-safe) + calibration flags persisted + measurability
**Files:** `engine/prereg/manifest.py`, `engine/calibrate/calibrate.py` (serialize dict 568-581), `engine/decide/measurability.py`, tests.
- [ ] **[A]** Add an independent `if self.schema_version < 3: result.pop(<the F6 fields>)` block ABOVE the existing `== 1` block in `to_dict()`. REQUIRED red test `test_v2_lock_still_verifies` (frozen golden v2 hex hash unchanged after the field add) + keep `test_real_2026_v1_lock_verifies` green; update `test_verify_lock_raises_on_mutation` for the new fields.
- [ ] **[H]** `__post_init__` invariant (mirror the sigma_u guard): any non-default F6 field ⇒ `schema_version >= 3` else raise; `compute_calibration` IGNORES F6 fields when `schema_version < 3` (enforced, not conventional).
- [ ] **[D-persist]** Add `thin_denominator: bool` + `min_recall_denominator: int` (+ the per-entry flag) to the EXPLICIT serialization dict at calibrate.py:568-581 (additive keys, no reorder) so the report (U9) can render them. Manifest fields (schema≥3, default-off): `recall_min_denominator: int = 0`, `recall_min_denominator_gate: bool = False`, `recall_floor_epsilon: float = 0.0`, `recall_min_denominator_rationale: str = ""` (K derivation per [F]). `measurability.py` carries the flag into `coverage.json` (additive); hard-excludes only if `recall_min_denominator_gate=True`.
- [ ] Commit `feat(prereg,calibrate): schema-3 F6 fields (lock-safe) + persist thin-cell flags (F6, U2-2)`.

### U2-3 (F6 e2e + oracle): inference stability + independent oracle recompute
**Files:** `engine/model/inference.py` (read-mostly), `engine/verify/oracle.py`/`check.py`, integration test.
- [ ] **[E]** Make the oracle independently recompute recall (+ precision) Beta posteriors from persisted tallies on its own scipy path and compare to engine `calibration.json` within tolerance — so "provisional until oracle agrees" validates the calibration, not just the arithmetic. (If deferred, DOCUMENT that the oracle does not validate F6 + add a direct unit check on the flag/floor transform.)
- [ ] Integration: thin/TP=0 cell + F6-enabled manifest → NUTS gate passes (or fails LOUD via DiagnosticsFailure), ordering of thick cells preserved, output provisional until oracle agrees. Commit `test(model,verify): F6 inference stability + oracle recomputes recall posteriors (F6, U2-3)`.

### U2-4: goldset_hash threading + sentinel normalization (scoped)
**Files:** `engine/cli/pipeline_executor.py`, `engine/cli/pipeline.py` (repro_bundle ~1015-1033), `engine/repro/bundle.py`, tests.
- [ ] **[G]** Change ONLY the goldset_hash WRITE default to `""`; leave `lockfile_hash`/`snapshot_hash` `"none"` at pipeline.py:1015/1017 UNTOUCHED. Keep `_verify_goldset_hash` early-return as `expected in (None, "", "none")` (accept legacy "none"). Regression test: the committed 2026 bundle bytes are UNCHANGED by U2 + `ReproductionBundle.read()` handles the legacy no-goldset-key bundle.
- [ ] **[M]** In `execute_infer_phase` (guarded by `_has_gold_files`) write `infer/goldset_hash.txt = _gold.provenance_hash`. In `repro_bundle_cmd`, when BOTH `infer/goldset_hash.txt` and `manifest.goldset_hash` are non-empty, ASSERT equality (manifest authoritative) and raise on mismatch; fallback file→manifest→`""`.
- [ ] Commit `fix(repro): thread goldset_hash (manifest-authoritative) + normalize write sentinel to "" (U2-4)`.

### U2-5: robustness divergence-gate forcing test (de-flaked) + spec §14 disclosure
**Files:** `tests/unit/` (split tests), spec §14, `engine/cli/pipeline_executor.py` (failure-artifact path only).
- [ ] **[J1/J2]** SPLIT: (a) a DETERMINISTIC `_check_diagnostics(divergences=1, ...)` unit test (authoritative gate proof, cross-platform stable); (b) an end-to-end variant forcing divergence via a MONKEYPATCHED/degenerate sampler (not a real seed-sensitive funnel), marked `slow`, run ubuntu-only (exclude from macos leg), with `pytest-timeout` added to dev deps + a real `--timeout` OR capped `num_warmup/num_samples/num_chains/max_tree_depth` so termination is provable.
- [ ] **[J3]** The integration assertion checks the failure file CONTENT names a divergence (e.g. contains "divergen") AND that `diagnostics_failure.txt` was NOT written (proves the PRIMARY passed and only the robustness spec gated). Defensive: `assert Path(spec).name == spec` at the failure-write site.
- [ ] Amend spec §14: divergences 0 (a `robustness_<spec>_failure.txt` is written on any violation; robustness specs apply the IDENTICAL R-hat<1.01 / ESS≥ess_fraction / divergences==0 gate as primary). Commit `test(model): de-flaked robustness divergence-gate forcing test + §14 disclosure (U2-5)`.

### U2-6: F8 strata-disjoint guard (re-targeted to the REAL double-count)
**Files:** `engine/verify/check.py` (`_build_strata`), `engine/cli/synthetic.py`, the real decide builder, a shared helper; `tests/unit/`.
- [ ] **[P]** Build `stratum_incident_sets` from `labeled_incidents` INSIDE the builders that hold those rows (NOT in `compute_concordance`, which only sees `entry_strata` and would false-positive on the 9/20 legitimately multi-stratum entries). Assert (1) GLOBAL pairwise incident-disjointness across strata (canonicalize ids first), AND (2) no stratum repeats in `entry_strata[e]` + stratum populations disjoint (the REAL incidence Σsize double-count vector at concordance.py:82-91). NEVER gate on `len(entry_strata[e])`. Treat an empty/absent set for a multi-stratum entry as a FAILURE. Confirm `verify/check.py` degrades gracefully when labeled_incidents is absent.
- [ ] Tests: passing multi-stratum-disjoint case; raising shared-incident case. Commit `feat(verify,decide): strata-disjoint incident guard prevents incidence double-count (F8, U2-6)`.

### U2-8: report_cmd robustness gate (grandfathered + content-validated)
**Files:** `engine/cli/pipeline.py` (report_cmd ~928-936); `tests/unit/test_pipeline_cli.py`.
- [ ] **[K/J3]** Gate the new raise ONLY for `schema_version >= 3` (or scope it to the decide phase producing the spread) so locked v1/v2 cycles still regenerate. Validate CONTENT: require a finite `weighted_kappa_median` per declared spec (reject null/non-finite, naming the spec) — a name-complete-but-null decoy spread must be refused. Regression tests: `report` on the committed 2026 cycle still SUCCEEDS; a null-kappa spread is refused. Commit `fix(report): grandfathered + content-validating robustness-spread gate (U2-8)`.

## Cross-unit obligations (tracked — NOT done in U2)
- **[D→U9]** Render the persisted thin-cell flags as a "Recall regularization / under-detected entries" report section + per-entry λ sensitivity panel; gate the report to require it when `recall_min_denominator>0`; a test asserts a flagged cell appears in rendered report text.
- **[F→U9]** Commit the ranking sensitivity grid (K∈{6,8,10}) as a report artifact; gate the report if top-tier ordering isn't invariant across it.
- **[I→U5]** Lock-acceptance check: the live RARR manifest must be `schema_version>=3` AND explicitly set `recall_min_denominator` (>0 with the disclosure section, or 0 with a written "thin cells left bare" rationale); fail the pre-Phase-3 gate otherwise.

## Parallel-safety (serialize on shared files)
manifest.py (U2-2), pipeline_executor.py (U2-4/U2-5/U2-6), pipeline.py (U2-4/U2-8), inference.py (U2-1/U2-3, integrity-critical) — execute serially in this session; F6 (U2-1/2/3) first.
