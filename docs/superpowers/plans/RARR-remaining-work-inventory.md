# RARR Engine-Upgrade — Remaining Work Inventory

> Branch `plan7/engine-upgrade-recall-pl`, PR #22. Generated 2026-06-30 by a 4-reader survey workflow
> (lessons ledger + RARR design spec + open plan/spec docs + in-code stubs) cross-verified against
> `docs/superpowers/specs/2026-06-22-recall-aware-robustness-reanalysis-design.md`, `LESSONS-rarr.md`,
> and `.superpowers/sdd/progress.md` (local-only, not tracked in this repository). The authoritative live docket remains `LESSONS-rarr.md`; this file
> is the point-in-time map the campaign in `RARR-campaign-execution.md` executes against.

## IMMEDIATE NEXT ACTION
**F-C — CLI integration coverage** via one shared minimal-cycle fixture. The only item the ledger labels a
*pre-Phase-3 BLOCKER*. One fixture (goldset + labeled_incidents where the classifier disagrees with consensus)
covering: (a) recall-flip reflects classifier not consensus, (b) `decide_real → run_oracle` contract / 8d-I3,
(c) the two F-B integration raises routed here (cal_tally completeness-raise + infer goldset-guard),
(d) live non-empty W end-to-end. **Hard constraint:** the fixture MUST include a future-dated record exercised
through the REAL adapter — hand-matched synthetic id-sets cannot catch the producer/verifier universe-drift bug
class (the F-B Critical #1 class).

## (A) Phase-2 remaining — pre-Phase-3 debt
1. **[BLOCKER] F-C** — CLI integration coverage (above). *(LESSONS:146,150; progress.md:97)*
2. **[BLOCKER] F6** — recall min-denominator gate; OOS-as-miss makes thin/zero denominators unstable; not implemented. *(LESSONS:81,146)*
3. **goldset_hash threading** — thread real hash into `write_reproduction_bundle` + normalize `""`/`"none"` sentinel (open: `pipeline_executor.py:591`, `pipeline.py:1033`). *(LESSONS:22,81)*
4. **Robustness-spec divergences/σ_u gating** — robustness specs measure-but-don't-gate divergences; σ_u never ESS-gated; undisclosed §14 carve-out. Add `extra_fields=("diverging",)` to all 3 sites + funnel→DiagnosticsFailure test, then gate-or-amend-§14. *(LESSONS:57,60; spec §14)*
5. **F8 — strata-disjoint guard** (prevent incidence double-counting). *(LESSONS:81,25)*
6. **F3 — wire σ_u sweep / `is_prior_dominated`** (`engine/model/sigma_u_sensitivity.py` dead; live only if hierarchical pooling added — see #23). *(LESSONS:60,81; spec §5.3 SD13)*
7. **report_cmd robustness file-presence gate** (rendering of σ_u/PL done; raise-on-missing-spread not). *(LESSONS:38,39)*
8. **F7 — bare-λ 0.20 baseline reproduction + prospective power statement** (§15). *(LESSONS:28,81; spec §15)*
9. **Provenance rot** — `engine_version` 0.3.0/1.1.0/1.2.0 inconsistency not confirmed reconciled before anchoring (spec §5.9).
10. **Curation-dir relocation** — untracked `cycles/2026/calibration/curation_review.md` + `phase2_quality_report.json` belong under `projects/owasp-llm/curation/2026/` per spec SD9.

## (B) Phase-3 — lock → RunPod bake-off run → primary/robustness/oracle/report
### B0. Pre-registration lock (lock-before-numbers; gates EVERY Phase-3 number)
11. **[BLOCKER]** Lock the RARR manifest with the full config grid + pinned 4th model. *(spec §6 step2, §7, §14)*
12. **[BLOCKER]** Choose + pin (revision SHA) the 4th labeling model alongside Qwen3-235B / Llama-3.1-405B / DeepSeek-V3; must pass the live injection gate. *(spec §5.2, §12 dec.3)*
13. **[BLOCKER] F3b** — lock+verify goldset hash, committed floor-label file, the 4 bake-off constants (LOCKBOX_FRACTION/BAKEOFF_ALPHA/MIN_CELL/seed); pre-commit `winner=None` rule; forbid post-hoc alpha edits; bind scored-classifier identity into `provenance_hash`/the lock. *(LESSONS:118,132,138)*
14. **[BLOCKER]** Create the NEW cycle dir (e.g. `cycles/2026-rarr/`) binding same snapshot/taxonomy/goldset hashes — never write into byte-immutable `cycles/2026/`. *(LESSONS:97; spec §8)*
15. **[BLOCKER]** Live injection gate against the new model (Stage-2 delimiter escaping done in 8e T5; the live/recorded-response gate is NOT built). *(spec §5.2,§10; LESSONS:96)*
### B1. Live run engineering
16. **[BLOCKER]** Wire live RunPod `predict_fn` into `bakeoff_cmd` (`engine/cli/bakeoff.py:103-108` still `NotImplementedError`; harness `run_bakeoff` + checkpoint cache + coverage guard are built/CI-green). *(LESSONS:94)*
17. **[BLOCKER]** Phase-3 producer contract: winner-classify on the NEW cycle dir MUST call `write_classify_coverage(out_dir, snapshot_hash=…, corpus_incident_ids=read_snapshot_universe_ids(…), in_scope_incident_ids=…)`. *(LESSONS:151)*
18. Phase-3 predict_fn must cover every lockbox incident (coverage guard built; supplying coverage is the run-time part). *(LESSONS:107)*
19. Recall pre-check → `posteriors.precheck.json` before the bake-off (estimator built; run-time deliverable). *(spec §6 step3)*
20. Cycle run sources recall through ONE gold merge (`calibrate_with_gold` discards base recall; iterative calls overwrite). *(LESSONS:10)*
### B2. Primary + robustness + oracle execution
21. Primary re-run (κ over λ·size on new labels = headline) + λ·size baseline from original labels + bare-λ 0.20. *(spec §6 step5, §2.1)*
22. Robustness live run on RunPod CPU (hierarchical pooling, raw-vs-corrected, tie-aware PL) → mechanical spread. *(spec §6 step6)*
23. **D3/σ_u oracle deliverable — lock decision:** add `hierarchical_pooling` to robustness_specs (makes D3 live + activates #6) OR formally accept the documented 2-of-3 gate. *(LESSONS:78)*
24. M1/M2 — oracle tolerance calibration (τ_incidence=0.95, τ_PL=0.70, σ_u band=0.75) against real numbers at lock; keep tier-match hard for D1. *(LESSONS:80)*
### B3. Selection-quality / stability gates (need real numbers)
25. F2 — selection-metric goal alignment + in-scope do-no-harm guard + projected measurable_count gain. *(LESSONS:119)*
26. I2 — adequate per-class lockbox BH power; interpret `None` = "no config significantly beats floor at this power." *(LESSONS:108)*
27. I1 — sanity-check emitted lockbox cell-sizes table (zero-cell guard fixed in code; live for ~5,972-row goldset). *(LESSONS:105)*
28. F6(8e) — winner's-curse optimism (winner on lockbox, recall on full goldset incl. lockbox → CI too narrow). Calibrate winner recall on DEV split or disclose. *(LESSONS:120)*
29. F8(decide) — goldset↔corpus TV divergence reweight decision (TV recorder done). *(LESSONS:121)*
30. F10 — winner-selection stability across pod re-runs (seed-sweep/re-run check). *(LESSONS:122)*
31. Phase-3 readiness consolidated tracker (floor=status-quo labels same id-space; OOS convention; keys⊇lockbox guarded; seed/fraction frozen recorded). *(LESSONS:109)*
### B4. Report assembly (run-dependent)
32. Old-labels-under-new-estimator bridge (SD15) + the two baselines (bare-λ 0.20, λ·size-from-original-labels). *(spec §6 step8, §8 SD15)*
33. Raw-count ranking displayed beside the recall-corrected (λ·size) ranking. *(spec §5.4)*
34. Prospective power statement render (§15) — see #8.

## (C) Separate workstreams — NOT on the RARR critical path
35. **Corpus-A adapter (Plan 2)** — DONE (code committed); only plan markdown un-git-added → `git add` the doc.
36. **Corpus-B corroboration (Plan 6)** — DONE (code committed); only plan markdown un-git-added → `git add` the doc.
37. **"What the data says" notebook** — DONE (notebook + 10 Acts committed); only plan + design-spec markdown un-git-added → `git add` the two docs.
38. `Stage2Protocol` abstract `NotImplementedError` is dead/superseded by `Stage2Classifier` — cleanup, not RARR work, not a blocker.
39. **F-E status-framing discipline** (not code) — keep "infrastructure landed" distinct from "correctness/goal achieved." Ongoing.

## (D) Accepted residuals — decided NOT to fix (disclosure only)
40. RM14 v1-canonical residual (v2-only; no real hole). RM14 itself RESOLVED.
41. 8e exploratory posture (single-author goldset, 75.3% blind/consensus disagreement, within-MCSE determinism, lockbox = optimistic gain). Not closeable in Phase 1; spec §9.
42. Merkle register + signer gate — reduced to provisional-flag in 8d (right-sized); §14 Merkle criterion FALSE by design.
43. Separate `oracle.uv.lock` (§5.7 SD5) — moot (oracle is pure, no extra deps).

## Blocker summary
Pre-run: F-C (1), F6 (2). Lock: manifest (11), 4th model (12), F3b (13), new cycle dir (14), live injection gate (15). Run: live predict_fn wiring (16), producer coverage-marker contract (17).
