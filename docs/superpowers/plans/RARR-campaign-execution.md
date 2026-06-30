# RARR Engine-Upgrade — Autonomous Campaign Execution Protocol

> Authorized 2026-06-30: the user is stepping away and directed me to drive the remaining RARR docket
> (`RARR-remaining-work-inventory.md`) to completion autonomously — planning each unit, running an
> adversarial premortem on it, remediating all findings per my recommendation, then implementing in the
> most parallelized-yet-safe way with full integration/testing/verification, committing along the way,
> going with my recommendation on any question. This file is the resumable contract; on resume, read it +
> `LESSONS-rarr.md` + `.superpowers/sdd/progress.md` and continue at the first unit not marked DONE.

## Per-unit loop (rigid — applies to every unit U1..U9)
1. **Plan** — `superpowers:writing-plans` → `docs/superpowers/plans/2026-06-30-<unit>.md` (bite-sized TDD tasks, exact code).
2. **Premortem** — `adversarial-premortem-complete` on the plan (6 perspectives; rounds scaled to stakes: full for high-stakes units U1/U5/U6/U7/U8, lighter for mechanical units).
3. **Remediate** — fix EVERY surviving finding per my recommendation; fold remediations into the plan before execution.
4. **Execute** — `superpowers:subagent-driven-development`: fresh implementer per task (cheapest model that fits) + two-stage task review (spec+quality, sonnet) + a final whole-increment review on opus; ONE fix wave for review findings.
5. **Verify** — `verification-before-completion`: FULL `uv run pytest -q` (no -k) before any push; `uv run ruff check .` + `uv run mypy engine tests` before every commit; F4 pin `tests/unit/test_recall_single_label_semantics.py` MUST stay green.
6. **Land** — commit (NO AI/Claude/Anthropic attribution anywhere), push to PR #22, verify CI green (`gh pr checks 22`).
7. **Record** — update `LESSONS-rarr.md` + `.superpowers/sdd/progress.md` + the `rarr-engine-upgrade-state` memory.

## Safeguard charter (NON-NEGOTIABLE — every plan + premortem must honor; flag any violation as Critical)
- **Leakage firewall:** lockbox split (`LOCKBOX_FRACTION=0.3`, `seed=42`, stratified, drawn ONCE, never re-drawn). Winner selected on the lockbox; winner-recall calibrated on the DEV split or explicitly disclosed as optimistic (winner's-curse, F6(8e)). No selection/test data reuse.
- **Lock-before-numbers:** the RARR manifest, the 4 bake-off constants, and the `winner=None` decision rule are FROZEN before any Phase-3 number is observed. No HARKing, no p-hacking, no optional stopping, no post-hoc α/constant edits.
- **Multiplicity:** Benjamini-Hochberg FDR across the full grid×class table; `min_cell=5` sparse cells excluded from the SELECTION metric only.
- **Independent verification:** no R, ever; the oracle is independent Python (`engine/verify/`); RARR output stays PROVISIONAL until the oracle agrees (2-of-3 or 3-of-3 per the #23 hierarchical-pooling lock decision).
- **Immutability:** the 2026 cycle + goldset are byte-immutable. ALL new Phase-3 work writes to a NEW cycle dir `projects/owasp-llm/cycles/2026-rarr/` (NEVER `cycles/2026/`).
- **Recall semantics:** single-label recall (F4 pin) preserved; OOS = recall miss (policy a); labeled_incidents stays in-scope-only; completeness guard (F-B) active.
- **★ PRESERVE PREVIOUS RANKINGS (explicit user mandate):** the as-published **bare-λ 0.20** ranking AND the **λ·size-from-original-labels** ranking are NEVER overwritten. New rankings live in the new cycle dir; the report renders previous-vs-new SIDE BY SIDE for compare/contrast. F7 (U3) reproduces + freezes the previous baselines as committed artifacts BEFORE the run.
- **Determinism/placement:** seed hygiene everywhere; RunPod is the LOCATION (CPU pods for NUTS, GPU pods for the classifier); parallelize aggressively; correctness/completeness over cost.

## Unit order (most-parallelized-yet-safe)
- **U1 — F-C: CLI integration coverage [BLOCKER, FIRST, alone].** One shared minimal-cycle fixture (real adapter + a FUTURE-DATED record) covering: recall-flip-reflects-classifier, `decide_real→run_oracle` (8d-I3), the two F-B routed raises (cal_tally completeness + infer goldset-guard), live non-empty W. Foundation reused by later integration tests.
- **U2 — Phase-2 integrity hardening [parallelizable].** F6 recall min-denominator (#2), goldset_hash threading (#3), robustness divergences/σ_u gating + §14 disclosure (#4), F8 strata-disjoint (#5), F3 σ_u-sweep wiring (#6, wired-but-inert until hierarchical in robustness_specs), report_cmd file-presence gate (#7). Different modules → worktree-parallel implement, then integrate + full-suite.
- **U3 — F7 baselines + prospective power [#8, serves the preserve-rankings mandate].** Reproduce + FREEZE bare-λ 0.20 and λ·size-from-original-labels as committed artifacts; add the prospective power statement. Offline; no live run.
- **U4 — Hygiene [#9,#10,#35-37].** Reconcile engine_version provenance; relocate curation artifacts to `projects/owasp-llm/curation/2026/`; `git add` the three completed-but-untracked plan/spec docs (corpus-a-adapter, corpus-b-corroboration, notebook + design). Do NOT commit cycle artifacts (npy/tar.gz/backup JSONs).
- **U5 — Phase-3 lock authoring [B0 #11-14, HIGH-STAKES].** Write the RARR manifest + full config grid; choose+pin the 4th model (revision SHA); F3b lock bindings (goldset hash, floor-label file, the 4 constants, winner=None rule, scored-classifier identity in provenance); scaffold `cycles/2026-rarr/`. Lock-before-numbers.
- **U6 — Live injection gate [#15].** Build the live/recorded-response injection gate the 4th model must pass before inclusion.
- **U7 — Live run wiring [B1 #16-20].** Wire the live RunPod `predict_fn` into `bakeoff_cmd`; winner-classify producer emits the coverage marker (F-B contract); recall pre-check; one-gold-merge discipline. Offline-testable with a mock predict_fn.
- **U8 — THE RUN [B2 #21-24 + B3 #25-31, EXPENSIVE/IRREVERSIBLE].** Execute on RunPod: bake-off → winner → winner-classify → recall calibrate → primary/robustness/oracle, in `cycles/2026-rarr/`. Generate NEW rankings; preserve previous. All leakage/overfitting safeguards. ← HARD-STOP boundary if RunPod creds/endpoint/weights are not reachable: stage everything ready-to-run, report, do NOT fabricate numbers.
- **U9 — Report assembly [B4 #32-34].** Old-vs-new bridge (SD15), side-by-side previous/new rankings, prospective-power render, provisional banner per oracle verdict.

## Hard-stop conditions (stage + report; never fabricate)
- RunPod credentials/endpoint/model weights not provisioned for U8's live run.
- Any genuinely irreversible external action beyond the granted authorization.
- A safeguard-charter violation that cannot be remediated without a user decision (surface it, keep the rest moving).

## Status (update as units land)
- U1 F-C: not started ← NEXT
- U2..U9: not started
