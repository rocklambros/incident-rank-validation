# Implementation Prompts — Per-Phase Pickup Commands

**Purpose:** copy-paste prompts for fresh Claude Code sessions to pick up each phase of the Incident-Rank Validation Framework as defined in `docs/HANDOFF.md` v2.5 and `docs/PRD.md`. Each phase has two prompts: a **plan-creation prompt** (produces `docs/superpowers/plans/<date>-<slug>.md`) and a **plan-execution prompt** (consumes the plan and produces commits + a phase tag).

**Operational discipline encoded in every prompt:**

1. The prompt invokes `/using-superpowers` first. That skill is the meta-gate that tells Claude to look for other applicable skills before responding. Per the Superpowers contract, plan creation MUST use `writing-plans`; plan execution MUST use `executing-plans` and (when the spec is even slightly vague) `brainstorming` first.
2. Every prompt names the previous phase's plan file, methodology-changelog entry, and tag range explicitly, and instructs the new session to read them BEFORE invoking the skill. This is the "learn lessons from the previous phase" mechanism — Claude reads what the prior phase discovered as residuals, gotchas, or out-of-scope items, and writes them into the new plan's **Inherited constraints from Phase N-1** section before any task list.
3. Every plan-creation prompt ends with `--ultrathink` because plan creation is high-judgment work and the SuperClaude framework's `--ultrathink` flag enables maximum-depth analysis (~32K tokens) per `~/.claude/FLAGS.md`.
4. Every prompt explicitly forbids re-brainstorming HANDOFF or modifying it. HANDOFF v2.5 is the locked spec; the prompts are how new phases consume it without drift.
5. After every phase closes (tag landed), Rock or Claude updates `docs/PRD.md`'s phase status from NEXT/BLOCKED → DONE and ensures the next phase's prompt in this file remains accurate. This file is a *living* index; phase-prompts evolve as the codebase teaches us what worked.

**Pattern every prompt follows:**

```
/using-superpowers <one-paragraph instruction>:
  (a) HANDOFF sections to read (with focus areas),
  (b) PRD section to read,
  (c) previous phase's plan file + changelog entry + git log range to read for lessons,
  (d) any human-state files to check (REVIEWERS.md, GOLDSET-STAFFING.md, PROVISIONING-PLAN.md),
  (e) explicit "do not re-brainstorm or modify HANDOFF",
  (f) the skill to invoke and the output file path,
  (g) the structural requirement that the new plan starts with an "Inherited constraints from Phase N-1" section,
  (h) the acceptance gate the plan must satisfy (PRD §N.M),
  (i) --ultrathink.
```

---

## Phase 2 — Corpus A adapter + snapshot + per-stratum bias profiles

**Status:** DONE. Tag: `v0.2.0-plan2`. Plan: `docs/superpowers/plans/2026-05-20-corpus-a-adapter.md`.
**Previous phase:** Plan 1 (engine + synthetic cycle).
**Previous-phase artifacts to learn from:**
- `docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md` — sections "Coverage matrices" (M1–M23 closures), "Residual risks mitigated", and "Residuals still acknowledged (not mitigable in Plan 1)".
- `docs/METHODOLOGY-CHANGELOG.md` entry `0.1.0 (Plan 1 v5, 2026-05-20)`.
- `git log v0.1.0-plan1` (the tag annotation) and `git log fd04f4e..v0.1.0-plan1` (the implementation commit range).
- Specific Plan 1 lessons that constrain Plan 2: (i) M1 added a `synthetic-stress` project precisely because the original synthetic was too clean — Plan 2's tests must exercise the actual bare-LLM03 contamination and the severity-default artifact, not idealized inputs; (ii) M2's `OverlapWeights` self-loop rejection sets the pattern that defensive checks fire at *construction* time, not at use time — Plan 2's `BiasProfile` validation should follow; (iii) M3's stratum-size sanity check fails loudly when `stratum_size < observed` — Plan 2's adapter must satisfy this contract; (iv) Plan 1's commit cadence was one feat/test per task — match it; (v) **Plan 1 v5.1 erratum lesson (CI verification gap):** Plan 1 claimed CI-green acceptance for criteria 6/7/8/9 against a workflow that never executed a single job (YAML flow-style `}` collision suppressed all runs from day one; PyYAML accepted it, GitHub Actions's stricter parser rejected it; five distinct CI bugs hid behind the first). If Plan 2 modifies CI (new adapter tests, snapshot scripts), the plan must include a task that confirms CI actually executes the new logic to a green run in the Actions tab — workflow-file presence and local-test-passing are not proof of CI execution. `actionlint` is necessary but not sufficient. Erratum in `docs/METHODOLOGY-CHANGELOG.md` "Plan 1 v5.1 erratum (2026-05-20)"; class lesson in `docs/PRD.md` §3.8.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §3 audit-findings F1-F6 and F-frame, the "Mixture" paragraph at end of §3, §4 Corpus-A-is-a-mixture row, §5.1 corpus-adapter and snapshotting paragraphs, §6 control 9 snapshot integrity) and docs/PRD.md §3 (Plan 2). Then read the previous phase's lessons: docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md (focus on the Coverage matrices section for M1, M2, M3 and the Residual risks still acknowledged section), docs/METHODOLOGY-CHANGELOG.md entry 0.1.0, and run git log v0.1.0-plan1 --stat to see the actual scope and cadence Plan 1 delivered. Identify what Plan 1 discovered that constrains Plan 2 — at minimum the bare-LLM03 contamination handling (M1 motivation), construction-time defensive validation pattern (M2), stratum-size sanity contract (M3), and per-task commit cadence. Check docs/REVIEWERS.md state (Plan 2 is not reviewer-gated but the state file should be unchanged at INTERIM). Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill with HANDOFF + PRD §3 as the approved spec, and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-corpus-a-adapter.md. The plan MUST start with an "Inherited constraints from Phase 1" section listing the concrete carry-forwards before any task list. Every task MUST satisfy at least one of the acceptance criteria in PRD §3.6, and the plan as a whole MUST satisfy all of them. Target tag: v0.2.0-plan2. --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-corpus-a-adapter.md (the Plan 2 implementation plan). Confirm the "Inherited constraints from Phase 1" section is present and the plan satisfies PRD §3.6 acceptance criteria. Invoke the Superpowers executing-plans skill to work the plan task-by-task. Commit per task with messages tagged (Plan 2). After every task, run uv run pytest -v and uv run mypy engine tests and uv run ruff check . — refuse to advance to the next task on any failure (HANDOFF §6 integrity discipline: fail loudly). When all tasks pass and PRD §3.6 acceptance criteria all hold, bump docs/METHODOLOGY-CHANGELOG.md to 0.2.0, commit as docs: record Plan 2 acceptance, and create annotated tag v0.2.0-plan2. Update docs/PRD.md phase-map table: Plan 2 status from NEXT to DONE, Plan 3 status (if currently BLOCKED on Plan 2) re-evaluated. Do not push without explicit instruction. --ultrathink
```

### Plan 2 — closure notes

- GenAI Agentic corpus is a two-stratum mixture (security + ai-harm). Adapter maps 8 raw fields to 9-field IncidentRecord with `severity` defaulting to `"medium"` when absent — flagged as F2-default-seed-contamination threat-to-validity.
- Content-addressed snapshot vendoring (SHA-256 of sorted JSONL) → hash `24806f1a…` for 7,714 incidents. This hash is the binding contract for all downstream phases.
- Future-dated incident filtering is adapter-level, not engine-level. Downstream phases must not re-filter.
- CI verification gap from Plan 1 erratum did NOT recur — Plan 2 confirmed CI green in Actions tab before tagging.
- Per-stratum bias profiles + quarantine predicates are adapter concerns, not engine concerns. This clean separation held through Plans 3–5.

---

## Phase 3 — Rubric drafting + adjudication + independent-reviewer signoff workflow

**Status:** DONE. Tags: `v0.3.0-rc1` (scaffolding), `v0.3.0-plan3` (freeze). Plan: `docs/superpowers/plans/2026-05-20-rubric-freeze-workflow.md`.
**Previous phase:** Plan 2.
**Previous-phase artifacts to learn from:**
- `docs/superpowers/plans/2026-05-20-corpus-a-adapter.md` — Plan 2 adapter behavior, contamination stratum details, severity-defaulting.
- `docs/METHODOLOGY-CHANGELOG.md` entry `0.2.0`.
- `git log v0.1.0-plan1..v0.2.0-plan2` for implementation history.
- `projects/owasp-llm/cycles/2026/corpora/genai_agentic/24806f1a…/provenance.json` — the snapshot binding.
- **Procedural lesson from Plans 1–2:** vote-blindness during drafting is procedural, not engine-enforced.

### Plan 3 — closure notes

- Rubric freeze completed in single-author mode (REVIEWERS.md = INTERIM). All 20 entries drafted with boundary rules; 18 boundary cells adjudicated, 3 left ambiguous.
- Scaffolding (Phase A/B) and freeze (Phase C) split cleanly into two PRs: `v0.3.0-rc1` (scaffolding) and `v0.3.0-plan3` (freeze). This two-PR pattern worked well — downstream phases should adopt it when a phase has a human-blocker partition.
- Adjudication log lives at `projects/owasp-llm/cycles/2026/prereg/adjudication_log.json` (not `.md` as originally planned). The JSON schema includes `resolution`, `rationale`, and `ambiguity_flag` per cell.
- Rubric hash (`rubric.json` SHA-256) is locked into `manifest.lock`. Any byte change to `rubric.json` after freeze breaks the lock verification — this is the intended behavior but surprised during Plan 5 debugging.
- Vote-blindness attestation (`viewed_results_before_signoff=false`) was procedurally maintained across all Plan 3 sessions.

---

## Phase 4 — Gold-set sampler + k-fold CV calibration + coding protocol + staffing & power calc

**Status:** DONE. Tag: `v0.4.0-plan4`. Plan: `docs/superpowers/plans/2026-05-21-gold-set-calibration-pipeline.md`.
**Previous phases:** Plans 1, 2, 3.
**Previous-phase artifacts to learn from:**
- `docs/superpowers/plans/2026-05-20-corpus-a-adapter.md` and `2026-05-20-rubric-freeze-workflow.md`.
- `docs/METHODOLOGY-CHANGELOG.md` entries `0.2.0` and `0.3.0`.
- `git log v0.1.0-plan1..v0.3.0-plan3`.
- `projects/owasp-llm/cycles/2026/prereg/rubric.json` — the frozen rubric.
- `projects/owasp-llm/cycles/2026/prereg/adjudication_log.json` — per-cell adjudications.
- `projects/owasp-llm/cycles/2026/corpora/genai_agentic/24806f1a…/incidents.jsonl` — the vendored snapshot (7,714 incidents).

### Plan 4 — closure notes

- Gold-set labels were committed as 13 batch JSON files under `projects/owasp-llm/cycles/2026/calibration/batches/`. Each batch has a UUID filename and contains per-incident labels with coder identity and timestamp.
- Calibration posteriors (`calibration/posteriors.json`) contain Beta(alpha, beta) parameters per entry×stratum for both recall and precision. Most entries have very low recall (alpha=1, beta=101 = uninformative) because the corpus has few incidents per entry in the gold set.
- k-fold CV result lives at `calibration/cv_result.json`. The fold variance was low enough to not trigger any calibration instability warnings.
- The real Stage-1 keyword classifier (`engine/classify/stage1.py`) was implemented here, not in Plan 5. Plan 5 only added the Stage-2 LLM classifier on top.
- `docs/GOLDSET-STAFFING.md` was created as scaffolding by Claude; Rock populated the coder names. The file exists and is populated.
- `confidence_threshold` was added to `PreregManifest` — this is the gate that routes low-confidence Stage-1 results to Stage-2 LLM classification.

### Plan 4 — prompts (archived, phase complete)

<details>
<summary>Plan-creation prompt (archived)</summary>

```
/using-superpowers brainstorm. Read docs/HANDOFF.md (focus on §5.3 entire gold-set + coding-protocol + measurement-model section, §5.4 Priors bullet, §6 control 11(b)(c), §9 items 1 and 2) and docs/PRD.md §5 (Plan 4). Then read the previous phases' lessons: docs/superpowers/plans/2026-05-20-corpus-a-adapter.md (Plan 2 adapter behavior, contamination stratum details), docs/superpowers/plans/2026-05-20-rubric-freeze-workflow.md (Plan 3 boundary-cell adjudications — these are oversampling targets), docs/METHODOLOGY-CHANGELOG.md entries 0.2.0 and 0.3.0, git log v0.1.0-plan1..v0.3.0-plan3. Read the frozen rubric at projects/owasp-llm/cycles/2026/prereg/rubric.json and the adjudication log to identify boundary-cell incidents that need oversampling. Read engine/calibrate/sampler.py and engine/calibrate/cv.py (the Plan 1 stubs) — these encode the function signatures and return shapes Plan 4 must honor without breaking existing tests; the plan must list those tests and confirm they still pass after the stubs are promoted to full implementations. Check docs/REVIEWERS.md (must be EXTERNAL for rubric reviewer; gold-set coders need separate identification) and confirm docs/GOLDSET-STAFFING.md exists with named coders, time budget, and power calc — if it does not, Task 0 of the plan must be "create docs/GOLDSET-STAFFING.md template; Rock populates" and the plan cannot fully execute until populated. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-goldset-calibration.md. The plan MUST start with an "Inherited constraints from Phases 1-3" section listing: (i) snapshot-hash binding from Plan 2, (ii) byte-identical-rubric requirement from Plan 3, (iii) boundary-cell oversampling targets from the adjudication log, (iv) Plan 1 sampler/CV stub contract that must remain green, (v) the never-falsely-low gate (HANDOFF §5.4) — the sampler must produce a sample that, when consumed downstream, does NOT yield falsely-low or falsely-precise posteriors for low-recall entries. Target tag: v0.4.0-plan4. --ultrathink
```

</details>

<details>
<summary>Plan-execution prompt (archived)</summary>

```
/using-superpowers Read docs/superpowers/plans/2026-05-21-gold-set-calibration-pipeline.md. Confirm the "Inherited constraints from Phases 1-3" section is present and docs/GOLDSET-STAFFING.md is populated with named coders (this gate is not negotiable — Plan 4 cannot complete without it). Invoke the Superpowers executing-plans skill. Commit per task with messages tagged (Plan 4). After every task, run the full test suite plus the Plan 1 never-falsely-low gate tests (uv run pytest tests/proofs/ -v) — refuse to advance on failure. When the sampler runs against the real snapshot, the snapshot-hash-binding test from PRD §3.6 must still hold. When all tasks pass and PRD §5.6 criteria hold, bump methodology-changelog to 0.4.0, commit, tag v0.4.0-plan4, update PRD phase-map. Coder labeling is wall-clock-bound (HANDOFF §5.3 — several hundred to ~1,000 labels); the engineering tasks complete, but the gold-set artifact is only fully populated after the coders finish their work. The plan must distinguish "engineering tasks complete (tag v0.4.0-plan4-engineering)" from "gold-set fully labeled (tag v0.4.0-plan4)". --ultrathink
```

</details>

---

## Phase 5 — Real LLM 2026 cycle (Stage-2 on RunPod, classify, infer, decide, report)

**Status:** DONE (internal-only, `non_publishable=True`). Tags: `v1.0.0-plan5-internal`, `v1.1.0-plan5`. Plan: `docs/superpowers/plans/2026-05-21-llm2026-cycle.md`.
**Previous phases:** Plans 1–4.

### Plan 5 — closure notes

- **OWASP survey format surprise:** real vote data is multi-sprint Likert (Importance 1–5 across 6 sprints × 20 entries), not simple ordinal rankings. The vote loader (`engine/vote/loader.py`) had to be rewritten with format auto-detection, name→entry-ID mapping (25+ name variants → 20 IDs), and Likert-to-rank conversion with average tie-breaking. 29 of 49 respondents survived the completeness filter. **Phase 8 (ASI) must check whether ASI uses a similar survey format before assuming the loader works unchanged.**
- **Stage-2 concurrent classification:** RunPod Stage-2 classifier uses `threading.local()` for per-thread httpx clients, 18 concurrent workers (3 endpoints × 6 batch slots). Thread safety was non-trivial — the `HttpRunPodClient` manages a client pool with lock-protected registration. This infrastructure is reusable for Phase 8 without changes.
- **NUTS convergence was clean:** 0 divergences, R-hat max 1.004, ESS sufficient. CPU-pinned JAX on the Jetson completed in reasonable wall time. No need for the recovery-from-divergence path that was engineered.
- **Concordance is low:** kappa 0.228 [-0.08, 0.54]. Four entries flagged: LLM01 and NEW-PMP (vote_over_ranks), LLM05 and LLM09 (vote_under_ranks). This is an honest finding, not a bug — the taxonomy's incident-frequency ranking simply doesn't match the community's importance ranking well. Phase 6 corpus B corroboration should interpret agreement/disagreement against this baseline.
- **85% measurability coverage (17/20 entries), 3 frame-blind entries (LLM04, LLM08, LLM10), 0 classifier-blind entries.** These 3 entries have `"flag": "no-data"` in `calibration/diagnostic.json` — no precision data and classified as frame-blind by the calibration pipeline. **Phase 7 (frame-coverage audit) has 3 upgrade candidates for the 2026 LLM cycle.**
- **Calibration posteriors persist at** `projects/owasp-llm/cycles/2026/calibration/posteriors.json` (not under `calibrate/` — the directory is `calibration/`). The k-fold CV result is at `calibration/cv_result.json` (not `fold_variance.json` as originally assumed in this file's Phase 5 prompt).
- **Selection bias test:** Kruskal-Wallis H=0.379, p=0.538, severity="low". No significant difference in vote ranks between frame-blind and measurable groups — excluding the 3 frame-blind entries does not introduce selection bias.
- **Post-Plan-5 measurability bugfix:** `decide_real` and `report_cmd` originally hardcoded all entries as "measurable" instead of reading frame-blind flags from `calibration/diagnostic.json`. This produced wrong `measurable_count` (20 instead of 17), `coverage_ratio` (1.0 instead of 0.85), NaN selection bias, and an empty frame-blind list. Fixed pre-Plan-6. The original Plan 5 closure notes and Phase 6/7 prompts below have been corrected to reflect the true 85% coverage.
- **Two-tag structure:** `v1.0.0-plan5-internal` was the initial cycle completion; `v1.1.0-plan5` includes pipeline execution fixes, WandB integration, and CI lint/type fixes. The version in `engine/version.py` is `1.1.0`.
- **Reproduction bundle** includes SHA-256 hashes for manifest, lockfile, stage2_manifest, calibration posteriors, and vote data. The tar.gz is at `projects/owasp-llm/cycles/2026/repro-bundle.tar.gz` (untracked, not committed — too large for git).

### Plan 5 — prompts (archived, phase complete)

<details>
<summary>Plan-creation prompt (archived)</summary>

```
/using-superpowers Moving to phase 5. Read docs/HANDOFF.md (focus on §5.2 Stage-2 classifier paragraph, §5.4 entire Bayesian inference section, §5.5 entire decision layer + outputs section, §6 entire integrity controls section especially control 11, §7.5 GPU + RunPod provider rule, §11 v2.3→v2.4 row M17 two-cycle parity) and docs/PRD.md §6 (Plan 5) and PRD §10 (Reviewer-identification track). Then read the previous phases' lessons: all four prior plan files in docs/superpowers/plans/, docs/METHODOLOGY-CHANGELOG.md entries 0.1.0 through 0.4.0, git log v0.1.0-plan1..v0.4.0-plan4 --stat. Read every artifact under projects/owasp-llm/cycles/2026/ to inventory what's already produced (rubric, manifest, snapshot, posteriors, fold variance). Read docs/REVIEWERS.md (Plan 5 internal-only requires INTERIM minimum; publishable requires EXTERNAL for both rubric and statistical reviewers, both attested BEFORE the first infer run per HANDOFF §6.11(e)). Read docs/PROVISIONING-PLAN.md — every TBD field must be resolved before the plan can complete (GPU type, model identity, weight hash, batch size, wall-time budget with 30% restart headroom, $500 cost ceiling). Read docs/GOLDSET-STAFFING.md to confirm gold-set is fully populated (engineering complete is not enough; the labels must exist). Read engine/classify/stage2_protocol.py — Plan 5 promotes this stub to full implementation; the protocol's class signatures are the binding contract. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-llm2026-cycle.md. The plan MUST start with an "Inherited constraints from Phases 1-4" section listing: (i) phase-gate enforcement is structural — no manual vote peeking during infer, separate session for decide; (ii) Stage-2 model swap forbidden mid-cycle, restart-from-classify is the only recovery; (iii) all gold-set hash-bindings must hold; (iv) reviewer attestation state determines non_publishable derivation, do not set by hand; (v) two-cycle parity holdout per M17 is a publication gate, not an execution gate; (vi) RunPod cost ceiling is hard, overrun aborts and logs as post-hoc per HANDOFF §6.11(f); (vii) NUTS stays CPU-pinned absolutely — HANDOFF §7.5 GPU non-determinism is methodology-breaking. The plan MUST satisfy every acceptance criterion in PRD §6.6. Target tag: v1.0.0-plan5-internal (INTERIM reviewers) or v1.0.0-plan5 (EXTERNAL reviewers attested). --ultrathink
```

</details>

<details>
<summary>Plan-execution prompt (archived)</summary>

```
/using-superpowers Read docs/superpowers/plans/2026-05-21-llm2026-cycle.md. Confirm the "Inherited constraints from Phases 1-4" section is present. Re-verify docs/REVIEWERS.md, docs/PROVISIONING-PLAN.md, and docs/GOLDSET-STAFFING.md states match plan assumptions. Invoke the Superpowers executing-plans skill. CRITICAL: the decide phase must be executed in a SEPARATE Claude Code session from infer, with the vote spreadsheet inaccessible during infer development (procedural vote-blindness, HANDOFF §6 control 2). Commit per task with messages tagged (Plan 5). After classify, monitor RunPod spend against the cost ceiling — abort and log as post-hoc if spend trends to overrun. After infer, gate on NUTS diagnostics (R-hat ≤ 1.01, ESS sufficient, zero divergences); a failure means no report. After decide, regenerate the reproduction bundle and verify it produces byte-identical (within MCSE) output on a clean checkout via the cross-platform CI diff job. When all tasks pass and PRD §6.6 criteria hold, bump methodology-changelog to 1.0.0 (major bump — first real cycle), commit, tag v1.0.0-plan5-internal or v1.0.0-plan5 per reviewer state, update PRD phase-map. Execute the M17 two-cycle parity synthetic re-run as a final task and open the 30-day external-sharing audit window. --ultrathink
```

</details>

---

## Phase 6 — Corpus B corroboration cross-check

**Status:** NEXT — all prerequisites met (Plan 5 cycle outputs exist).
**Previous phases:** Plans 1–5.
**Previous-phase artifacts to learn from:**
- All prior plan files in `docs/superpowers/plans/`; methodology-changelog through `1.0.0`.
- `projects/owasp-llm/cycles/2026/classify/labeled_incidents.json` — 6,674 corpus A labels at cycle-end (Stage-1 + Stage-2 merged).
- `projects/owasp-llm/cycles/2026/results/concordance.json` — the concordance result with measurability data embedded. **Note:** there is no standalone `measurability_map.json` file; measurability is computed at runtime from `calibration/diagnostic.json` flags and reconstructed in `report_cmd`. The report (`results/report.md`) shows 85% coverage (17/20), 3 frame-blind entries (LLM04, LLM08, LLM10), 0 classifier-blind.
- `projects/owasp-llm/cycles/2026/results/report.md` — kappa 0.228, 4 flagged entries. Corpus B corroboration should be interpreted against this low-concordance baseline.
- **Critical lesson from HANDOFF §4 Corpus B row:** corpus B is qualitative corroboration only, NEVER a posterior input. Plan 6's biggest residual risk is silently extending corpus B into the likelihood. The plan must include a regression test asserting `engine/model/inference.py` does not import or read corpus B artifacts.
- **N-size lesson from HANDOFF §11 v1.0→v2.0 first row:** corpus B has ~dozens of incidents. Statistical triangulation at that N is degenerate. Plan 6 is *agreement reporting*, not testing.
- **Plan 5 classification architecture lesson:** classification is two-stage (Stage-1 keyword/indicator → Stage-2 LLM for low-confidence residue). Corpus B incidents should be classified through the same two-stage pipeline for apples-to-apples comparison, not through a separate ad-hoc method.
- **Plan 5 concordance lesson:** kappa 0.228 means incident-frequency ranking and community-importance ranking weakly agree. Corpus B agreement/disagreement sits on top of this already-weak signal — interpret accordingly.
- **Plan 5 measurability lesson:** 3 entries (LLM04, LLM08, LLM10) are frame-blind — they have no precision data in the calibration diagnostic. Corpus B corroboration should note which of the ~46 ASI incidents map to these frame-blind entries, as those mappings cannot strengthen the posterior.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §4 Corpus B role row, §5.5 corpus B corroboration bullet, §11 v1.0→v2.0 first row on the triangulation demotion) and docs/PRD.md §7 (Plan 6). Then read the Plan 5 closure notes in claudedocs/IMPLEMENTATION_PROMPTS.md (the "Plan 5 — closure notes" section) for concrete discoveries that constrain Plan 6 — especially the 85% measurability (3 frame-blind entries), low kappa (0.228), survey format surprise, selection bias result (H=0.379, p=0.538), and the fact that measurability verdicts are derived from calibration/diagnostic.json flags (no standalone measurability_map.json file). Read docs/METHODOLOGY-CHANGELOG.md through 1.0.0, git log v0.4.0-plan4..v1.1.0-plan5 --stat to understand Plan 5's scope. Read projects/owasp-llm/cycles/2026/classify/labeled_incidents.json to inventory corpus A's 6,674 labels. Read projects/owasp-llm/cycles/2026/results/concordance.json and results/report.md to understand the baseline concordance, flagged entries, and measurability map (3 frame-blind: LLM04, LLM08, LLM10). Inspect the corpus B source at ~/github_projects/www-project-top-10-for-large-language-model-applications/initiatives/agent_security_initiative/ASI Agentic Exploits & Incidents/ASI_Agentic_Exploits_Incidents.md (read-only) to understand its incident enumeration and labeling. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-corpus-b-corroboration.md. The plan MUST start with an "Inherited constraints from Phases 1-5" section listing: (i) corpus B is corroboration only — NEVER a posterior input, must include a regression test on engine/model/inference.py confirming no corpus B import; (ii) N is dozens — this is agreement reporting, not statistical testing; (iii) systematic divergence is a published finding per HANDOFF §4, never a silent adjustment; (iv) incident-id overlap may be weak — text-match fallback must be defined and its limitations declared in the artifact; (v) corpus B incidents should pass through the same two-stage classification pipeline (Stage-1 + Stage-2) as corpus A for consistency; (vi) baseline concordance is weak (kappa 0.228) — corpus B agreement reporting must be interpreted in that context; (vii) 3 entries are frame-blind (LLM04, LLM08, LLM10) — corpus B mappings to these entries are reportable but cannot strengthen the posterior. Target tag: v1.2.0-plan6 (note: v1.1.0 is already taken by Plan 5 CI fixes). --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-corpus-b-corroboration.md. Confirm the "Inherited constraints from Phases 1-5" section is present. Invoke the Superpowers executing-plans skill. Commit per task with messages tagged (Plan 6). The regression test on engine/model/inference.py must be the FIRST task — if it fails (corpus B accidentally extended into the model), stop and fix before anything else. When all tasks pass and PRD §7.5 criteria hold, bump methodology-changelog to 1.2.0, commit, tag v1.2.0-plan6, update PRD phase-map. Regenerate the cycle report with the corpus B corroboration section added; the report's headline (HANDOFF §5.5) remains the measurability map and kappa, NOT corpus B agreement. --ultrathink
```

---

## Phase 7 — Staged frame-coverage audit extension

**Status:** NEXT — 3 frame-blind entries exist (LLM04, LLM08, LLM10) as upgrade candidates.
**Previous phases:** Plans 1–5 (Plan 6 not required).
**Previous-phase artifacts to learn from:**
- All prior plan files; methodology-changelog through `1.0.0` (or `1.2.0` if Plan 6 ran first).
- **The 2026 LLM cycle has 3 frame-blind entries (LLM04, LLM08, LLM10)** (85% measurability coverage per `results/report.md`). These entries have `"flag": "no-data"` in `calibration/diagnostic.json` — no precision data, classified as frame-blind by the calibration pipeline. Phase 7 should assess whether external reference lists can be constructed for any of these entries to upgrade their measurability status.
- There is no standalone `measurability_map.json` file. Measurability verdicts are derived at runtime from `calibration/diagnostic.json` flags and reported in `results/report.md`.
- **Critical honesty lesson from HANDOFF §4 Frame-coverage-audit row:** if no external reference list is feasible for an entry, the audit cannot be built for that entry, and the entry stays unmeasurable. *That is the honest outcome, not a failure.*
- **Lesson from HANDOFF §9 item 9 framing as a blocking gate:** the acceptance criterion for external reference-list construction must be settled per entry before that entry's audit is built. Per-entry tags admissible.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §4 Frame-coverage-audit row, §6 control 7, §9 item 9, §3 F-frame critical paragraph) and docs/PRD.md §8 (Plan 7). Read projects/owasp-llm/cycles/2026/results/report.md — the Measurability Map section shows 3 frame-blind entries: LLM04, LLM08, LLM10. Read projects/owasp-llm/cycles/2026/calibration/diagnostic.json to understand why these are frame-blind (all have `"flag": "no-data"`, `has_precision_data: false`, `precision_sample_size: 0`). Read the Plan 5 closure notes in claudedocs/IMPLEMENTATION_PROMPTS.md (especially the measurability bugfix note), read docs/METHODOLOGY-CHANGELOG.md through current, git log v0.4.0-plan4..v1.1.0-plan5 --stat. For each of the 3 frame-blind entries, determine if an external reference list (incidents NOT sourced from CVE or harm databases) is feasible — if not, that entry stays unmeasurable. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-frame-coverage-audit.md. The plan MUST start with an "Inherited constraints from Phases 1-5" section listing: (i) no audit on a weak reference list — refusal is the honest outcome; (ii) per-entry scope, per-entry tags admissible; (iii) measurability map regeneration is the engine's job, the audit only produces the per-entry bound + uncertainty inputs; (iv) the staged audit is a declared extension, not a primary deliverable; (v) the 3 frame-blind entries are LLM04, LLM08, LLM10 — these had zero precision samples in calibration. Target tag: v1.3.0-plan7 (or per-entry sub-tags). --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-frame-coverage-audit.md. Confirm the "Inherited constraints from Phases 1-5" section is present, especially the "no audit on a weak reference list" criterion. Invoke the Superpowers executing-plans skill. For each of the 3 candidate entries (LLM04, LLM08, LLM10), the first task is reference-list feasibility — if infeasible for a given entry, mark as REFUSED and document why. Upgrading even one entry from frame-blind to measurable improves coverage from 85% toward 90%+. Commit per audited entry with messages tagged (Plan 7, entry <id>). After any upgrade, rerun decide-real and report to regenerate artifacts with the new measurability state. Bump methodology-changelog, tag per-entry sub-tags. Update PRD phase-map. --ultrathink
```

---

## Phase 8 — OWASP ASI Top 10 cycle

**Status:** FUTURE — engine reuse exercise, no engine changes expected. Runnable after Plan 5 (does not require Plans 6 or 7).
**Previous phases:** Plans 1–5 (Plans 6, 7 are not prerequisites).
**Previous-phase artifacts to learn from:**
- All prior plan files in `docs/superpowers/plans/`; methodology-changelog through `1.0.0` (or current).
- `projects/owasp-llm/cycles/2026/` — the full LLM cycle is the worked reference example. ASI mirrors its layout exactly. Key artifact paths: `prereg/`, `corpora/<adapter>/<hash>/`, `calibration/`, `classify/`, `infer/`, `results/`.
- `engine/` — the engine is taxonomy-neutral per HANDOFF §7.4. Any change here is a red flag and must be a documented methodology-changelog entry with a semver bump.
- **Critical reuse lesson from HANDOFF §7.4:** the *whole point* of Plan 8 is that the engine doesn't change. If during Plan 8 you find yourself editing `engine/`, stop and ask whether the change should be a documented engine upgrade (methodology-changelog bump, applies to LLM too, requires re-run of Plan 5 reproduction bundle) or an ASI-specific adapter responsibility (lives under `engine/adapters/` only, no engine-core change).
- **Critical methodology lesson from HANDOFF §7.4:** "for the ASI project both natural corpora are agentic-focused and may share selection bias. The single-channel plus declared-stratum design from v2.0 applies. Do not reintroduce a triangulation claim without a genuinely independent, comparable-N corpus and a measured bias-independence assessment." This is a *named trap* — Plan 8 must include a regression check that the ASI cycle uses single-channel modeling.
- **Reviewer-independence lesson:** ASI external reviewers must be identified separately. If Rock asks for full independence across projects, the LLM rubric reviewer cannot also be the ASI rubric reviewer. This is a `docs/REVIEWERS.md` (or `docs/REVIEWERS-asi.md`) decision.
- **Plan 5 vote-format lesson (carries into Plan 8):** the LLM vote data was multi-sprint Likert survey format, not simple rankings. The vote loader auto-detects format, but ASI may use a different survey structure. Plan 8 must verify the ASI vote data format before assuming the loader works unchanged — if ASI uses a novel format, a third format handler is needed in `engine/vote/loader.py`.
- **Plan 5 RunPod infrastructure lesson:** Stage-2 concurrent classification (thread-safe `HttpRunPodClient`, 18 workers) worked without issues. ASI can reuse the same RunPod infrastructure and `stage2_manifest.json` schema. Cost ceiling and model identity will differ.
- **Plan 5 calibration path lesson:** calibration artifacts live under `calibration/` (not `calibrate/`). The directory contains `posteriors.json`, `cv_result.json`, 13 batch files under `batches/`, and multiple provenance files. ASI needs its own calibration gold set — the LLM gold set does NOT apply to ASI entries.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §7.4 entire dual-purpose section, §9 item 10 ASI authoritative location, §4 Crosswalk-authorship row for reviewer independence) and docs/PRD.md §9 (Plan 8). Then read the Plan 5 closure notes in claudedocs/IMPLEMENTATION_PROMPTS.md for concrete lessons that carry into Plan 8 — especially the vote-format surprise, RunPod infrastructure reuse, and calibration path. Read docs/METHODOLOGY-CHANGELOG.md through current; git log v0.4.0-plan4..v1.1.0-plan5 --stat. Read projects/owasp-llm/cycles/2026/ exhaustively as the reference cycle layout — note actual directory names and artifact paths. Locate the authoritative ASI Top 10 entry definitions and ASI community vote at OWASP working-group sources (HANDOFF §9 item 10 — actual paths resolved during planning). Check docs/REVIEWERS.md or create docs/REVIEWERS-asi.md — ASI rubric reviewer and statistical reviewer must be identified separately and may need to differ from LLM reviewers per Rock's independence preference. Do not re-brainstorm HANDOFF or PRD; do not modify either. Do NOT modify engine/ unless a discovered need is a documented methodology change with semver bump and applies to LLM project too. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-asi-cycle.md. The plan MUST start with an "Inherited constraints from Phases 1-5" section listing: (i) engine-neutrality is the point — no engine/ edits without semver bump + LLM re-run; (ii) single-channel modeling for ASI per HANDOFF §7.4 — both ASI corpora share selection bias; (iii) Plan 5 is the structural reference, mirror its task order; (iv) reviewer independence per Rock's preference, separate REVIEWERS file admissible; (v) all integrity controls from Plans 1-5 apply unchanged; (vi) ASI needs its own gold-set calibration — LLM gold set does not apply; (vii) verify ASI vote data format before assuming loader compatibility. Target tag: v2.0.0-plan8 (major bump iff any engine change; minor v1.3.0-plan8 if pure reuse). --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-asi-cycle.md. Confirm the "Inherited constraints from Phases 1-5" section is present, especially engine-neutrality. Invoke the Superpowers executing-plans skill. After EVERY commit, run git diff v1.1.0-plan5..HEAD -- engine/ — any non-empty diff is a methodology change that requires a changelog entry, a semver bump decision, and a re-run of the LLM Plan 5 reproduction bundle to confirm no regression. Commit per task with messages tagged (Plan 8). When all tasks pass and PRD §9.5 criteria hold, bump methodology-changelog (major if engine touched, minor if pure reuse), tag accordingly, update PRD phase-map. The ASI cycle's report must surface the HANDOFF §7.4 standing caveat about agentic-corpora-share-bias. --ultrathink
```

---

## Meta-discipline: things every prompt assumes

These are the rules every prompt above relies on; they are not repeated in each prompt for brevity, but a session executing a prompt must honor them:

1. **No AI attribution** anywhere — commit messages, PR descriptions, code comments, file headers, changelogs. The author and committer stay human (`~/.claude/CLAUDE.md` Attribution section, enforced by `~/.claude/hooks/PreToolUse-no-ai-attribution.sh`).
2. **No `--no-verify` on commits.** Pre-push gitleaks scan and any pre-commit hook must pass. If a hook fails, fix the underlying issue.
3. **No `git push --force` to main.** No destructive git operations without explicit instruction.
4. **Confirm before risky actions** — Stage-2 RunPod provisioning, force pushes, deleting cycle artifacts, modifying engine after Plan 1.
5. **Update `docs/PRD.md` phase-map status** when a phase tag lands. This file is a living index; stale status defeats its purpose.
6. **Update this file (`claudedocs/IMPLEMENTATION_PROMPTS.md`)** when a phase teaches a lesson worth carrying into the next prompt. The prompts evolve as the codebase teaches us what worked.
7. **Honor the SessionStart MemPalace protocol:** before answering about past project context, `mempalace_search` first; never guess. For architectural decisions surfaced during planning, file via `mempalace_kg_add` + `mempalace_add_drawer` (wing=incident-rank-validation, room=decisions). Before stopping, `mempalace_diary_write` with the session's outcomes in AAAK format.
8. **Auto-memory:** if a phase produces a non-obvious learning (a discovery that would surprise a future Claude reading the code alone), save it to `~/.claude/projects/-Users-klambros-github-projects-incident-rank-validation/memory/` as a `feedback` or `project` type memory and link it from `MEMORY.md`.

---

## After every phase: update the lesson trail

When a phase closes (tag landed, PRD status updated), the executing session should also:

1. **Append a `## Plan N — closure notes` section to this file** under the relevant phase, summarizing what surprised, what was harder than expected, and what subsequent phases should watch for. Keep it to ~5 bullets. This is the highest-value mechanism for "learn lessons from the previous phase" — concrete, in-context observations that the next prompt can point at directly.
2. **Update the previous-phase artifacts list** at the top of the next phase's section if a new file emerged that's worth reading (e.g., Plan 2 might surface a `docs/CORPUS-A-NOTES.md` that Plan 3 should consume).
3. **Audit `docs/PRD.md`'s Risks-and-known-gotchas paragraphs** for the next phase — if Plan N discovered a new risk that Plan N+1 inherits, add it.

The discipline is: prompts age. Each completed phase teaches the next phase's prompt how to be more accurate. The file is rebuilt incrementally, not rewritten.
