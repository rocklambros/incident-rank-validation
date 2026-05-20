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

**Status (per PRD §2):** NEXT.
**Previous phase:** Plan 1 (engine + synthetic cycle).
**Previous-phase artifacts to learn from:**
- `docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md` — sections "Coverage matrices" (M1–M23 closures), "Residual risks mitigated", and "Residuals still acknowledged (not mitigable in Plan 1)".
- `docs/METHODOLOGY-CHANGELOG.md` entry `0.1.0 (Plan 1 v5, 2026-05-20)`.
- `git log v0.1.0-plan1` (the tag annotation) and `git log fd04f4e..v0.1.0-plan1` (the implementation commit range).
- Specific Plan 1 lessons that constrain Plan 2: (i) M1 added a `synthetic-stress` project precisely because the original synthetic was too clean — Plan 2's tests must exercise the actual bare-LLM03 contamination and the severity-default artifact, not idealized inputs; (ii) M2's `OverlapWeights` self-loop rejection sets the pattern that defensive checks fire at *construction* time, not at use time — Plan 2's `BiasProfile` validation should follow; (iii) M3's stratum-size sanity check fails loudly when `stratum_size < observed` — Plan 2's adapter must satisfy this contract; (iv) Plan 1's commit cadence was one feat/test per task — match it.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §3 audit-findings F1-F6 and F-frame, the "Mixture" paragraph at end of §3, §4 Corpus-A-is-a-mixture row, §5.1 corpus-adapter and snapshotting paragraphs, §6 control 9 snapshot integrity) and docs/PRD.md §3 (Plan 2). Then read the previous phase's lessons: docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md (focus on the Coverage matrices section for M1, M2, M3 and the Residual risks still acknowledged section), docs/METHODOLOGY-CHANGELOG.md entry 0.1.0, and run git log v0.1.0-plan1 --stat to see the actual scope and cadence Plan 1 delivered. Identify what Plan 1 discovered that constrains Plan 2 — at minimum the bare-LLM03 contamination handling (M1 motivation), construction-time defensive validation pattern (M2), stratum-size sanity contract (M3), and per-task commit cadence. Check docs/REVIEWERS.md state (Plan 2 is not reviewer-gated but the state file should be unchanged at INTERIM). Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill with HANDOFF + PRD §3 as the approved spec, and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-corpus-a-adapter.md. The plan MUST start with an "Inherited constraints from Phase 1" section listing the concrete carry-forwards before any task list. Every task MUST satisfy at least one of the acceptance criteria in PRD §3.6, and the plan as a whole MUST satisfy all of them. Target tag: v0.2.0-plan2. --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-corpus-a-adapter.md (the Plan 2 implementation plan). Confirm the "Inherited constraints from Phase 1" section is present and the plan satisfies PRD §3.6 acceptance criteria. Invoke the Superpowers executing-plans skill to work the plan task-by-task. Commit per task with messages tagged (Plan 2). After every task, run uv run pytest -v and uv run mypy engine tests and uv run ruff check . — refuse to advance to the next task on any failure (HANDOFF §6 integrity discipline: fail loudly). When all tasks pass and PRD §3.6 acceptance criteria all hold, bump docs/METHODOLOGY-CHANGELOG.md to 0.2.0, commit as docs: record Plan 2 acceptance, and create annotated tag v0.2.0-plan2. Update docs/PRD.md phase-map table: Plan 2 status from NEXT to DONE, Plan 3 status (if currently BLOCKED on Plan 2) re-evaluated. Do not push without explicit instruction. --ultrathink
```

---

## Phase 3 — Rubric drafting + adjudication + independent-reviewer signoff workflow

**Status (per PRD §4):** BLOCKED on external rubric reviewer; workflow scaffolding may proceed.
**Previous phase:** Plan 2 (if DONE; otherwise Plan 1 for scaffolding-only).
**Previous-phase artifacts to learn from:**
- `docs/superpowers/plans/YYYY-MM-DD-corpus-a-adapter.md` (the Plan 2 plan) — focus on what the corpus A adapter discovered about contamination, severity defaulting, and stratum heterogeneity. These observations inform which rubric entries need careful boundary cells.
- `docs/METHODOLOGY-CHANGELOG.md` entry `0.2.0`.
- `git log v0.1.0-plan1..v0.2.0-plan2` for implementation history.
- `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/provenance.json` if Plan 2 has landed — informs the rubric's snapshot binding.
- **Procedural lesson from Plan 1 + Plan 2 carried into Plan 3:** vote-blindness during drafting is procedural, not engine-enforced. The drafter (human + Claude) must avoid reading `~/github_projects/GenAI-LLM-Top10/2026/polling/` while drafting. The Plan 3 session itself must enforce this by *not opening* those files and by attesting `viewed_results_before_signoff=false`.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §4 Crosswalk-authorship row, §5.2 rubric artifact paragraph and rollup sub-test paragraph, §6 control 5, §6 control 11(b), (d), (e), §9 item 2) and docs/PRD.md §4 (Plan 3). Then read the previous phase's lessons: if Plan 2 is DONE, read docs/superpowers/plans/YYYY-MM-DD-corpus-a-adapter.md (focus on what the adapter discovered about real-corpus messiness — severity defaulting, contamination rate, stratum heterogeneity) and docs/METHODOLOGY-CHANGELOG.md entry 0.2.0; otherwise read docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md (focus on the rubric-attestation schema stub) and note that Plan 3 freeze cannot complete without Plan 2 done. Check docs/REVIEWERS.md current state — Plan 3 freeze requires external rubric reviewer; scaffolding does not. Confirm vote-blindness in this session BEFORE starting: do not open ~/github_projects/GenAI-LLM-Top10/2026/polling/ at any point; the attestation will record viewed_results_before_signoff=false. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill with HANDOFF + PRD §4 as the approved spec, and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-rubric-freeze-workflow.md. The plan MUST start with an "Inherited constraints from Phases 1-2" section listing concrete carry-forwards (especially: procedural vote-blindness, rubric-attestation schema from Plan 1 promoted to populated artifact, snapshot-hash binding if Plan 2 is done). The plan MUST distinguish workflow-scaffolding tasks (runnable now) from freeze tasks (blocked on external reviewer). Target tag: v0.3.0-plan3 (workflow scaffolding may merge as v0.3.0-rc1 with freeze deferred). --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-rubric-freeze-workflow.md. Confirm the "Inherited constraints from Phases 1-2" section is present. Re-confirm vote-blindness procedurally: this session does not read the polling directory. Invoke the Superpowers executing-plans skill to work the plan task-by-task. The plan partitions into scaffolding tasks (can complete now) and freeze tasks (require external reviewer). Execute scaffolding tasks; for freeze tasks, leave structured stubs and a clear blocker note. Commit per task with messages tagged (Plan 3). After scaffolding tasks complete, bump methodology-changelog to 0.3.0-rc1 (or 0.3.0 if freeze also completes) and tag accordingly. Update docs/PRD.md phase-map: Plan 3 status reflects new state. If freeze tasks remain blocked, surface the blocker in PRD §4.1 with a clear statement of what unblocks it. --ultrathink
```

---

## Phase 4 — Gold-set sampler + k-fold CV calibration + coding protocol + staffing & power calc

**Status (per PRD §5):** BLOCKED on Plan 2, Plan 3 freeze, gold-set coder identification.
**Previous phases:** Plans 1, 2, 3.
**Previous-phase artifacts to learn from:**
- `docs/superpowers/plans/YYYY-MM-DD-corpus-a-adapter.md` and `YYYY-MM-DD-rubric-freeze-workflow.md`.
- `docs/METHODOLOGY-CHANGELOG.md` entries `0.2.0` and `0.3.0`.
- `git log v0.1.0-plan1..v0.3.0-plan3`.
- `projects/owasp-llm/cycles/2026/prereg/rubric.json` — the frozen rubric. Boundary cells flagged as `(both-labels, ambiguous)` are the incidents that need oversampling in the recall frame.
- `projects/owasp-llm/cycles/2026/prereg/adjudication_log.md` — Rock's per-cell adjudications inform which strata are heterogeneous and need finer stratification.
- `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/incidents.json` — the snapshot. Gold set is hash-bound; sampler reads this exact snapshot.
- **Critical lesson from Plan 1 (Plan 4 promotes stubs to implementations):** the calibration sampler stub at `engine/calibrate/sampler.py` and the CV stub at `engine/calibrate/cv.py` raise `NotImplementedError` with messages naming Plan 4. Plan 4 replaces those with full implementations, and removes the NotImplementedError-raise tests in favor of behavior tests. The schema contracts those stubs encoded (function signature, return shape) are the binding contracts Plan 4 must honor without breaking Plan 1's tests that import them.
- **Human-state file that must exist before Plan 4 can start:** `docs/GOLDSET-STAFFING.md` with two human coders and a third adjudicator named, a time budget, and a power calculation. This file does not exist yet; the Plan 4 plan must create it as a Task 0 prerequisite, but the *content* (named coders) cannot be filled in by Claude — Rock must populate.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §5.3 entire gold-set + coding-protocol + measurement-model section, §5.4 Priors bullet, §6 control 11(b)(c), §9 items 1 and 2) and docs/PRD.md §5 (Plan 4). Then read the previous phases' lessons: docs/superpowers/plans/YYYY-MM-DD-corpus-a-adapter.md (Plan 2 adapter behavior, contamination stratum details), docs/superpowers/plans/YYYY-MM-DD-rubric-freeze-workflow.md (Plan 3 boundary-cell adjudications — these are oversampling targets), docs/METHODOLOGY-CHANGELOG.md entries 0.2.0 and 0.3.0, git log v0.1.0-plan1..v0.3.0-plan3. Read the frozen rubric at projects/owasp-llm/cycles/2026/prereg/rubric.json and the adjudication log to identify boundary-cell incidents that need oversampling. Read engine/calibrate/sampler.py and engine/calibrate/cv.py (the Plan 1 stubs) — these encode the function signatures and return shapes Plan 4 must honor without breaking existing tests; the plan must list those tests and confirm they still pass after the stubs are promoted to full implementations. Check docs/REVIEWERS.md (must be EXTERNAL for rubric reviewer; gold-set coders need separate identification) and confirm docs/GOLDSET-STAFFING.md exists with named coders, time budget, and power calc — if it does not, Task 0 of the plan must be "create docs/GOLDSET-STAFFING.md template; Rock populates" and the plan cannot fully execute until populated. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-goldset-calibration.md. The plan MUST start with an "Inherited constraints from Phases 1-3" section listing: (i) snapshot-hash binding from Plan 2, (ii) byte-identical-rubric requirement from Plan 3, (iii) boundary-cell oversampling targets from the adjudication log, (iv) Plan 1 sampler/CV stub contract that must remain green, (v) the never-falsely-low gate (HANDOFF §5.4) — the sampler must produce a sample that, when consumed downstream, does NOT yield falsely-low or falsely-precise posteriors for low-recall entries. Target tag: v0.4.0-plan4. --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-goldset-calibration.md. Confirm the "Inherited constraints from Phases 1-3" section is present and docs/GOLDSET-STAFFING.md is populated with named coders (this gate is not negotiable — Plan 4 cannot complete without it). Invoke the Superpowers executing-plans skill. Commit per task with messages tagged (Plan 4). After every task, run the full test suite plus the Plan 1 never-falsely-low gate tests (uv run pytest tests/proofs/ -v) — refuse to advance on failure. When the sampler runs against the real snapshot, the snapshot-hash-binding test from PRD §3.6 must still hold. When all tasks pass and PRD §5.6 criteria hold, bump methodology-changelog to 0.4.0, commit, tag v0.4.0-plan4, update PRD phase-map. Coder labeling is wall-clock-bound (HANDOFF §5.3 — several hundred to ~1,000 labels); the engineering tasks complete, but the gold-set artifact is only fully populated after the coders finish their work. The plan must distinguish "engineering tasks complete (tag v0.4.0-plan4-engineering)" from "gold-set fully labeled (tag v0.4.0-plan4)". --ultrathink
```

---

## Phase 5 — Real LLM 2026 cycle (Stage-2 on RunPod, classify, infer, decide, publish)

**Status (per PRD §6):** BLOCKED on Plans 2–4 plus external reviewers for publishable output (internal-only run permitted with `non_publishable=True`).
**Previous phases:** Plans 1–4.
**Previous-phase artifacts to learn from:**
- All prior plan files in `docs/superpowers/plans/`.
- `docs/METHODOLOGY-CHANGELOG.md` entries `0.1.0` through `0.4.0`.
- `git log v0.1.0-plan1..v0.4.0-plan4`.
- `projects/owasp-llm/cycles/2026/prereg/` (frozen rubric + manifest + attestations).
- `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/` (vendored snapshot + provenance).
- `projects/owasp-llm/cycles/2026/calibrate/posteriors.json` (Beta posteriors from Plan 4).
- `projects/owasp-llm/cycles/2026/calibrate/fold_variance.json` (k=5 CV variance from Plan 4).
- `docs/PROVISIONING-PLAN.md` — must be populated with concrete GPU type, model identity, weight hash, batch size, wall-time budget, $500 cost ceiling (HANDOFF §7.5 + M9).
- `docs/REVIEWERS.md` — INTERIM permits internal-only run; EXTERNAL required for publishable output.
- **Cumulative integrity-control lesson from Plans 1–4:** the engine enforces phase gates structurally; the *human* must not contaminate the gates by manual peeking. Plan 5's biggest residual risk is vote leakage during interactive `infer` debugging. The plan must enforce a separate Claude Code session for `decide` with the vote file inaccessible during `infer` development.
- **Cost lesson from HANDOFF §7.5:** Stage-2 mid-cycle model swap is forbidden. If the Stage-2 manifest needs to change mid-run, the cycle restarts from `classify` — meaning RunPod cost is *all-or-nothing*, not incremental. The provisioning plan's wall-time budget must include a 30% headroom for one full restart per HANDOFF v2.5 §7.5 misestimate protocol.
- **Two-cycle parity lesson (M17):** external sharing is gated on a 30-day audit window after a second cycle (synthetic re-run against the same engine version) produces matching headline shape. Plan 5's plan must include a Task N+1 for the synthetic parity run and a calendar reminder for the 30-day window.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §5.2 Stage-2 classifier paragraph, §5.4 entire Bayesian inference section, §5.5 entire decision layer + outputs section, §6 entire integrity controls section especially control 11, §7.5 GPU + RunPod provider rule, §11 v2.3→v2.4 row M17 two-cycle parity) and docs/PRD.md §6 (Plan 5) and PRD §10 (Reviewer-identification track). Then read the previous phases' lessons: all four prior plan files in docs/superpowers/plans/, docs/METHODOLOGY-CHANGELOG.md entries 0.1.0 through 0.4.0, git log v0.1.0-plan1..v0.4.0-plan4 --stat. Read every artifact under projects/owasp-llm/cycles/2026/ to inventory what's already produced (rubric, manifest, snapshot, posteriors, fold variance). Read docs/REVIEWERS.md (Plan 5 internal-only requires INTERIM minimum; publishable requires EXTERNAL for both rubric and statistical reviewers, both attested BEFORE the first infer run per HANDOFF §6.11(e)). Read docs/PROVISIONING-PLAN.md — every TBD field must be resolved before the plan can complete (GPU type, model identity, weight hash, batch size, wall-time budget with 30% restart headroom, $500 cost ceiling). Read docs/GOLDSET-STAFFING.md to confirm gold-set is fully populated (engineering complete is not enough; the labels must exist). Read engine/classify/stage2_protocol.py — Plan 5 promotes this stub to full implementation; the protocol's class signatures are the binding contract. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-llm2026-cycle.md. The plan MUST start with an "Inherited constraints from Phases 1-4" section listing: (i) phase-gate enforcement is structural — no manual vote peeking during infer, separate session for decide; (ii) Stage-2 model swap forbidden mid-cycle, restart-from-classify is the only recovery; (iii) all gold-set hash-bindings must hold; (iv) reviewer attestation state determines non_publishable derivation, do not set by hand; (v) two-cycle parity holdout per M17 is a publication gate, not an execution gate; (vi) RunPod cost ceiling is hard, overrun aborts and logs as post-hoc per HANDOFF §6.11(f); (vii) NUTS stays CPU-pinned absolutely — HANDOFF §7.5 GPU non-determinism is methodology-breaking. The plan MUST satisfy every acceptance criterion in PRD §6.6. Target tag: v1.0.0-plan5-internal (INTERIM reviewers) or v1.0.0-plan5 (EXTERNAL reviewers attested). --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-llm2026-cycle.md. Confirm the "Inherited constraints from Phases 1-4" section is present. Re-verify docs/REVIEWERS.md, docs/PROVISIONING-PLAN.md, and docs/GOLDSET-STAFFING.md states match plan assumptions. Invoke the Superpowers executing-plans skill. CRITICAL: the decide phase must be executed in a SEPARATE Claude Code session from infer, with the vote spreadsheet inaccessible during infer development (procedural vote-blindness, HANDOFF §6 control 2). Commit per task with messages tagged (Plan 5). After classify, monitor RunPod spend against the cost ceiling — abort and log as post-hoc if spend trends to overrun. After infer, gate on NUTS diagnostics (R-hat ≤ 1.01, ESS sufficient, zero divergences); a failure means no report. After decide, regenerate the reproduction bundle and verify it produces byte-identical (within MCSE) output on a clean checkout via the cross-platform CI diff job. When all tasks pass and PRD §6.6 criteria hold, bump methodology-changelog to 1.0.0 (major bump — first real cycle), commit, tag v1.0.0-plan5-internal or v1.0.0-plan5 per reviewer state, update PRD phase-map. Execute the M17 two-cycle parity synthetic re-run as a final task and open the 30-day external-sharing audit window. Do not externally share before the window closes. --ultrathink
```

---

## Phase 6 — Corpus B corroboration cross-check

**Status (per PRD §7):** FUTURE — runnable anytime after Plan 5 cycle outputs exist.
**Previous phases:** Plans 1–5.
**Previous-phase artifacts to learn from:**
- All prior plan files; methodology-changelog through `1.0.0`.
- `projects/owasp-llm/cycles/2026/classify/labeled_incidents.json` — corpus A labels at cycle-end.
- `projects/owasp-llm/cycles/2026/results/measurability_map.json` — informs which entries are interesting overlap targets with corpus B.
- **Critical lesson from HANDOFF §4 Corpus B row:** corpus B is qualitative corroboration only, NEVER a posterior input. Plan 6's biggest residual risk is silently extending corpus B into the likelihood. The plan must include a regression test asserting `engine/model/inference.py` does not import or read corpus B artifacts.
- **N-size lesson from HANDOFF §11 v1.0→v2.0 first row:** corpus B has ~dozens of incidents. Statistical triangulation at that N is degenerate. Plan 6 is *agreement reporting*, not testing.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §4 Corpus B role row, §5.5 corpus B corroboration bullet, §11 v1.0→v2.0 first row on the triangulation demotion) and docs/PRD.md §7 (Plan 6). Then read the previous phases' lessons: all prior plan files in docs/superpowers/plans/, docs/METHODOLOGY-CHANGELOG.md through 1.0.0, git log v0.1.0-plan1..v1.0.0-plan5 --stat. Read projects/owasp-llm/cycles/2026/classify/labeled_incidents.json to inventory corpus A's labels and projects/owasp-llm/cycles/2026/results/measurability_map.json to identify entries where corpus B overlap is meaningful. Inspect the corpus B source at ~/github_projects/www-project-top-10-for-large-language-model-applications/initiatives/agent_security_initiative/ASI Agentic Exploits & Incidents/ASI_Agentic_Exploits_Incidents.md (read-only) to understand its incident enumeration and labeling. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-corpus-b-corroboration.md. The plan MUST start with an "Inherited constraints from Phases 1-5" section listing: (i) corpus B is corroboration only — NEVER a posterior input, must include a regression test on engine/model/inference.py confirming no corpus B import; (ii) N is dozens — this is agreement reporting, not statistical testing; (iii) systematic divergence is a published finding per HANDOFF §4, never a silent adjustment; (iv) incident-id overlap may be weak — text-match fallback must be defined and its limitations declared in the artifact. Target tag: v1.1.0-plan6. --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-corpus-b-corroboration.md. Confirm the "Inherited constraints from Phases 1-5" section is present. Invoke the Superpowers executing-plans skill. Commit per task with messages tagged (Plan 6). The regression test on engine/model/inference.py must be the FIRST task — if it fails (corpus B accidentally extended into the model), stop and fix before anything else. When all tasks pass and PRD §7.5 criteria hold, bump methodology-changelog to 1.1.0, commit, tag v1.1.0-plan6, update PRD phase-map. Regenerate the cycle report with the corpus B corroboration section added; the report's headline (HANDOFF §5.5) remains the measurability map and kappa, NOT corpus B agreement. --ultrathink
```

---

## Phase 7 — Staged frame-coverage audit extension

**Status (per PRD §8):** FUTURE — optional; only runnable if an external reference list is feasible per HANDOFF §9 item 9.
**Previous phases:** Plans 1–5 (Plan 6 not required).
**Previous-phase artifacts to learn from:**
- All prior plan files; methodology-changelog through `1.0.0` (or `1.1.0` if Plan 6 ran first).
- `projects/owasp-llm/cycles/2026/results/measurability_map.json` — frame-blind entries are the upgrade candidates. Entries already measurable need no audit.
- **Critical honesty lesson from HANDOFF §4 Frame-coverage-audit row:** if no external reference list is feasible for an entry, the audit cannot be built for that entry, and the entry stays unmeasurable. *That is the honest outcome, not a failure.* Plan 7's biggest residual risk is producing an audit on a too-weak reference list and inflating an entry's status. The plan must include an explicit "Audit feasibility refusal" criterion per entry — if the reference list is weak, the audit is *not produced*, and the entry's status stays `frame-blind-unmeasurable`.
- **Lesson from HANDOFF §9 item 9 framing as a blocking gate:** the acceptance criterion for external reference-list construction must be settled per entry before that entry's audit is built. This means Plan 7 may be partial — auditing entries A, B, C but not D, E if D, E lack feasible reference lists. Per-entry tags (`v1.2.0-plan7-llm04`, etc.) are admissible.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §4 Frame-coverage-audit row, §6 control 7, §9 item 9, §3 F-frame critical paragraph) and docs/PRD.md §8 (Plan 7). Then read the previous phases' lessons: all prior plan files, docs/METHODOLOGY-CHANGELOG.md through current, git log v0.1.0-plan1..HEAD --stat. Read projects/owasp-llm/cycles/2026/results/measurability_map.json — the frame-blind entries listed there are the upgrade candidates; entries already measurable or classifier-blind-but-bounded do not need an audit. For each candidate entry, the plan must determine if an external reference list (incidents NOT sourced from CVE or harm databases) is feasible — if not, that entry stays unmeasurable and is excluded from the audit scope, not weakened to fit. Do not re-brainstorm HANDOFF or PRD; do not modify either. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-frame-coverage-audit.md. The plan MUST start with an "Inherited constraints from Phases 1-5" section listing: (i) no audit on a weak reference list — refusal is the honest outcome; (ii) per-entry scope, per-entry tags admissible; (iii) measurability map regeneration is the engine's job, the audit only produces the per-entry bound + uncertainty inputs; (iv) the staged audit is a declared extension, not a primary deliverable — if it never ships, the affected entries stay unmeasurable and that is acceptable per HANDOFF §4. Target tag: v1.2.0-plan7 (or per-entry sub-tags). --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-frame-coverage-audit.md. Confirm the "Inherited constraints from Phases 1-5" section is present, especially the "no audit on a weak reference list" criterion. Invoke the Superpowers executing-plans skill. For each candidate entry, the first task is reference-list feasibility — if infeasible, mark the entry's audit as REFUSED and document why, then proceed to the next entry. Commit per audited entry with messages tagged (Plan 7, entry <id>). When audited entries have bounds + uncertainties, regenerate the measurability map and verify entries transition only with documented bounds. Bump methodology-changelog per audited entry, tag per-entry sub-tags. Update PRD phase-map status to "PARTIAL" or "DONE-with-refusals" as accurate. --ultrathink
```

---

## Phase 8 — OWASP ASI Top 10 cycle

**Status (per PRD §9):** FUTURE — engine reuse exercise, no engine changes expected.
**Previous phases:** Plans 1–7 (all are lessons; engine changes from any of them apply unchanged to ASI).
**Previous-phase artifacts to learn from:**
- All prior plan files; methodology-changelog through whatever version is current.
- `projects/owasp-llm/cycles/2026/` — the full LLM cycle is the worked reference example. ASI mirrors its layout exactly.
- `engine/` — the engine is taxonomy-neutral per HANDOFF §7.4. Any change here is a red flag and must be a documented methodology-changelog entry with a semver bump.
- **Critical reuse lesson from HANDOFF §7.4:** the *whole point* of Plan 8 is that the engine doesn't change. If during Plan 8 you find yourself editing `engine/`, stop and ask whether the change should be a documented engine upgrade (methodology-changelog bump, applies to LLM too, requires re-run of Plan 5 reproduction bundle) or an ASI-specific adapter responsibility (lives under `engine/adapters/` only, no engine-core change).
- **Critical methodology lesson from HANDOFF §7.4:** "for the ASI project both natural corpora are agentic-focused and may share selection bias. The single-channel plus declared-stratum design from v2.0 applies. Do not reintroduce a triangulation claim without a genuinely independent, comparable-N corpus and a measured bias-independence assessment." This is a *named trap* — Plan 8 must include a regression check that the ASI cycle uses single-channel modeling.
- **Reviewer-independence lesson:** ASI external reviewers must be identified separately. If Rock asks for full independence across projects, the LLM rubric reviewer cannot also be the ASI rubric reviewer. This is a `docs/REVIEWERS.md` (or `docs/REVIEWERS-asi.md`) decision.

### Plan-creation prompt

```
/using-superpowers Read docs/HANDOFF.md (focus on §7.4 entire dual-purpose section, §9 item 10 ASI authoritative location, §4 Crosswalk-authorship row for reviewer independence) and docs/PRD.md §9 (Plan 8). Then read the previous phases' lessons: all plan files in docs/superpowers/plans/ — the LLM 2026 cycle (Plan 5) is the worked reference and ASI mirrors its structure; docs/METHODOLOGY-CHANGELOG.md through current; git log v0.1.0-plan1..HEAD --stat. Read projects/owasp-llm/cycles/2026/ exhaustively as the reference cycle layout. Locate the authoritative ASI Top 10 entry definitions and ASI community vote at OWASP working-group sources (HANDOFF §9 item 10 — actual paths resolved during planning). Check docs/REVIEWERS.md or create docs/REVIEWERS-asi.md — ASI rubric reviewer and statistical reviewer must be identified separately and may need to differ from LLM reviewers per Rock's independence preference. Do not re-brainstorm HANDOFF or PRD; do not modify either. Do NOT modify engine/ unless a discovered need is a documented methodology change with semver bump and applies to LLM project too. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/YYYY-MM-DD-asi-cycle.md. The plan MUST start with an "Inherited constraints from Phases 1-7" section listing: (i) engine-neutrality is the point — no engine/ edits without semver bump + LLM re-run; (ii) single-channel modeling for ASI per HANDOFF §7.4 — both ASI corpora share selection bias; (iii) Plan 5 is the structural reference, mirror its task order; (iv) reviewer independence per Rock's preference, separate REVIEWERS file admissible; (v) all integrity controls from Plans 1-5 apply unchanged. Target tag: v2.0.0-plan8 (major bump iff any engine change; minor v1.3.0-plan8 if pure reuse). --ultrathink
```

### Plan-execution prompt

```
/using-superpowers Read docs/superpowers/plans/YYYY-MM-DD-asi-cycle.md. Confirm the "Inherited constraints from Phases 1-7" section is present, especially engine-neutrality. Invoke the Superpowers executing-plans skill. After EVERY commit, run git diff v1.0.0-plan5..HEAD -- engine/ — any non-empty diff is a methodology change that requires a changelog entry, a semver bump decision, and a re-run of the LLM Plan 5 reproduction bundle to confirm no regression. Commit per task with messages tagged (Plan 8). When all tasks pass and PRD §9.5 criteria hold, bump methodology-changelog (major if engine touched, minor if pure reuse), tag accordingly, update PRD phase-map. The ASI cycle's report must surface the HANDOFF §7.4 standing caveat about agentic-corpora-share-bias. --ultrathink
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
