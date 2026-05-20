# PRD — Incident-Rank Validation Framework

Version: 1.0
Owner: Rock Lambros
Date authored: 2026-05-20
Source of truth for *what* and *why*: `docs/HANDOFF.md` v2.5.
Source of truth for Plan 1 execution detail: `docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md` v5 (tag `v0.1.0-plan1`).

## 0. Purpose of this document

`HANDOFF.md` is the approved design spec. It is a "PRD+" in the sense that it locks the methodology, but it is not phase-structured. After Plan 1 landed in 30+ commits, "what is the next phase and how do I pick it up" is no longer obvious from reading HANDOFF top-to-bottom.

This file is the phase index. It does three things and only three things:

1. Lists every phase, in order, with a single status (DONE / NEXT / BLOCKED / FUTURE) and the literal "pickup" command for a fresh Claude Code session.
2. For each phase: states the goal, the inputs it consumes, the deliverables it produces, the gates that must clear before it can start, and the acceptance criteria that prove it is done.
3. Names the cross-cutting blockers (reviewer identification, frame-coverage audit feasibility) that gate later phases regardless of engineering progress.

This file does *not* duplicate HANDOFF methodology, redesign anything, or contain task-level detail. Per-phase implementation plans are produced by invoking the Superpowers `writing-plans` skill against HANDOFF + the phase entry in this file, and land under `docs/superpowers/plans/<date>-<slug>.md`. Plan 1 is the worked example of that pattern.

## 1. How to pick up the next phase

In a fresh Claude Code session, in the repo root:

```
Read docs/HANDOFF.md (the approved spec) and docs/PRD.md (the phase map).
Pick up the next phase marked NEXT in docs/PRD.md.
Follow the pickup command literally. Do not re-brainstorm; do not modify HANDOFF.
```

The pickup command for each phase is the single line under "Pickup command" in that phase's section. It tells you which Superpowers skill to invoke and which HANDOFF sections + PRD phase to read as input.

Hard rules for every phase pickup, regardless of phase:

- HANDOFF v2.5 §6 (integrity controls) and §6 control 11 (information firewall) override any contradictory engineering convenience. If a phase plan appears to relax an integrity control, the plan is wrong, not the control.
- Never use the words "validation against reality" or "ground truth" for any concordance output. The standing scope-and-bias caveat (HANDOFF §6 control 10) is non-negotiable language.
- Plan 1 produced the engine; no phase from Plan 2 onward extends the engine *for its own sake*. Engine changes are admissible only when a phase's deliverables require them, and any such change is a methodology-changelog entry (`docs/METHODOLOGY-CHANGELOG.md`) and a semver bump.
- The phase ordering is not a suggestion. Plans 2, 3, 4 can be interleaved in parts (see §5 Critical path notes), but Plan 5 (real LLM 2026 cycle) cannot start until the listed Plan 5 prerequisites all clear, and cannot publish until reviewers are external (see §10 Reviewer-identification track).

## 2. Phase map

| Phase | Title | Status | Pickup command |
|---|---|---|---|
| Plan 1 | Engine + Synthetic Cycle | DONE (tag `v0.1.0-plan1`) | n/a |
| Plan 2 | Corpus A adapter + snapshot + per-stratum bias profiles | NEXT | "Invoke `writing-plans` against HANDOFF v2.5 §3, §4 (Corpus A is a mixture), §5.1 (corpus adapter abstraction + snapshotting), and PRD §3 (Plan 2). Produce `docs/superpowers/plans/<date>-corpus-a-adapter.md`." |
| Plan 3 | Rubric drafting, Rock adjudication, independent-reviewer signoff workflow | BLOCKED on external rubric reviewer (REVIEWERS.md INTERIM); workflow construction may proceed | "Invoke `writing-plans` against HANDOFF v2.5 §4 (Crosswalk authorship), §5.2 (rubric artifact), §6 control 11(b)(d)(e), and PRD §4 (Plan 3). Produce `docs/superpowers/plans/<date>-rubric-freeze-workflow.md`." |
| Plan 4 | Gold-set sampler, k-fold CV calibration, coding protocol, staffing + power calc | BLOCKED on gold-set coder identification (HANDOFF §9 item 1, item 2) and Plan 2 (needs real adapter to draw against); engine-side stubs from Plan 1 may be promoted | "Invoke `writing-plans` against HANDOFF v2.5 §5.3, §6 control 11(b)(c), §9 items 1–2, and PRD §5 (Plan 4). Produce `docs/superpowers/plans/<date>-goldset-calibration.md`." |
| Plan 5 | Real LLM 2026 cycle — Stage-2 classifier on RunPod, classify, infer, decide, publish | BLOCKED on Plan 2, Plan 3 (rubric frozen + reviewer-signed), Plan 4 (gold-set built), REVIEWERS external for *publishable* output | "Invoke `writing-plans` against HANDOFF v2.5 §5.2 Stage-2, §5.4, §5.5, §6 control 11, §7.5 (GPU + RunPod), and PRD §6 (Plan 5). Produce `docs/superpowers/plans/<date>-llm2026-cycle.md`." |
| Plan 6 | Corpus B corroboration cross-check (qualitative; not a posterior input) | FUTURE — runnable anytime after Plan 5 cycle outputs exist | "Invoke `writing-plans` against HANDOFF v2.5 §4 (Corpus B role), §5.5 (corpus B corroboration bullet), and PRD §7 (Plan 6). Produce `docs/superpowers/plans/<date>-corpus-b-corroboration.md`." |
| Plan 7 | Staged frame-coverage audit extension (per-entry; gated; optional) | FUTURE — only if external-reference list is feasible per HANDOFF §9 item 9 | "Invoke `writing-plans` against HANDOFF v2.5 §4 (Frame-coverage audit), §6 control 7, §9 item 9, and PRD §8 (Plan 7). Produce `docs/superpowers/plans/<date>-frame-coverage-audit.md`." |
| Plan 8 | OWASP ASI Top 10 cycle (different taxonomy, same engine) | FUTURE — engine reuse, no engine changes | "Invoke `writing-plans` against HANDOFF v2.5 §7.4 and PRD §9 (Plan 8). Produce `docs/superpowers/plans/<date>-asi-cycle.md`." |

Status definitions:

- **DONE**: deliverables landed on main, acceptance criteria all hold, tag present.
- **NEXT**: prerequisites cleared; no engineering reason not to start; this is what "pick up the next phase" means today.
- **BLOCKED**: a named external prerequisite (reviewer identity, coder identity, audit-list construction) is unmet. Internal engineering scaffolding *for* the phase may still be admissible; the phase itself cannot close until the blocker clears.
- **FUTURE**: deliberately not started; either depends on an upstream phase output, or is an optional declared extension that may never ship without integrity loss.

## 3. Plan 2 — Corpus A adapter + snapshot + per-stratum bias profiles

### 3.1 Status: NEXT

### 3.2 Goal

Produce a working corpus A adapter that emits the canonical incident-record schema (`engine/schema.py`) from a frozen, content-hashed snapshot of `~/github_projects/genai_agentic_incidents/data/incidents.json`, with **per-sub-corpus stratum bias profiles** declared on each stratum (HANDOFF §3 mixture + §5.1 corpus-adapter paragraph). No rubric, no classification, no inference. Just: real bytes flowing through the canonical schema with the contamination quarantine rules active.

### 3.3 Prerequisites (must hold before starting)

1. Plan 1 tag `v0.1.0-plan1` present (✅).
2. `engine/schema.py` and `engine/adapters/base.py` unchanged from Plan 1 acceptance, OR any change is a documented methodology-changelog entry with a semver bump.
3. Source corpus accessible read-only at `~/github_projects/genai_agentic_incidents/`.
4. Snapshot policy from HANDOFF §5.1: each cycle vendors a content-hashed snapshot plus `provenance.json` (source repo, commit SHA, pull date, adapter version). Snapshot lands under `projects/owasp-llm/cycles/2026/corpora/genai_agentic/`.

### 3.4 Inputs

- Read-only source: `~/github_projects/genai_agentic_incidents/data/incidents.json` (7,714 incidents, weekly auto-refresh).
- Audit reference: `~/github_projects/genai_agentic_incidents/claudedocs/owasp-mapping-quality-audit.md` — quote-only, not a config source.
- Stratum definitions (HANDOFF §3 mixture paragraph):
  - `corpus`: `security` (~7,350), `ai-harm` (~364).
  - `category`: `real-world` (~5,791), `vulnerability-disclosure` (~1,571), research/threat-report remainder.
  - Contamination stratum: bare `["LLM03"]` CVE-default rows (HANDOFF §3 F2; ~907 entries).
- Schema and adapter ABC from Plan 1.

### 3.5 Deliverables

1. `engine/adapters/genai_agentic.py` — concrete adapter implementing `engine/adapters/base.py:CorpusAdapter`. Emits one canonical record per source incident.
2. `engine/adapters/genai_agentic_bias.py` (or co-located) — declared `BiasProfile` per stratum, with per-stratum quarantine rules. Bare-LLM03 quarantine lives here per HANDOFF §5.1 ("Source-specific quarantine rules ... live in the adapter, declared in its bias profile").
3. Snapshot vendoring script: `engine/cli/snapshot.py` (or extend existing) — pulls the source, content-hashes, writes `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/incidents.json` plus `provenance.json` (source repo, commit SHA, pull date, adapter version, engine version).
4. Drift hook integration: the existing drift detector (`engine/snapshot/drift.py`) consumes the new snapshot path. Per-entry count drift + burst detection runs at snapshot time.
5. `projects/owasp-llm/cycles/2026/` directory created with the snapshot vendored.
6. Severity-defaulting disclosure: adapter records the severity-`"Medium"`-default-on-missing artifact per HANDOFF §3 ("`severity` is defaulted to ... a zero unknown-severity rate is itself an artifact"). Severity is emitted as `unknown` when the source-side default-flag is detected, not silently propagated as `"Medium"`.
7. Future-dated row repair: HANDOFF §4 Temporal — adapter drops or repairs rows dated after the snapshot date. Behavior committed in code, not config.
8. Tests under `tests/unit/test_adapter_genai_agentic.py` covering: schema round-trip, every stratum populated, bare-LLM03 quarantine fires on default-seed CVE rows, severity-default detection, future-dated repair, snapshot hash byte-stability across platforms.

### 3.6 Acceptance criteria

1. `uv run pytest tests/unit/test_adapter_genai_agentic.py -v` green.
2. `uv run pytest -v` (full suite) green — no regressions in Plan 1 tests.
3. `uv run mypy engine tests`, `uv run ruff check .`, `uv run semgrep --config .semgrep.yml --error engine/` — all zero errors.
4. Snapshot vendored to `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/`; `provenance.json` carries all six fields.
5. Drift detector runs on the vendored snapshot and emits a drift report (first run will be the baseline; the report's existence and shape is what is verified).
6. Per-stratum incident counts in the adapter output match the audit-reference counts within tolerance (declared in the test, not in the report).
7. Methodology changelog entry: "0.2.0 (Plan 2): genai_agentic corpus A adapter, per-stratum bias profiles, snapshot vendoring."
8. Commit message: `feat(adapters): genai_agentic corpus A adapter + per-stratum bias profiles + snapshot vendoring (Plan 2)`.
9. Tag `v0.2.0-plan2` on the merge commit.

### 3.7 Out of scope

- No rubric. No classification of the real data. The classifier from Plan 1 must not run against this snapshot (the rubric is not frozen yet; running classify pre-freeze violates HANDOFF §6 control 11(b)).
- No gold set. Gold-set sampling is Plan 4 and depends on this snapshot existing.
- No inference. NUTS does not touch real data until Plan 5.

### 3.8 Risks and known gotchas

- Source repo refreshes weekly. Run the snapshot script once and freeze. Re-running mid-Plan-2 invalidates any test that pinned the hash.
- `severity` defaulting is an artifact in the source ingest. The adapter must detect "this was defaulted by the source pipeline" not "this was set to Medium by a human" — practically, look at the source row's provenance/quality field for the `defaulted` indicator, or treat missing-severity-field-pre-default as the trigger. Confirm with the source repo's `claudedocs/`.
- The `owasp_llm` field on incoming records is non-authoritative (HANDOFF §4 row). The adapter passes it through as `native_labels` metadata only and never lets it influence `bias_profile` or quarantine decisions.

### 3.9 Pickup command

```
Read docs/HANDOFF.md (focus on §3, §4 Corpus-A-is-a-mixture row, §5.1 corpus-adapter and snapshotting paragraphs) and docs/PRD.md §3 (Plan 2). Invoke the Superpowers writing-plans skill with these as the approved spec and write the per-task plan to docs/superpowers/plans/<today>-corpus-a-adapter.md. Do not re-brainstorm; do not modify HANDOFF or PRD. The plan must satisfy every acceptance criterion in PRD §3.6.
```

## 4. Plan 3 — Rubric drafting + adjudication + independent-reviewer signoff workflow

### 4.1 Status: BLOCKED (workflow construction may proceed; freeze cannot)

Blocker: external rubric reviewer not identified. Per HANDOFF §4 Crosswalk authorship, the rubric must be signed off by "an independent OWASP working-group member who is not the ranking author." `docs/REVIEWERS.md` records the interim single-author state (Rock = rubric reviewer = ranking author). Plan 3 *workflow scaffolding* (drafting tool, adjudication log format, attestation file format) may be built. The actual rubric *freeze* requires the external signoff.

### 4.2 Goal

Produce the frozen, vote-blind, hash-locked rubric for the 2026 LLM Top 10 cycle, with:

- per-entry rubric content (HANDOFF §5.2 Artifact 1: id, canonical name, in-scope statement, exclusions, pairwise boundary rules, positive indicators, negative indicators, expected co-occurrence pairs);
- Rock's adjudication log (committed, timestamped, machine-readable);
- the independent rubric-reviewer's attestation (committed signed text + sha256 per HANDOFF §6 control 11(e));
- the rubric-drafting attestation (HANDOFF §6 control 11(d)) declaring whether the drafter viewed corpus samples before drafting (if yes, the report carries a "corpus-informed rubric" caveat);
- the prereg manifest's `rubric_hash` populated from the frozen rubric and locked via `engine/prereg/lock.py`.

### 4.3 Prerequisites (must hold before starting)

1. Plan 1 tag present.
2. Plan 2 NOT required for workflow scaffolding. Plan 2 IS required if the drafter intends to look at any real-corpus samples (HANDOFF §6 control 11(d) attestation).
3. The 20 2026 entry definitions (read-only, located at `~/github_projects/GenAI-LLM-Top10/2026/LLM01_*.md` through `LLM10_*.md` plus `new_entry_candidates/*.md`). Drafting reads these.
4. Vote results MUST NOT be loaded into the drafting session. HANDOFF §6 control 2 requires vote-blindness; the CLI structurally cannot join the vote before `decide`, but the drafter (human + Claude) must also avoid reading the `Analysis` and `Results` sheets while drafting. This is procedural, not enforced by the engine.
5. For *freeze* and Plan 4 start: external rubric reviewer identified in `docs/REVIEWERS.md` with attestation file + sha256.

### 4.4 Inputs

- 20 entry definitions (read-only paths above).
- HANDOFF §5.2 (rubric content shape), §6 control 11(b)(d)(e) (firewall mechanics around drafting).
- `engine/prereg/rubric_attestation.py` (Plan 1 stub) — populated here.
- `engine/prereg/signoff.py` (Plan 1) — consumed here for the reviewer signoff payload.

### 4.5 Deliverables

1. `projects/owasp-llm/cycles/2026/prereg/rubric.json` (or `.toml`) — per-entry rubric content, machine-readable, hash-stable.
2. `projects/owasp-llm/cycles/2026/prereg/adjudication_log.md` — Rock's per-cell adjudications, dated, with rationale. Boundary cells flagged with the `(both-labels, ambiguous)` marker per HANDOFF §5.2 ("Genuine 50/50 calls are recorded as both labels with ambiguity, and the ambiguity propagates into the model as label uncertainty rather than being resolved by fiat").
3. `projects/owasp-llm/cycles/2026/prereg/rubric_attestation.json` — populated `viewed_corpus_samples` field (per HANDOFF §6 control 11(d)).
4. `projects/owasp-llm/cycles/2026/prereg/rubric_reviewer_signoff.txt` — committed signed attestation text (when external reviewer available).
5. `projects/owasp-llm/cycles/2026/prereg/rubric_reviewer_signoff.sha256` — recorded hash.
6. `projects/owasp-llm/cycles/2026/prereg/manifest.json` (or `.toml`) — prereg manifest with `rubric_hash`, `signed_at` (git-derived per M8), `reviewer_identity`, `viewed_results_before_signoff=false`.
7. Drafting tool / workflow doc: `docs/RUBRIC-WORKFLOW.md` documenting the procedural vote-blindness rule, the boundary-cell adjudication procedure, and the freeze step.
8. Tests: extend `tests/unit/test_prereg.py` to cover rubric-hash stability, attestation-required-before-classify gate, viewed-results-disclosure surfacing in the report.

### 4.6 Acceptance criteria

1. `rubric.json` contains all 20 entries (16 ranked + 4 rolled-up candidates per HANDOFF §2 + §5.2 rollup sub-test).
2. Every entry has all eight required fields per HANDOFF §5.2 Artifact 1.
3. Every pairwise-adjacent entry pair has an explicit boundary rule, or is documented as "no overlap, by inspection."
4. `adjudication_log.md` covers every boundary cell flagged during drafting; no boundary cell is unresolved.
5. `rubric_attestation.json` `viewed_corpus_samples` is set (true or false, with rationale). If true, the report contract from Plan 5 must surface the "corpus-informed rubric" caveat.
6. `engine/prereg/lock.py` accepts the rubric and emits a hash. The hash is stable across re-runs (no timestamps in the hashed body).
7. CLI `classify` refuses to run if the rubric attestation is missing (regression test).
8. `non_publishable=True` if the reviewer signoff is missing OR the reviewer identity matches the ranking author per `docs/REVIEWERS.md` (regression test).
9. Commit messages tagged `(Plan 3)`; tag `v0.3.0-plan3` on the freeze commit.

### 4.7 Out of scope

- No classification. `classify` is invoked first in Plan 4 (against the gold-set sample only, not the full corpus) and Plan 5 (full corpus).
- No vote handling. Vote loading is Plan 5.

### 4.8 Risks and known gotchas

- The biggest risk is procedural vote leakage during drafting. If the drafter sees the CASE 2 ordering, the rubric is contaminated and the cycle is non-publishable, regardless of what the manifest says. Mitigation: a separate Claude Code session for drafting, with `~/github_projects/GenAI-LLM-Top10/2026/polling/` not accessible, OR explicit "do not read the polling directory" instruction at session start and recorded in the rubric attestation.
- Rolled-up candidates (HANDOFF §2: cross-modal-safety-bypass, llm-artifact-promotion-trust-failure, systemic-insecure-code-generation, compositional-finetuning-alignment-subversion) each get their own rubric entry per HANDOFF §5.2 rollup sub-test. Forgetting these breaks Plan 5's rollup analysis.
- Boundary cells are *not* resolved by fiat. Genuine 50/50 cases stay as `(both-labels, ambiguous)` and propagate as label uncertainty into Plan 4's gold-set coding. Forcing a single label here destroys downstream measurement.

### 4.9 Pickup command

```
Read docs/HANDOFF.md (focus on §4 Crosswalk authorship, §5.2, §6 control 11(b)(d)(e)) and docs/PRD.md §4 (Plan 3). Confirm docs/REVIEWERS.md state before assuming rubric freeze is possible — INTERIM mode means workflow scaffolding only. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/<today>-rubric-freeze-workflow.md.
```

## 5. Plan 4 — Gold-set sampler, k-fold CV calibration, coding protocol, staffing + power calc

### 5.1 Status: BLOCKED

Blockers:

- Plan 2 must be DONE — the sampler draws from the real adapter output.
- Plan 3 rubric must be FROZEN — coders cannot label against an unfrozen rubric (HANDOFF §6 control 11(b)).
- Two human domain coders + a third adjudicator must be named (HANDOFF §9 item 2). This is the human bottleneck.
- Power calculation must be produced and committed (HANDOFF §9 item 1) before sample size is set.

### 5.2 Goal

Produce the calibration gold set, the per-entry per-stratum precision/recall Beta posteriors, and the k-fold cross-validation fold-variance disclosure (HANDOFF §6 control 11(c)). The gold set is HANDOFF's *critical path* (§5.3: "the load-bearing artifact and the project critical path").

### 5.3 Prerequisites (must hold before starting)

1. Plans 2 and 3 DONE; rubric hash-locked.
2. `docs/GOLDSET-STAFFING.md` (new) committed with: two human domain coder names, third adjudicator name, time budget, power calculation, target sample size with stratum allocation.
3. Reviewer identities updated in `docs/REVIEWERS.md` if coder = reviewer (must not be).

### 5.4 Inputs

- Vendored corpus A snapshot from Plan 2.
- Frozen rubric from Plan 3 (hash-locked).
- HANDOFF §5.3 (gold-set + measurement model), §6 control 11(b)(c).
- `engine/calibrate/sampler.py` (Plan 1 stub) — promoted here.
- `engine/calibrate/cv.py` (Plan 1 stub) — promoted here.
- `engine/calibrate/beta.py` (Plan 1) — consumed.

### 5.5 Deliverables

1. `engine/calibrate/sampler.py` — full implementation: two-frame sampling (precision frame + recall/coverage frame), stratified by `corpus_stratum`, `source_class`, classifier confidence; oversamples rare and classifier-blind entries; includes out-of-scope sink + named contamination stratum (bare-LLM03). HANDOFF §5.3 paragraph 3.
2. `engine/calibrate/cv.py` — full implementation: k=5 fold cross-validation per HANDOFF §6 control 11(c); fold variance computed and emitted alongside the Beta posteriors.
3. `projects/owasp-llm/cycles/2026/goldset/sample.json` — the actual sampled incident ids (bound to snapshot hash per HANDOFF §5.1 "The gold-set artifact records and is bound to the snapshot content hash; a cycle refuses to run if the gold-set snapshot hash and the cycle snapshot hash differ").
4. `projects/owasp-llm/cycles/2026/goldset/labels.json` — dual independent coder labels per incident (HANDOFF §5.3 "Coding protocol" paragraph). Schema: incident_id, coder_a_label, coder_b_label, adjudicator_label (if needed), confidence, ambiguous flag.
5. `projects/owasp-llm/cycles/2026/goldset/adjudication_log.json` — third-coder adjudications, on the record.
6. `projects/owasp-llm/cycles/2026/goldset/krippendorff_alpha.json` — alpha per stratum + overall.
7. `projects/owasp-llm/cycles/2026/calibrate/posteriors.json` — per-entry per-stratum Beta posteriors with gold-set-sample-size pseudo-counts (HANDOFF §5.4 "Priors" bullet).
8. `projects/owasp-llm/cycles/2026/calibrate/fold_variance.json` — k=5 fold variance per entry per stratum.
9. Coding tool / interface: a minimal labeling UI or notebook to feed coders, blind to classifier label and blind to vote (HANDOFF §5.3 "Coding protocol" paragraph).
10. Tests: schema/contracts for sampler outputs; CV variance computation; snapshot-hash binding regression.

### 5.6 Acceptance criteria

1. Power calculation produced and committed; target sample size justified against confidence-interval-width-on-rare-entries criterion.
2. Two-frame sample drawn; precision-frame and recall-frame sizes match the power calc.
3. Contamination stratum (bare-LLM03 CVE-default rows) explicitly sampled; HANDOFF §5.3 measured-not-blanket-dropped requirement satisfied.
4. Out-of-scope sink sampled at the rate required to estimate the leak rate.
5. Dual coding completed against the byte-identical frozen rubric; coders demonstrably blind to classifier label and vote (procedure documented in coding interface; auditable).
6. Krippendorff's alpha computed per stratum and overall; reported in `krippendorff_alpha.json`.
7. Adjudication log is exhaustive: every disagreement is on the record with the third coder's reasoning.
8. k=5 fold variance computed; fold-variance disclosure renders in the report contract (verified in Plan 5 acceptance, smoke-tested here).
9. Gold-set snapshot-hash binding enforced: a cycle that consumes gold-set artifacts against a mismatched snapshot hash refuses to run.
10. Methodology changelog: "0.4.0 (Plan 4): gold set built, calibration posteriors produced, k=5 CV active."
11. Tag `v0.4.0-plan4`.

### 5.7 Out of scope

- No full-corpus classification. Stage-1 classifier runs only against the sampled incidents to populate the precision frame; full-corpus classification is Plan 5.
- No NUTS inference yet. The Beta posteriors are inputs to Plan 5's NUTS run.

### 5.8 Risks and known gotchas

- Coder time is the bottleneck, not engineering. HANDOFF §5.3: "order of magnitude is several hundred to about 1,000 labels." Plan 4 cannot finish faster than the coders.
- Coder fatigue degrades alpha. Mitigation: batch sampling delivered to coders in <=200-incident batches; alpha computed per batch as well as overall to detect drift.
- The frame-blind incidents (the thing the gold set *cannot* estimate) must not silently appear as "low recall" entries. The frame-blind detection from Plan 1's measurability map runs first; gold-set sampling skips frame-blind entries for the recall frame and reports them as unmeasurable (HANDOFF §3 paragraph 8: "classifier-blind and frame-blind are different failures and must never be conflated"). Verify this in the sampler tests.

### 5.9 Pickup command

```
Read docs/HANDOFF.md (focus on §5.3, §5.4 Priors bullet, §6 control 11(b)(c), §9 items 1 and 2) and docs/PRD.md §5 (Plan 4). Confirm docs/REVIEWERS.md and docs/GOLDSET-STAFFING.md state before starting — both must be populated with external identities. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/<today>-goldset-calibration.md.
```

## 6. Plan 5 — Real LLM 2026 cycle (Stage-2 classifier on RunPod, classify, infer, decide, publish)

### 6.1 Status: BLOCKED (publication requires external reviewers; internal-only run permitted with `non_publishable=True`)

Blockers:

- Plans 2, 3, 4 all DONE.
- For *publishable* output: both external reviewers identified per `docs/REVIEWERS.md` (rubric reviewer and statistical reviewer, neither = Rock).
- `docs/PROVISIONING-PLAN.md` populated with concrete GPU type, model identity, weight hash, batch size, wall-time budget, cost ceiling (default $500, HANDOFF §7.5 + M9).
- Two-cycle parity holdout (HANDOFF §11 v2.3→v2.4 row M17): 30-day reviewer audit before any external sharing.

### 6.2 Goal

Run the real 2026 LLM Top 10 cycle end-to-end: Stage-1 classify the full corpus A snapshot, Stage-2 LLM-classify the ambiguous and multi-label residue on RunPod, run NUTS measurement-error inference per HANDOFF §5.4, build the vote-rank posterior, compute the transparency-first concordance, produce the measurability map + flag list + threats register + pre-reg diff + reproduction bundle. Publish (or stamp non-publishable, per reviewer state).

### 6.3 Prerequisites (must hold before starting)

1. Plan 1, 2, 3, 4 tags all present.
2. Reviewer identification: see §10 Reviewer-identification track. INTERIM mode permits internal-only execution; publishable output requires external identities committed and attested.
3. `docs/PROVISIONING-PLAN.md` fully populated for this cycle's Stage-2 run. Per-workload estimate written to `projects/owasp-llm/cycles/2026/provenance/local_run_estimate.json` per HANDOFF §7.5 v2.5.
4. Reviewer signoffs (rubric + statistical) committed BEFORE the first `infer` run for the cycle, per HANDOFF §6 control 11(e). A signoff with `viewed_results_before_signoff=True` is disclosed in the report but does not block; absent signoff blocks publishable output only.
5. Two-cycle parity prerequisite acknowledged (M17): a second cycle (e.g., a synthetic re-run with the same code) is held for 30 days post-Plan-5-completion before any external sharing.

### 6.4 Inputs

- Plan 2's vendored corpus A snapshot (`projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/`).
- Plan 3's frozen rubric (`projects/owasp-llm/cycles/2026/prereg/rubric.json` + manifest).
- Plan 4's gold-set posteriors (`projects/owasp-llm/cycles/2026/calibrate/posteriors.json`).
- Vote data from `~/github_projects/GenAI-LLM-Top10/2026/polling/results/OWASP_Top10_LLM_Candidates_Voting_Results_2026.xlsx`, sheets `Results` and `Raw Results (Anonymized)`. Loaded only at `decide`, never earlier (CLI phase gate enforces).
- Stage-2 model: pinned identity + weight provenance hash + deterministic seed in the Stage-2 manifest per HANDOFF §5.2 + §7.5.

### 6.5 Deliverables

1. Stage-2 implementation: `engine/classify/stage2.py` (promoted from `stage2_protocol.py` stub). RunPod client integration, delimiter-fenced incident text, ignore-instructions-in-incident-text discipline (HANDOFF §5.2 Stage-2 paragraph).
2. Stage-2 manifest: `projects/owasp-llm/cycles/2026/stage2_manifest.json` — model id, weight hash, prompt hash, batch size, seed, RunPod job ids.
3. Full-corpus classification output: `projects/owasp-llm/cycles/2026/classify/labeled_incidents.json`.
4. NUTS posterior draws: `projects/owasp-llm/cycles/2026/infer/posterior.npz` (or `.nc`), with diagnostics gated (R-hat ≤ 1.01, ESS sufficient per manifest fraction, zero post-warmup divergences) per HANDOFF §5.4 Inference bullet. Run refuses to emit a report if diagnostics fail.
5. Vote-rank posterior: `projects/owasp-llm/cycles/2026/vote/vote_posterior.json` (bootstrap over `Raw Results (Anonymized)`).
6. Decision-layer outputs: `projects/owasp-llm/cycles/2026/results/measurability_map.json`, `concordance.json`, `flag_list.json`, `rollup_subtest.json`, `selection_bias.json` (Kruskal-Wallis), `robustness_spread.json` (cherry-picking spread per HANDOFF §6 control 11(g)).
7. Twin output: `projects/owasp-llm/cycles/2026/results/twin.json` and `twin_vs_nuts_agreement.json`.
8. Pre-reg diff: `projects/owasp-llm/cycles/2026/results/prereg_diff.json`.
9. Threats register: `projects/owasp-llm/cycles/2026/results/threats_register.json`.
10. Final report: `projects/owasp-llm/cycles/2026/results/report.md` rendered with the measurability map leading, standing scope-and-bias caveat, coverage ratio, denominator, pre-registered measurability minimum as tag, PRE-PUBLISH CHECKLIST footer (M6), and `non_publishable` flag derived from reviewer state.
11. Reproduction bundle: `projects/owasp-llm/cycles/2026/results/repro_bundle.tar.gz` — single command regenerates the full report from frozen inputs (HANDOFF §5.5 Reproduction bundle bullet).
12. RunPod cost log: actual spend vs cost ceiling, committed to `projects/owasp-llm/cycles/2026/provenance/runpod_cost.json`.

### 6.6 Acceptance criteria

1. Full pipeline runs end-to-end via the CLI phase sequence: `prereg` → `classify` → `calibrate` → `infer` → `decide` → `report`. Each phase gate enforces its preconditions.
2. NUTS diagnostics pass per HANDOFF §5.4; run refuses to emit a report if they fail.
3. Never-falsely-low release gate (the Plan-1 prior-predictive test on synthetic) is re-run and passes against the real-cycle hyperparameters.
4. Report leads with the measurability map (HANDOFF §5.5 first bullet). Headline kappa is the quadratic-weighted Cohen's kappa with the binary tier-membership kappa reported alongside.
5. Standing scope-and-bias caveat present verbatim per HANDOFF §6 control 10.
6. Coverage ratio and measurable-subset denominator present on the headline.
7. `non_publishable` field correctly derived (True iff any reviewer = ranking author or any reviewer attestation missing).
8. Pre-reg diff is non-empty *only* if a deviation occurred; every deviation has a rationale.
9. Reproduction bundle regenerates the report byte-identically (or within MCSE for NUTS-driven cells) on a clean checkout — verified by the cross-platform diff CI job (M5).
10. RunPod actual cost ≤ cost ceiling in `PROVISIONING-PLAN.md`. Overrun aborts and logs as post-hoc per HANDOFF §6 control 11(f).
11. Methodology changelog: "1.0.0 (Plan 5): first real LLM 2026 cycle." Semver major bump.
12. Tag `v1.0.0-plan5-internal` if reviewers are INTERIM; `v1.0.0-plan5` only if external reviewers attested.
13. Two-cycle parity holdout (M17): a synthetic re-run executed against the same engine version produces matching headline shape; the 30-day audit window opens. External sharing is gated on this window closing.

### 6.7 Out of scope

- No corpus B corroboration. That is Plan 6 (qualitative artifact, not a posterior input). Plan 5 must not pull corpus B into the inference.
- No frame-coverage audit. That is Plan 7 (optional, declared extension).
- No ASI cycle. That is Plan 8.

### 6.8 Risks and known gotchas

- Vote leakage during interactive debugging. The CLI phase gate prevents code paths from joining vote data before `decide`, but a developer who manually peeks at the spreadsheet while iterating on `infer` violates the spirit of HANDOFF §6 control 2 and contaminates the run. Mitigation: a separate Claude Code session for `decide`, with the vote file inaccessible during `infer` development.
- Stage-2 model swap mid-cycle is forbidden (HANDOFF §7.5: "mid-cycle model swap is not"). If a Stage-2 run fails partway through and the model identity needs to change, the cycle restarts from `classify` with a new Stage-2 manifest.
- NUTS-on-CPU is slow for 7,000-incident-class workloads. Budget multiple hours for `infer`. Do not move NUTS to GPU under any circumstance — HANDOFF §7.5: "GPU non-determinism in cuBLAS reductions changes posteriors run-to-run and is methodology-breaking."
- Selection-bias result (Kruskal-Wallis, M14): if p < 0.05, the headline kappa is computed over a vote-correlated subset. This is a *report flag*, not a stop condition. The report explains the implication; the cycle still publishes.
- Two-cycle parity audit (M17) is a *publication* gate, not an execution gate. The cycle can run, produce artifacts, and be internally reviewed during the 30-day window. External sharing is what M17 blocks.

### 6.9 Pickup command

```
Read docs/HANDOFF.md (focus on §5.2 Stage-2, §5.4, §5.5, §6 control 11, §7.5 GPU + RunPod rule) and docs/PRD.md §6 (Plan 5) and §10 (Reviewer-identification track). Confirm Plans 2, 3, 4 tags are present, docs/REVIEWERS.md, docs/PROVISIONING-PLAN.md, and docs/GOLDSET-STAFFING.md states before starting. Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/<today>-llm2026-cycle.md.
```

## 7. Plan 6 — Corpus B corroboration cross-check

### 7.1 Status: FUTURE (runnable after Plan 5 cycle outputs exist)

### 7.2 Goal

Produce the declared agree/disagree artifact comparing corpus A's curated-head labels against corpus B labels on the incidents they share. This is qualitative corroboration of corpus A's curated head only — never a posterior input (HANDOFF §4 Corpus B role row, §5.5 corpus B corroboration bullet).

### 7.3 Prerequisites

1. Plan 5 outputs exist (corpus A labeled incidents).
2. Corpus B accessible: `~/github_projects/www-project-top-10-for-large-language-model-applications/initiatives/agent_security_initiative/ASI Agentic Exploits & Incidents/ASI_Agentic_Exploits_Incidents.md`.

### 7.4 Deliverables

1. Adapter: `engine/adapters/owasp_asi.py` — emits canonical records from corpus B (corroboration-only; bias_profile flagged as `qualitative_corroboration_only`).
2. Cross-check tool: `engine/decide/corpus_b_corroboration.py` — computes incident-id overlap (or text-match overlap if id-overlap is weak), reports per-incident agree/disagree, surfaces systematic divergence.
3. Output: `projects/owasp-llm/cycles/2026/results/corpus_b_corroboration.json`.
4. Report integration: a section in `report.md` reports the agreement rate and any systematic divergence as a finding (HANDOFF §4: "Systematic divergence is a published finding, never a silent posterior adjustment").
5. Tests asserting corpus B is *never* in the likelihood (regression check on `engine/model/inference.py`).

### 7.5 Acceptance criteria

1. Corroboration artifact computed and reported.
2. Inference module unchanged — corpus B does not enter the likelihood.
3. Report section reads as a declared artifact, not a posterior input.
4. Tag `v1.1.0-plan6`.

### 7.6 Pickup command

```
Read docs/HANDOFF.md (focus on §4 Corpus B role row, §5.5 corpus B corroboration bullet) and docs/PRD.md §7 (Plan 6). Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/<today>-corpus-b-corroboration.md.
```

## 8. Plan 7 — Staged frame-coverage audit extension

### 8.1 Status: FUTURE (optional; only runnable if external-reference list is feasible per HANDOFF §9 item 9)

### 8.2 Goal

Build a per-entry, pre-registered, gated external frame-coverage audit that can raise a specific entry from `unmeasurable` to `bounded`, only by producing a frame-coverage bound with quantified uncertainty against an external reference list of incidents *not* sourced from CVE or harm databases (HANDOFF §4 Frame-coverage-audit row).

### 8.3 Prerequisites

1. Acceptance criterion for the external reference list resolved (HANDOFF §9 item 9). This is the substantive blocker — if no reference list is feasible for an entry, the audit cannot be built for that entry, and the entry stays unmeasurable. That is the honest outcome, not a failure.
2. Plan 5 cycle outputs exist (the measurability map identifies which entries are candidates for upgrade).

### 8.4 Deliverables

1. Per-entry reference-list construction tool (likely manual + Claude-assisted; not a fully automated module).
2. Per-entry frame-coverage bound: `projects/owasp-llm/cycles/2026/framecoverage/<entry_id>/bound.json`.
3. Measurability map upgrade: the entry transitions from `frame-blind-unmeasurable` to `classifier-blind-but-bounded` (or `measurable` if classifier recall is also adequate).
4. Methodology changelog entry documenting the audit construction per entry.

### 8.5 Acceptance criteria

1. Each audited entry has a documented external reference list, a bound, and a quantified uncertainty.
2. Measurability map regenerated; upgrades visible.
3. No entry transitions out of `unmeasurable` without the bound + uncertainty.
4. Tag `v1.2.0-plan7` (or per-entry sub-tags if audits are episodic).

### 8.6 Risks

- The audit cannot estimate what is structurally invisible. If a class of incidents never appears in *any* reference list (because no one records it), the entry stays unmeasurable forever. That is methodology, not failure.

### 8.7 Pickup command

```
Read docs/HANDOFF.md (focus on §4 Frame-coverage-audit row, §6 control 7, §9 item 9) and docs/PRD.md §8 (Plan 7). Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/<today>-frame-coverage-audit.md.
```

## 9. Plan 8 — OWASP ASI Top 10 cycle

### 9.1 Status: FUTURE (engine reuse; no engine changes)

### 9.2 Goal

Run the full pipeline for the OWASP Agentic Security Initiative (ASI) Top 10. Same engine; different taxonomy, different vote, different incident corpus (HANDOFF §7.4).

### 9.3 Prerequisites

1. Authoritative ASI Top 10 entry definitions located (HANDOFF §7.4 + §9 item 10).
2. ASI community vote located.
3. ASI external reviewers identified (rubric + statistical), distinct from Rock and from the LLM project reviewers if Rock asks for full independence.

### 9.4 Deliverables

1. `projects/owasp-asi/project.toml` populated.
2. `projects/owasp-asi/cycles/<year>/` populated with the same artifact layout as `projects/owasp-llm/cycles/2026/`.
3. ASI-specific adapter for the curated corpus (`engine/adapters/owasp_asi_exploits.py`, distinct from the corroboration-only Plan 6 adapter).
4. Full cycle outputs in `projects/owasp-asi/cycles/<year>/results/`.

### 9.5 Acceptance criteria

1. Engine modules under `engine/` are unchanged (or changes are documented methodology-changelog entries with semver bumps; reuse is the *point*).
2. Cycle runs end-to-end; same integrity controls as Plan 5.
3. The HANDOFF §7.4 caveat is surfaced in the report: "for the ASI project both natural corpora are agentic-focused and may share selection bias. The single-channel plus declared-stratum design from v2.0 applies. Do not reintroduce a triangulation claim without a genuinely independent, comparable-N corpus and a measured bias-independence assessment."

### 9.6 Pickup command

```
Read docs/HANDOFF.md (focus on §7.4) and docs/PRD.md §9 (Plan 8). Invoke the Superpowers writing-plans skill and write the per-task plan to docs/superpowers/plans/<today>-asi-cycle.md.
```

## 10. Reviewer-identification track (cross-cutting)

This is not a phase. It is a continuous, human-bottleneck workstream that gates publishable output in Plan 5 and Plan 8.

### 10.1 Current state

`docs/REVIEWERS.md` is INTERIM as of 2026-05-20: Rock = rubric reviewer = statistical reviewer = ranking author. Any manifest signed in this state derives `non_publishable=True`.

### 10.2 What needs to happen, by phase

- **Before Plan 4 can start**: gold-set coders identified in `docs/GOLDSET-STAFFING.md` (HANDOFF §9 item 2). Coders must not be Rock.
- **Before Plan 5 can publish (not run)**: external rubric reviewer (an OWASP working-group member who is not Rock and does not report to Rock in OWASP work) AND external statistical reviewer (independent of Rock and of the rubric reviewer), both identified, attested, and signed off BEFORE the first `infer` run.
- **Before Plan 8 publishable**: ASI rubric + statistical reviewers identified, attested, signed off.

### 10.3 What enables Plan 5 internal-only execution despite INTERIM

Per `docs/REVIEWERS.md`: Plan 5 may run as internal-only with `non_publishable=True`. The engine continues to enforce all integrity controls; only the publication permission flips. This permits cycle dry-runs against real data before external reviewers are seated, surfacing engineering issues without committing to a publication date.

### 10.4 Action

Rock is the only person who can move this. Engineering cannot. The action item lives in `docs/REVIEWERS.md` "Path to publishable" section.

### 10.5 How to update REVIEWERS.md when a reviewer agrees

This sub-section exists so a future Rock — or a collaborator — knows the mechanics without re-deriving them from HANDOFF §6 control 11(e). Reviewer attestation is not free-form; it is a content-hashed artifact the engine reads.

**Trigger:** a reviewer has agreed to sign off. You have their name, affiliation, and either a signed-text attestation already in hand OR a commitment to provide one by a named date.

**Steps (do them in this order; the engine's `verify_committed` will reject out-of-order states):**

1. **Confirm independence.** The reviewer must not be Rock and, for the OWASP work, must not report to Rock. For the statistical reviewer, also must not be the rubric reviewer. If independence is in doubt, the reviewer is INTERIM and the engine will still derive `non_publishable=True`. Better to leave INTERIM than to claim independence that doesn't hold.

2. **Write the attestation file.** Path convention: `docs/REVIEWERS/<reviewer-name-slug>-<role>.txt`, where `<role>` is `rubric` or `statistical`. Slug is lowercase-hyphenated. Example: `docs/REVIEWERS/jane-doe-rubric.txt`. The file is plain text and contains the reviewer's signed statement: who they are, what they reviewed, the date they signed, what they viewed before signing (per HANDOFF §6 control 11(e) — `viewed_results_before_signoff`), and their attestation that the review was independent. The exact wording is the reviewer's call; the structural fields are mandatory.

3. **Compute the SHA-256.** Run `shasum -a 256 docs/REVIEWERS/<reviewer-name-slug>-<role>.txt`. Record the hash — it goes into REVIEWERS.md and into the cycle's prereg manifest.

4. **Update `docs/REVIEWERS.md`.** Replace the INTERIM block for that role with the populated block from the file's existing "Path to publishable" section:

   ```
   ### Rubric reviewer (or Statistical reviewer)
   - Name: <reviewer name>
   - Affiliation: <affiliation; MUST NOT be Rock or report to Rock in OWASP work>
   - Availability commitment: <signoff schedule, e.g., "will sign within 2 weeks of rubric freeze">
   - Attestation file: docs/REVIEWERS/<reviewer-name-slug>-<role>.txt
   - Attestation SHA-256: <hash from step 3>
   ```

   Also update the top-of-file status line if both reviewers are now populated: `## Current state (YYYY-MM-DD): EXTERNAL — both reviewers independent`. Date is today's date, not the reviewer's signoff date.

5. **Commit.** Commit message: `docs(reviewers): identify <role> reviewer <name> + attestation hash`. The attestation file and REVIEWERS.md change go in the same commit so the hash and the file land atomically. If they don't, `verify_committed` will reject the manifest at the next `infer` run.

6. **Update the cycle's prereg manifest** (only if a cycle is currently in flight; otherwise this happens when the cycle starts). The manifest's `reviewer_identity` and `reviewer_attestation_hash` fields get populated; `signed_at` is derived from `git log` of the attestation file per M8 (don't set it manually — the engine will reject a hand-set timestamp). `viewed_results_before_signoff` is set from the attestation text. Run `engine/prereg/lock.py` to re-lock.

7. **Re-derive `non_publishable`.** With both reviewers EXTERNAL and attested, the next `decide` run will derive `non_publishable=False` automatically. Don't set it by hand — the engine derives it from reviewer state. If you want to verify before running the full pipeline, the prereg manifest will show the derived value.

**Common pitfalls:**

- Setting `signed_at` manually in the manifest. The engine derives it from `git log` of the attestation file (HANDOFF v2.4 M8). A mismatch fails verification.
- Committing the attestation file in one commit and updating REVIEWERS.md + the hash in a later commit. `verify_committed` reads the hash in REVIEWERS.md and recomputes from the file at HEAD; if the file at HEAD has been edited since the hash was recorded, verification fails. Atomic commit prevents this.
- Treating a verbal "yes I'll sign" as an attestation. The attestation is the file. Until the file is committed with the hash recorded, the reviewer is INTERIM regardless of any out-of-band conversation.
- Naming the same person as rubric reviewer and statistical reviewer. HANDOFF §4 Crosswalk authorship requires "a separate independent statistical reviewer." Same person = INTERIM for the statistical role.
- Forgetting to update the cycle manifest after updating REVIEWERS.md. The cycle's `non_publishable` derivation reads from the manifest, not from REVIEWERS.md directly. If the cycle was locked with INTERIM reviewers and you later identify EXTERNAL ones, re-lock the manifest.

**Counter-case — coder identification (Plan 4 prerequisite):** the gold-set coders are tracked in `docs/GOLDSET-STAFFING.md`, not `docs/REVIEWERS.md`. That file doesn't exist yet; Plan 4 creates it. The mechanics are similar (names, attestation that they will code blind to classifier label and vote, time-budget commitment) but the file lives separately because coders are not reviewers — they generate the gold-set labels, they don't sign off on rubric or model. Plan 4's pickup command takes care of the file creation; this sub-section is about reviewer attestation only.

## 11. Methodology changelog discipline

Every phase that lands on main bumps `docs/METHODOLOGY-CHANGELOG.md` with a semver entry. Engine changes that alter methodology (likelihood family, hyperparameter defaults, statistic choice, gate behavior) are *major* bumps, regardless of how small the diff. Adapter additions, project additions, and report-template additions are *minor* bumps. Bug fixes that do not change methodology are *patch* bumps.

| Plan | Expected version | Methodology impact |
|---|---|---|
| Plan 1 | 0.1.0 | Engine baseline. |
| Plan 2 | 0.2.0 | Corpus A real adapter, per-stratum bias profiles, severity-default disclosure. |
| Plan 3 | 0.3.0 | Frozen rubric; rubric-attestation contract enforced. |
| Plan 4 | 0.4.0 | Calibration posteriors active; k=5 CV variance disclosed. |
| Plan 5 | 1.0.0 | First real cycle; methodology-claiming output. Major bump. |
| Plan 6 | 1.1.0 | Corpus B corroboration artifact. |
| Plan 7 | 1.2.0+ | Frame-coverage audit; per-entry upgrades. |
| Plan 8 | 2.0.0 | ASI project added; if it forces any engine change, major bump; if pure reuse, minor. |

## 12. What this document is NOT

- Not a re-derivation of HANDOFF. If this document and HANDOFF disagree, HANDOFF wins and this document is wrong.
- Not a task list. The Superpowers `writing-plans` skill produces task lists per phase under `docs/superpowers/plans/`.
- Not a status dashboard. The status field per phase is updated when a phase tag lands; live progress lives in commit messages and PR descriptions.
- Not a defense against scope creep. That is HANDOFF §7.3 (YAGNI scope boundary). This document inherits it.
