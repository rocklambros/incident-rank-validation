# Handoff: Incident-Rank Validation Framework

Version: 1.0
Owner: Rock Lambros (effort lead, OWASP GenAI Top 10 work)
Date authored: 2026-05-19
Status: design approved through Section 3, Sections 4 and 5 proposed and pending review
Purpose of this file: a self-contained carry-over so work can resume in a new standalone repository with zero prior conversation context.

---

## 0. How to use this file in the new repo

1. Create the new standalone repository (see Section 7 for name and posture). Keep the three referenced source repos read-only and untouched.
2. Copy this file into the new repo at `docs/HANDOFF.md` and commit it as the first commit on a feature branch.
3. Start a fresh Claude Code session in the new repo. Tell it: "Read docs/HANDOFF.md. This is an approved design spec. Continue from Section 8."
4. The next formal step is the Superpowers `writing-plans` skill, using this document as the approved spec input. Do not re-brainstorm. The design decisions in Sections 2 through 6 are settled. Open questions in Section 9 are resolved during planning, not by reopening settled decisions.
5. Honor the pre-registration discipline in Section 6 from the first line of code. The integrity controls are the point of the exercise, not overhead.

---

## 1. Context and provenance

Rock leads the 2026 OWASP Top 10 for LLM Applications update and wants data-driven decisions. The community voted on all 2026 candidate entries. The question this framework answers: does the community vote ranking match what real-world incident data shows.

Source material locations on the authoring machine (read-only inputs, do not modify):

- Vote results: `~/github_projects/GenAI-LLM-Top10/2026/polling/results/OWASP_Top10_LLM_Candidates_Voting_Results_2026.xlsx`. Sheets: `Analysis` (narrative of how Rock ranked), `Results` (CASE 1 and CASE 2 tables), `Raw Results (Anonymized)`.
- 2026 entry definitions: `~/github_projects/GenAI-LLM-Top10/2026/LLM01_*.md` through `LLM10_*.md` (10 incumbents) and `~/github_projects/GenAI-LLM-Top10/2026/new_entry_candidates/*.md` (10 candidates). 20 entries total.
- Incident corpus A: `~/github_projects/genai_agentic_incidents` (`data/incidents.json`, 7,714 incidents, auto-refreshes weekly). Audit at `~/github_projects/genai_agentic_incidents/claudedocs/owasp-mapping-quality-audit.md`.
- Incident corpus B: `~/github_projects/www-project-top-10-for-large-language-model-applications/initiatives/agent_security_initiative/ASI Agentic Exploits & Incidents/ASI_Agentic_Exploits_Incidents.md` (human-curated, ASI-axis, small-N).
- Temporal anchor: `~/github_projects/www-project-top-10-for-large-language-model-applications/Archive/2_0_voting/` (prior voting round, for vote-stability cross-check).

Working posture Rock expects: brutally honest, evidence over assertion, surface the strongest counterargument before being asked, name tradeoffs explicitly, no marketing language, no fabricated metrics.

---

## 2. The vote data, as it actually is

The `Results` sheet has two constructions:

- CASE 1: incumbents fixed at distinctness 5.0, four low-distinctness candidates rolled into incumbents. 16 ranked line items.
- CASE 2 (the authoritative deployable 2026 ranking, "Adjusted Rank"): incumbents assigned the average distinctness of remaining candidates so new candidates compete fairly. 16 ranked line items.

CASE 2 order: 1 LLM01 Prompt Injection, 2 LLM02 Sensitive Information Disclosure, 3 LLM03 Supply Chain, 4 LLM06 Excessive Agency, 5 Persistent Memory Poisoning (new), 6 LLM04 Data and Model Poisoning, 7 LLM08 Vector and Embedding Weaknesses, 8 LLM10 Unbounded Consumption, 9 LLM09 Misinformation, 10 MCP Tool Interface Exploitation (new), 11 LLM07 Hidden Context Exposure, 12 LLM05 Improper Output Handling, 13 Model Misalignment, 14 Inference-Time Side-Channel Disclosure, 15 Weaponized LLM Abuse, 16 Model Scheming and Deceptive Alignment.

Four candidates were rolled into incumbents in CASE 1: cross-modal-safety-bypass into LLM01, llm-artifact-promotion-trust-failure into LLM03, systemic-insecure-code-generation into LLM05, compositional-finetuning-alignment-subversion into LLM04.

---

## 3. Critical constraint: the incident corpus is a weak measurement instrument

The audit (`owasp-mapping-quality-audit.md`, 2026-05-19, N=7,714) found:

- F1: 83.3 percent carry `owasp_llm`, but 98.3 percent of the corpus is machine-labeled with no human OWASP review. Coverage is recall of a heuristic, not accuracy.
- F2 (critical): `ingest_cve_nvd_expanded.py` seeds every CVE with `["LLM03"]` / `["ASI04"]` before refinement. About 907 entries are bare `["LLM03"]`, about 768 are the exact `LLM03 + ASI04` double default. Treat CVE-class single-`LLM03` as unknown, not supply chain.
- F3 (critical): hand spot-check found roughly 1 of 6 default-seeded entries defensible. Label quality is bimodal: a small well-labeled head of famous incidents, a long default-seeded tail.
- F4: the `owasp_llm` distribution (LLM05 40 percent, LLM03 25 percent, LLM09 25 percent, LLM04/08/10 near zero) is the shape of the classifier ruleset, not the threat landscape. LLM04, LLM08, LLM10 are near-absent because no ingest pathway emits them. This is a structural blind spot, not evidence of low real-world prevalence.
- F5: 99 percent of `mitre_atlas` is OWASP-to-ATLAS backfill, circular for the 58 ATLAS-sourced rows. ATLAS is not an independent cross-check.
- F6: `quality_tier == reviewed` means catalogued, not human-reviewed. Only 1.6 percent `curated` carries a human-confirmed label.

Consequence for the method: the incident-derived ranking is a noisy, contaminated, partially blind estimate. It is never treated as truth. Low incident count for an entry can mean rare in reality or invisible to the classifier, and the method must distinguish these, never conflate them. The corpus is a strong discovery index and a weak measurement instrument. The README's four-taxonomy framing overstates three of the four.

Corpus strengths that are real and usable: deterministic, idempotent, CI-drift-stable pipeline, every entry has a resolvable URL, dedupe transitive-closure logic is sound.

---

## 4. Settled requirement decisions (do not reopen)

| Decision | Resolution |
|---|---|
| Purpose | All of: validate and defend the vote, allow the analysis to change the ranking, publish the methodology, internal sanity check. Maximum rigor, pre-registered, publishable. |
| Calibration | Build a hand-labeled gold set AND triangulate against a second independent corpus. Maximum rigor. |
| Second corpus | OWASP ASI Exploits and Incidents tracker (human-curated, ASI-axis, independent bias from corpus A). Prior voting archive used only as a vote-stability temporal anchor, not a real-world signal. |
| Crosswalk authorship | Claude drafts the per-entry classification rubric from entry definitions only, blind to vote order. Rock adjudicates boundary and exclusion cells. Rubric is frozen and hash-locked before any concordance number exists. Single-author correlated-error risk accepted and mitigated by the rationale table plus reporting mapping uncertainty as a model input. No clean-room agent. |
| Native taxonomy labels | Non-authoritative for every corpus. The 2025 LLM codes are not a join key and not ground truth. Incidents are re-classified directly against the current cycle's frozen rubric. At most, a corpus native label may enter as one non-authoritative weak classifier feature, and Rock may veto even that. |
| Unit of analysis | All 20 2026 entries scored independently. Compare the 16 CASE 2 items to the vote. Separately test the 4 rollup decisions as a pre-registered finding type. |
| Decision rule | Bayesian inference engine produces a posterior over the incident ranking. A tiered concordance plus per-entry probabilistic flag decision layer sits on the posterior. A non-Bayesian robustness twin is reported alongside so the headline is not model-captive. |
| Repo posture | Private now, public-ready. Public-grade controls from the first commit. Flip to public on publication. |
| Reusability | Standalone repo. Reusable across cycles and across projects (different Top 10 lists), with swappable incident corpora. |
| Dual purpose | The framework serves the OWASP LLM Top 10 AND the OWASP Agentic (ASI) Top 10. Same principle, different ranked taxonomy. See Section 7.4. |

---

## 5. The design (five sections)

### 5.1 Architecture and integrity (approved)

A single forward pipeline with frozen gates:

```
current-cycle entry definitions
  -> [1] classification rubric (frozen, versioned, vote-blind)
  -> [2] classify both corpora to entries
  -> [3] gold set: stratified hand-labels -> per-entry precision/recall with uncertainty
  -> [4] Bayesian measurement-error model: latent prevalence per entry, two channels, blind spots censored
  -> [5] posterior over incident ranking + non-Bayesian robustness twin
  -> [6] decision layer: vote vs posterior, tier concordance, per-entry flag, rollup sub-test, robustness surface
  -> [7] outputs: concordance report, flag list, threats register, pre-reg diff, reproduction bundle
```

Engine vs cycle data split. The engine (classification harness, Bayesian model, decision layer, report and pre-reg tooling) is stable and semver-versioned with a methodology changelog. Everything that varies by project or year is data in a project-cycle directory. Year-over-year comparability holds at the methodology level only. Entries get renamed and renumbered each cycle, so per-entry prevalence does not trend across years. Only the process metric ("did the vote match incidents this cycle") trends. This limitation belongs in the report template, not a footnote.

Corpus adapter abstraction. The engine never sees a source schema. Each corpus has an adapter that emits a canonical incident record: `id, date, text(title/description/impact), severity(normalized, may be unknown), source_class, provenance/quality, native_labels(non-authoritative metadata only), source_url, bias_profile`. The `bias_profile` is mandatory and declared per adapter. The Bayesian triangulation requires each channel's declared bias structure, which prevents a future maintainer from adding a corpus and treating it as clean. Source-specific quarantine rules (drop bare `["LLM03"]` CVE singletons) live in the adapter, declared in its bias profile.

Snapshotting. Corpus A auto-refreshes weekly. A live pull would make a pre-registered analysis irreproducible. Each cycle vendors a content-hashed snapshot plus `provenance.json` (source repo, commit SHA, pull date, adapter version). The engine reads only frozen snapshots. The referenced repos are read once per corpus per cycle, read-only. This is also what keeps the source repos clean.

Pre-registration as a tool mechanism. The CLI is phased: `prereg` (writes and hash-locks rubric, primary spec, robustness list, flag threshold, statistic) then `classify` then `calibrate` then `infer` then `decide`/`report`. The `decide` phase refuses to run unless a committed, hash-matching prereg exists and the vote data was absent from the inputs hashed during `classify` and `infer`. The tool structurally cannot join the vote before the crosswalk is frozen. A rule that must hold every time is a mechanism, not a discipline.

### 5.2 Rubric and classifier (approved)

Two distinct artifacts.

Artifact 1, the frozen rubric (Claude drafts, Rock adjudicates, frozen, vote-blind). Per entry: id, canonical name, in-scope statement, explicit exclusions and pairwise boundary rules against adjacent entries, positive indicators, negative indicators, expected co-occurrence pairs. Boundary cells Rock adjudicates. Genuine 50/50 calls are recorded as both labels with ambiguity, and the ambiguity propagates into the model as label uncertainty rather than being resolved by fiat. Committed and hash-locked in the `prereg` phase. Rock's adjudication log is committed and timestamped before `classify`.

Artifact 2, the classifier (an instrument with measured error, not a source of truth). Stage 1: deterministic indicator and keyword pass from the rubric, auditable and reproducible. Stage 2: model-assisted adjudication only for ambiguous or multi-label incidents, prompted with the frozen rubric, emitting a rationale and confidence per assignment. Pinned model, pinned prompt, fixed seed for reproducibility. The classifier's accuracy is not asserted. It is estimated per-entry by the gold set and de-biased inside the posterior.

Out-of-scope sink. Most CVE-class and generic incidents map to no entry. An out-of-scope incident is not evidence of low prevalence for anything. A bare `["LLM03"]` default-seed CVE the rubric cannot place goes to the sink, not to LLM03. This is where the contamination quarantine executes.

Rollup sub-test. Each rolled-up candidate gets its own rubric entry and is classified independently of the parent it was folded into. Test whether the candidate carries a large distinct incident cluster the parent does not absorb. Direction and magnitude reported per rollup. Pre-registered finding type.

### 5.3 Gold set and measurement model (approved)

Honest limit stated first: the gold set does not create signal where the corpora have none. For a structural blind-spot entry it yields a low measured recall with quantified uncertainty, which proves the blind spot rather than asserting it, and stops "few incidents" being misread as "rare in reality." That distinction is the scientific payoff.

Two-frame sampling, because recall cannot be estimated from the classifier's own positives. Precision frame: stratified sample of classifier-positive assignments per entry, yields per-entry false-positive rate. Recall/coverage frame: a sample drawn independently of the classifier label, stratified by `source_class` and confidence, oversampling rare and blind-spot entries, low-confidence and multi-label assignments, the out-of-scope sink, and a named contamination stratum (bare `["LLM03"]` CVE-default rows).

The contamination stratum is measured, not blanket-dropped. Of bare-default CVE rows, what fraction actually evidence some entry and which. This recovers true signal and gives the model a measured leak rate instead of a guess.

Coding protocol. Dual independent coding against the byte-identical frozen rubric, blind to classifier label and blind to vote. Krippendorff's alpha reported. Disagreements adjudicated on the record by a third coder (Rock or a CODEOWNER). Adjudication log is a committed artifact. Exact sample size comes from a power calculation. Order of magnitude is several hundred to about 1,000 labels, driven by the confidence-interval width needed on the rare entries, where the honest result may still be indeterminate.

Measurement model output per entry e and corpus c. Primary: per-entry precision and recall as Beta posteriors from the labeled counts. Upgrade and robustness: a block confusion matrix among the declared overlap clusters only, because errors flow along the boundary pairs the rubric named, not at random. Primary is the simple version so the method is not hostage to a matrix that cannot be populated.

### 5.4 Bayesian inference and triangulation (proposed, pending review)

Goal: a posterior over each entry's latent prevalence, then over the ranking, accounting for classifier error, contamination, blind spots, and multiple corpora.

Model sketch, to be confirmed by prior-predictive checks in the new repo:

- Latent incident intensity lambda_e >= 0 per entry e, the quantity ranked on.
- Observed classified count y_{e,c} per corpus c is a noisy function of the truth through the measurement model: true positives discounted by recall r_{e,c}, false positives added through (1 - precision) leakage along the declared overlap structure, contamination stratum contributing a measured leak.
- Likelihood: Negative-Binomial on counts for over-dispersion (Poisson is too tight given reporting bursts and clustering). Robustness spec: Dirichlet-Multinomial on prevalence shares. Choice documented via prior predictive checks and the data.
- Priors: weakly-informative on lambda_e scaled to corpus size. Beta priors on r_{e,c} and p_{e,c} centered on the gold-set estimates with the gold-set sample size as pseudo-counts. The gold set literally parameterizes these priors. That is the entire point of calibration. Leak parameters priored from the block confusion in the upgrade spec, or a precision-implied scalar in the primary spec.
- Blind spot and partial identification: where the gold-set recall interval is uninformative, lambda_e is only weakly identified and the posterior stays wide by construction. Low counts plus unknown recall must yield a wide posterior, never a low one. Verify with a synthetic blind-spot entry simulation that confirms the posterior is wide, not low.
- Triangulation: corpora share lambda_e but have separate measurement parameters. If bias profiles declare correlated selection bias between channels, add a shared bias factor with a prior on the correlation. Do not assume independence. The posterior tightens only to the extent the channels are genuinely independent. This keeps triangulation honest rather than a width-shrinking trick.
- Temporal: a recency-weighting or trailing-window spec as a robustness axis, since the Top 10 is about current risk. Primary window pre-registered, surface reports all.
- Inference: NUTS via a pinned probabilistic programming library (numpyro or PyMC, pinned version). Diagnostics gated: R-hat <= 1.01, sufficient effective sample size, zero post-warmup divergences. A run that fails diagnostics fails loudly and emits no report.
- Output: posterior draws of lambda_e, ranked per draw, giving a per-entry rank distribution, a full-ordering posterior, and P(entry in tier).

### 5.5 Decision layer and outputs (proposed, pending review)

- Vote side: CASE 2 ranking mapped to pre-registered tiers (Top 5, 6 to 10, 11 to 20 for the LLM project, adjusted per the target Top 10 structure for other projects).
- Concordance headline: posterior distribution of tier agreement (weighted Cohen's kappa and a tier confusion matrix) between vote tiers and incident tiers, integrated over the posterior, reported as median plus credible interval. Supporting: posterior of Spearman rho and Kendall tau, plus a permutation null answering whether observed concordance beats chance.
- Per-entry probabilistic flag: flag entry e if P(incident tier != vote tier) > tau_flag (tau_flag pre-registered, for example 0.80) AND the divergence direction is consistent across all robustness specs AND the entry is not in the indeterminate or blind-spot set. A wide posterior cannot exceed tau_flag, so blind spots are excluded by construction. Verify this in tests. A flag triggers documented working-group adjudication and never an automatic ranking change. Direction reported: vote over-ranks, vote under-ranks, or indeterminate.
- Rollup sub-test output: per rolled-up candidate, posterior of its independent incident intensity and P(it carries a distinct cluster the parent does not absorb), reported as rollup supported, contradicted, or indeterminate.
- Non-Bayesian robustness twin: the plain tiered point estimate from de-biased counts with no posterior, computed independently. If the twin and the Bayesian headline disagree in direction, that disagreement is a reported finding, not reconciled away.
- Threats-to-validity register: a shipped structured artifact enumerating every known bias (audit F1 through F6, corpus selection bias, crosswalk single-author risk, gold-set coder subjectivity, blind spots, cross-year and cross-project non-comparability) with the design's mitigation or bound and the residual risk. Part of the publishable output.
- Pre-registration diff: the final report includes a pre-reg vs actual diff. Any deviation is disclosed with rationale.
- Reproduction bundle: a single command regenerates the full report from frozen cycle inputs. Snapshot hashes, provenance, engine version, and the dependency lockfile are recorded in the report header.

---

## 6. Integrity controls (the reason the exercise exists)

1. Pre-registration committed and hash-locked before any concordance number exists: rubric, prevalence-signal primary spec, robustness spec list, flag threshold, statistic.
2. Vote-blindness enforced by the phase gate. Vote data is not an input to `classify`, `calibrate`, or `infer`. It enters only at `decide`.
3. Native taxonomy labels never used as a join key or ground truth.
4. One pre-registered primary specification. Everything else is a declared robustness spec. The full robustness surface is reported whole, never mined for the flattering cell.
5. Single-author crosswalk risk is disclosed in the threats register and bounded by the rationale table and by carrying mapping ambiguity into the model as uncertainty.
6. Public-grade supply-chain controls from the first commit: pinned dependencies with a lockfile (NIST SP 800-218), gitleaks secret scanning, semgrep SAST gate on executable additions, a vulnerability disclosure stub, and a NOTICE file with attribution for vendored CC-BY-4.0 (corpus A data) and OWASP (corpus B) snapshots.

---

## 7. The standalone, reusable, dual-purpose repo

### 7.1 Name and posture

Suggested repo name: `incident-rank-validation` (taxonomy-neutral on purpose). Private now, public-ready. Python project.

### 7.2 Layout

```
incident-rank-validation/
  engine/                     stable, semver, methodology changelog
    adapters/                 base interface plus one module per corpus
    classify/  calibrate/  model/  decide/  report/
    prereg/                   hash-lock and phase-gate enforcement
  projects/
    owasp-llm/
      project.toml            taxonomy source, default corpora, bias profiles, tier definition
      cycles/
        2026/
          prereg/             frozen rubric, specs, threshold
          taxonomy/           snapshot of the 20 entry definitions
          corpora/            genai_agentic/ , owasp_asi/  (snapshots + provenance)
          goldset/            stratified hand-labels + adjudication log
          vote/               CASE 2 ranking, joined only at decide
          results/            generated report, posterior, flag list, prereg diff
    owasp-asi/
      project.toml
      cycles/
        20xx/ ...
  tests/                      engine unit tests plus a synthetic end-to-end cycle
  pyproject.toml + lockfile   pinned dependencies
```

### 7.3 Scope boundary (YAGNI, so the build does not bloat)

Adapters are plain Python modules implementing one interface. Projects and cycles are directories. Config is one typed file per project. No plugin marketplace, no web UI, no general meta-analysis platform. Reusability means next year, same team, maybe a new adapter, a new taxonomy snapshot, or a new project. Nothing more.

### 7.4 Dual purpose: OWASP LLM Top 10 and OWASP Agentic Top 10

The engine is taxonomy-neutral. A project is the triple (ranked taxonomy, community vote, incident corpora). `owasp-llm` is one project. `owasp-asi` is another. Switching projects changes data, never engine code.

The only Agentic-specific work when that project runs:

- Locate the authoritative OWASP Agentic (ASI) Top 10 entry definitions and the ASI community vote. These are different artifacts from the LLM ones and live in the OWASP Agentic Security Initiative material. Resolve their exact location at that time.
- The OWASP ASI Exploits and Incidents tracker is the natural curated corpus for the ASI project and is purpose-built for the Agentic Top 10.
- Real caveat to record now: for the ASI project both natural corpora are agentic-focused and may share selection bias (both over-represent agentic tooling and CVE incidents, both under-represent non-agentic harm). Triangulation only adds value when channel biases are independent. The project must assess corpus bias-independence before relying on triangulation. If the two channels are not independent enough, either declare the bias correlation in the model (Section 5.4 shared bias factor) or source a third, independently biased ASI corpus. Do not let the posterior tighten on correlated errors.

Everything else (rubric, classifier, gold set, Bayesian model, decision layer, integrity controls) transfers unchanged.

---

## 8. What to do next, concretely

1. Create the repo and the layout in Section 7.2. First commit on a feature branch (the LLM Top 10 repo uses PR-only workflow, expect the same discipline here).
2. Copy this file to `docs/HANDOFF.md`.
3. Run the Superpowers `writing-plans` skill with this document as the approved spec. The plan must sequence: engine skeleton and CLI phase gate, the canonical schema and adapter base, the two LLM-cycle adapters, the rubric drafting and adjudication workflow, the classifier, the gold-set sampler and coding protocol, the Bayesian model with prior-predictive and blind-spot simulation checks, the decision layer, the report and reproduction bundle, and the public-grade controls.
4. Build engine-first with the synthetic end-to-end cycle test before touching real corpus data. This protects pre-registration: the machinery is proven on synthetic data before the real rubric is frozen.
5. Only then run the LLM 2026 cycle: draft rubric vote-blind, Rock adjudicates, freeze, classify, build gold set, infer, decide, report.

---

## 9. Open questions to resolve during planning (not by reopening Section 4)

1. Exact gold-set sample size and stratum allocation, from a power calculation targeting usable confidence-interval width on the rare entries.
2. Who codes the gold set (two human domain coders plus a third adjudicator) and the time budget. This is the human bottleneck.
3. Bayesian likelihood family final choice (Negative-Binomial vs Dirichlet-Multinomial) decided by prior-predictive checks, with the alternative kept as a robustness spec.
4. The numeric value of tau_flag, pre-registered before `decide`.
5. The pinned classifier model and the pinned probabilistic programming library and versions.
6. Whether any corpus native label is used at all as a weak classifier feature. Rock leaned toward no. Default to no unless a measured benefit is shown on the gold set.
7. The primary recency window and weighting for the temporal spec.
8. For the ASI project later: authoritative ASI Top 10 definitions and vote location, and the corpus bias-independence assessment in Section 7.4.

---

## 10. Memory pointers on the authoring machine

Auto-memory for the GenAI-LLM-Top10 project records this work at:
`~/.claude/projects/-Users-klambros-github-projects-GenAI-LLM-Top10/memory/`

- `incident-dataset-measurement-weaknesses.md`: the corpus A weaknesses in Section 3.
- `ranking-validation-methodology.md`: the converged design summary and a pointer to this file.

These are project-scoped to the LLM Top 10 repo. The new repo should rely on this file as the single source of truth, not on that machine-local memory.
