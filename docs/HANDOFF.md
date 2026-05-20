# Handoff: Incident-Rank Validation Framework

Version: 2.5
Owner: Rock Lambros (effort lead, OWASP GenAI Top 10 work)
Date authored: 2026-05-19 (v1.0). Revised 2026-05-19 (v2.0). Revised 2026-05-19 (v2.1). Revised 2026-05-19 (v2.2). Revised 2026-05-19 (v2.3). Revised 2026-05-20 (v2.4). Revised 2026-05-20 (v2.5).
Status: v1.0 approved through Section 6. v2.0 closed remediation items 0–10. v2.1 replaced suppression-gate publication with transparency-first publication. v2.2 closed Premortem 2 R1–R33: declared overlap weights, per-stratum exposure semantics, quadratic-weighted kappa, raised N/A threshold (4 measurable), reviewer ID at Plan 1 acceptance, attested signoff, drift enforcement, cross-cycle refusal. v2.3 (2026-05-19) adds the information-firewall discipline (§6 control 11) and GPU/compute posture (§7.5): hyperparameter pre-registration, classifier-rule freeze before gold-set draw, k-fold calibration CV, rubric drafting attestation, reviewer signoff timing + viewed-results disclosure, post-hoc analysis register, robustness-spec cherry-picking adjustment, selection-bias quantification. GPU pinned to CPU for inference; permitted only for Stage-2 LLM classification under RunPod with committed provisioning plan. v2.5 (2026-05-20) makes the GPU provider-selection rule explicit: use RunPod by default; use local Jetson GPU only when a workload can complete in under 30 minutes wall time. Section 11 logs every change.
Purpose of this file: a self-contained carry-over so work can resume in a new standalone repository with zero prior conversation context.

---

## 0. How to use this file in the new repo

1. The standalone repository exists (Section 7 name and posture). The three referenced source repos stay read-only and untouched.
2. This file lives at `docs/HANDOFF.md` and is version 2.0.
3. Start a fresh Claude Code session in the repo. Tell it: "Read docs/HANDOFF.md. This is an approved design spec at v2.0. Continue from Section 8."
4. The next formal step is the Superpowers `writing-plans` skill, using this document as the approved spec input. Do not re-brainstorm. The v2.0 design decisions in Sections 1 through 6 are settled. The v1.0 "do not reopen" guard no longer applies to the items the premortem revised. Section 11 is the authoritative list of what changed and why. Open questions in Section 9 are blocking gates resolved during planning, not by reopening Section 4.
5. Honor the pre-registration discipline in Section 6 from the first line of code. The integrity controls are the point of the exercise, not overhead.

---

## 1. Context and provenance

Rock leads the 2026 OWASP Top 10 for LLM Applications update and wants data-driven decisions. The community voted on all 2026 candidate entries. The question this framework answers, as reframed by the v2.0 premortem: for the subset of entries with demonstrable incident-corpus frame-coverage, does the community vote ranking concord with the incident-derived ranking, with measurement uncertainty modeled on both the incident side and the vote side. Entries the corpus frame cannot observe are reported as unmeasurable, never as low-prevalence.

Source material locations on the authoring machine (read-only inputs, do not modify):

- Vote results: `~/github_projects/GenAI-LLM-Top10/2026/polling/results/OWASP_Top10_LLM_Candidates_Voting_Results_2026.xlsx`. Sheets: `Analysis` (narrative of how Rock ranked), `Results` (CASE 1 and CASE 2 tables), `Raw Results (Anonymized)` (about 1,000 respondent rows, the resampling base for the vote-side posterior).
- 2026 entry definitions: `~/github_projects/GenAI-LLM-Top10/2026/LLM01_*.md` through `LLM10_*.md` (10 incumbents) and `~/github_projects/GenAI-LLM-Top10/2026/new_entry_candidates/*.md` (10 candidates). 20 entries total.
- Incident corpus A: `~/github_projects/genai_agentic_incidents` (`data/incidents.json`, 7,714 incidents, auto-refreshes weekly). Audit at `~/github_projects/genai_agentic_incidents/claudedocs/owasp-mapping-quality-audit.md`. v2.0: corpus A is a mixture, not one instrument (see Section 3, F4 and F-frame, and Section 4).
- Incident corpus B: `~/github_projects/www-project-top-10-for-large-language-model-applications/initiatives/agent_security_initiative/ASI Agentic Exploits & Incidents/ASI_Agentic_Exploits_Incidents.md` (human-curated, ASI-axis, about 68 lines, dozens of incidents). v2.0: reclassified as qualitative corroboration of corpus A's curated head, not a triangulation channel. See Section 4.
- Temporal anchor: `~/github_projects/www-project-top-10-for-large-language-model-applications/Archive/2_0_voting/` (prior voting round, for vote-stability cross-check).

Working posture Rock expects: brutally honest, evidence over assertion, surface the strongest counterargument before being asked, name tradeoffs explicitly, no marketing language, no fabricated metrics.

---

## 2. The vote data, as it actually is

The `Results` sheet has two constructions:

- CASE 1: incumbents fixed at distinctness 5.0, four low-distinctness candidates rolled into incumbents. 16 ranked line items.
- CASE 2 (the authoritative deployable 2026 ranking, "Adjusted Rank"): incumbents assigned the average distinctness of remaining candidates so new candidates compete fairly. 16 ranked line items.

CASE 2 order: 1 LLM01 Prompt Injection, 2 LLM02 Sensitive Information Disclosure, 3 LLM03 Supply Chain, 4 LLM06 Excessive Agency, 5 Persistent Memory Poisoning (new), 6 LLM04 Data and Model Poisoning, 7 LLM08 Vector and Embedding Weaknesses, 8 LLM10 Unbounded Consumption, 9 LLM09 Misinformation, 10 MCP Tool Interface Exploitation (new), 11 LLM07 Hidden Context Exposure, 12 LLM05 Improper Output Handling, 13 Model Misalignment, 14 Inference-Time Side-Channel Disclosure, 15 Weaponized LLM Abuse, 16 Model Scheming and Deceptive Alignment.

Four candidates were rolled into incumbents in CASE 1: cross-modal-safety-bypass into LLM01, llm-artifact-promotion-trust-failure into LLM03, systemic-insecure-code-generation into LLM05, compositional-finetuning-alignment-subversion into LLM04.

The vote is a self-selected convenience sample recruited by broadcast (`polling/comms/`: LinkedIn, Slack, mailing list). It is a measured instrument with its own error, not ground truth. v2.0 builds a vote-side posterior over it (Section 5.4).

---

## 3. Critical constraint: both the classifier and the corpus frame are weak measurement instruments

The audit (`owasp-mapping-quality-audit.md`, 2026-05-19, N=7,714) found:

- F1: 83.3 percent carry `owasp_llm`, but 98.3 percent of the corpus is machine-labeled with no human OWASP review. Coverage is recall of a heuristic, not accuracy.
- F2 (critical): `ingest_cve_nvd_expanded.py` seeds every CVE with `["LLM03"]` / `["ASI04"]` before refinement. About 907 entries are bare `["LLM03"]`, about 768 are the exact `LLM03 + ASI04` double default. Treat CVE-class single-`LLM03` as unknown, not supply chain.
- F3 (critical): hand spot-check found roughly 1 of 6 default-seeded entries defensible. Label quality is bimodal: a small well-labeled head of famous incidents, a long default-seeded tail.
- F4: the `owasp_llm` distribution is the shape of the classifier ruleset, not the threat landscape. LLM04, LLM08, LLM10 are near-absent because no ingest pathway emits them. Structural blind spot, not evidence of low real-world prevalence.
- F5: 99 percent of `mitre_atlas` is OWASP-to-ATLAS backfill, circular for the 58 ATLAS-sourced rows. ATLAS is not an independent cross-check.
- F6: `quality_tier == reviewed` means catalogued, not human-reviewed. Only 1.6 percent `curated` carries a human-confirmed label.
- F-frame (critical, added v2.0): the corpus sampling frame is blind, not only the classifier. Corpus A is built by a CVE/GHSA/OSV keyword crawl plus harm-database ingestion (`scripts/ingest_cve_nvd_expanded.py`). A deployed-application failure that never becomes a CVE or a harm-database row is never eligible to enter the corpus. `map_owasp_and_atlas` line 217 assigns LLM01 only when a literal prompt-injection regex fires; every other attack vector appends to a hard `["LLM03"]` seed. Measured residue: flattened `owasp_llm` is LLM05 3118, LLM09 1929, LLM03 1928, LLM01 366, LLM04 124, LLM10 45, LLM08 25. A gold set sampled from within the corpus estimates classifier-stage recall. It cannot estimate ingestion-frame coverage, because incidents that never entered the frame are absent from the sampling base. Classifier-blind (estimable from within the corpus) and frame-blind (not estimable from within the corpus) are different failures and must never be conflated.
- Mixture (added v2.0): corpus A is not one instrument. The `corpus` field is `security` (7,350) and `ai-harm` (364). The `category` field is `real-world` (5,791), `vulnerability-disclosure` (1,571), and a research/threat-report remainder. These sub-corpora have different selection mechanisms and must carry separate declared bias profiles and separate measurement parameters (Sections 4 and 5.1). `severity` is defaulted to "Medium" in ingest when missing, so a zero "unknown severity" rate is itself an artifact. Severity is not a trustworthy field.

Consequence for the method: the incident-derived ranking is a noisy, contaminated, partially blind estimate. It is never treated as truth. Low incident count for an entry can mean rare in reality, invisible to the classifier, or never sampled by the frame, and the method must distinguish these and never conflate them. An entry without demonstrable frame-coverage is reported as unmeasurable. It is not assigned a low prevalence and it is not flagged. Frame-coverage is raised from unknown to bounded only by the pre-registered staged frame-coverage audit (Section 4), never by within-corpus sampling. The corpus is a strong discovery index and a weak measurement instrument.

Corpus strengths that are real and usable: deterministic, idempotent, CI-drift-stable pipeline, every entry has a resolvable URL, dedupe transitive-closure logic is sound.

---

## 4. Settled requirement decisions (v2.0, supersedes v1.0 where they differ, do not reopen)

| Decision | Resolution |
|---|---|
| Purpose | Validate the ranking only for entries with demonstrable corpus frame-coverage. Entries without it are reported unmeasurable, not flagged. Publish the methodology and the per-entry measurability verdict. Internal sanity check allowed for any entry. Maximum rigor on the measurable subset, honesty about the rest. |
| Frame-coverage audit (staged) | Primary deliverable does not depend on it. A pre-registered, gated, per-entry external frame-coverage audit is a declared extension. It can raise a specific entry from unmeasurable to bounded only if it produces a frame-coverage bound with quantified uncertainty against an external reference list of incidents not sourced from CVE or harm databases. If the audit is never built, affected entries stay unmeasurable. No integrity loss from the extension not shipping, only a missed upside. |
| Calibration | Hand-labeled gold set on the single modeled channel (corpus A), with per-sub-corpus measurement parameters. No triangulation channel. |
| Corpus B role | Qualitative corroboration of corpus A's curated head only. Not a modeled Bayesian channel. The agreement between corpus A's curated-head labels and corpus B on the incidents they share is computed and reported as a declared agree/disagree artifact. Systematic divergence is a published finding, never a silent posterior adjustment. |
| Vote measurement | The vote is a roughly 1,000-respondent convenience sample. Build a vote-rank posterior by resampling respondents (bootstrap over the `Raw Results (Anonymized)` rows). Concordance integrates over both the vote posterior and the incident posterior. The vote is never treated as exact. |
| Crosswalk authorship | Claude drafts the per-entry classification rubric from entry definitions only, blind to vote order. Rock adjudicates boundary and exclusion cells. An independent OWASP working-group member who is not the ranking author signs off on the frozen rubric. A separate independent statistical reviewer signs off on the inference model. Both reviewers MUST be identified by name and affiliation at Plan 1 acceptance (v2.2); if no candidates are identified, Plan 4 (gold-set work) and Plan 5 (real-data cycle) cannot start. Signoffs are *attested* (committed signed text + sha256), not self-declared booleans (v2.2). The signoff artifacts are committed pre-registration inputs, required before `classify`. If no independent reviewer is available at run time, internal-only runs may proceed but the report is stamped "single-author rubric, uncontrolled" and is not publishable. Rubric is frozen and hash-locked before any concordance number exists. |
| Corpus A is a mixture | Declared bias profile per sub-corpus stratum (`corpus` and `category`), not one profile per adapter. The Bayesian model carries stratum-specific measurement parameters. |
| Native taxonomy labels | Non-authoritative for every corpus. The 2025 LLM codes are not a join key and not ground truth. Incidents are re-classified directly against the current cycle's frozen rubric. At most a corpus native label may enter as one non-authoritative weak classifier feature, and Rock may veto even that. |
| Unit of analysis | All 20 2026 entries scored independently. Compare the 16 CASE 2 items to the vote. Separately test the 4 rollup decisions as a pre-registered finding type. |
| Decision rule | Bayesian inference engine produces a posterior over the incident ranking. A tiered concordance plus per-entry probabilistic flag decision layer sits on the joint of the incident posterior and the vote posterior. A non-Bayesian robustness twin is reported alongside so the headline is not model-captive. |
| Temporal | Recency is a pre-registered primary dimension, not only a robustness axis. The corpus spans 1983 to future-dated rows. The adapter drops or repairs rows dated after the snapshot date. |
| Repo posture | Private now, public-ready. Public-grade controls from the first commit. Flip to public on publication, subject to the Section 6 publication gate. |
| Reusability | Standalone repo. Reusable across cycles and across projects (different Top 10 lists), with swappable incident corpora. |
| Dual purpose | The framework serves the OWASP LLM Top 10 AND the OWASP Agentic (ASI) Top 10. Same principle, different ranked taxonomy. See Section 7.4. |

---

## 5. The design (five sections)

### 5.1 Architecture and integrity (approved, v2.0 edits)

A single forward pipeline with frozen gates:

```
current-cycle entry definitions
  -> [1] classification rubric (frozen, versioned, vote-blind, independently reviewed)
  -> [2] classify corpus A to entries (per-stratum)
  -> [3] gold set: stratified hand-labels -> per-entry, per-stratum precision/recall with uncertainty
  -> [4] Bayesian measurement-error model: latent prevalence per entry, single channel,
         per-stratum measurement params, frame-blind entries censored as unmeasurable
  -> [5] posterior over incident ranking + non-Bayesian robustness twin
  -> [5b] vote-rank posterior (bootstrap over respondents), joined only at decide
  -> [6] decision layer: measurability map first, then concordance over both posteriors,
         tier concordance, per-entry flag, rollup sub-test, robustness surface,
         corpus B corroboration cross-check
  -> [7] outputs: measurability map, concordance report, flag list, threats register,
         pre-reg diff, reproduction bundle
```

Engine vs cycle data split. The engine (classification harness, Bayesian model, decision layer, report and pre-reg tooling) is stable and semver-versioned with a methodology changelog. Everything that varies by project or year is data in a project-cycle directory. Year-over-year comparability holds at the methodology level only. Entries get renamed and renumbered each cycle, so per-entry prevalence does not trend across years. Only the process metric ("did the vote match incidents this cycle") trends. This limitation belongs in the report template, not a footnote.

Corpus adapter abstraction. The engine never sees a source schema. Each corpus has an adapter that emits a canonical incident record: `id, date, text(title/description/impact), severity(normalized, may be unknown), source_class, corpus_stratum, provenance/quality, native_labels(non-authoritative metadata only), source_url, bias_profile`. The `bias_profile` is mandatory and declared per sub-corpus stratum, not once per adapter, because corpus A is a mixture (Section 3). The Bayesian model requires each stratum's declared bias structure, which prevents a future maintainer from pooling heterogeneous sub-corpora into one ill-defined channel. Source-specific quarantine rules (drop bare `["LLM03"]` CVE singletons) live in the adapter, declared in its bias profile.

Snapshotting. Corpus A auto-refreshes weekly. A live pull would make a pre-registered analysis irreproducible. Each cycle vendors a content-hashed snapshot plus `provenance.json` (source repo, commit SHA, pull date, adapter version). The engine reads only frozen snapshots. The gold-set artifact records and is bound to the snapshot content hash; a cycle refuses to run if the gold-set snapshot hash and the cycle snapshot hash differ. A between-snapshot drift and anomaly check (per-entry count drift and burst detection across consecutive weekly snapshots) runs at snapshot time. A drifted or anomalous snapshot requires manual sign-off before a cycle consumes it. Adversarial ingestion is a declared threat in the register, because public CVE/GHSA/OSV are open submission surfaces, descriptions are attacker-controlled, and `infer_attack_vector` is pure regex. Snapshot plus hash makes a cycle reproducible. It does not make the snapshot clean, so drift detection plus sign-off is the compensating control.

Pre-registration as a tool mechanism. The CLI is phased: `prereg` (writes and hash-locks rubric, primary spec, robustness list, flag threshold, statistic, measurability gate, independent-reviewer sign-off) then `classify` then `calibrate` then `infer` then `decide`/`report`. The `decide` phase refuses to run unless a committed, hash-matching prereg exists, the independent-reviewer sign-off is present or the run is marked non-publishable, and the vote data was absent from the inputs hashed during `classify` and `infer`. The tool structurally cannot join the vote before the crosswalk is frozen. A rule that must hold every time is a mechanism, not a discipline.

### 5.2 Rubric and classifier (approved, v2.0 edits)

Two distinct artifacts.

Artifact 1, the frozen rubric (Claude drafts, Rock adjudicates, independent OWASP working-group member who is not the ranking author signs off, frozen, vote-blind). Per entry: id, canonical name, in-scope statement, explicit exclusions and pairwise boundary rules against adjacent entries, positive indicators, negative indicators, expected co-occurrence pairs. Boundary cells Rock adjudicates. Genuine 50/50 calls are recorded as both labels with ambiguity, and the ambiguity propagates into the model as label uncertainty rather than being resolved by fiat. Committed and hash-locked in the `prereg` phase. Rock's adjudication log and the independent reviewer's sign-off are committed and timestamped before `classify`.

Artifact 2, the classifier (an instrument with measured error, not a source of truth). Stage 1: deterministic indicator and keyword pass from the rubric, auditable and reproducible. Stage 2: model-assisted adjudication only for ambiguous or multi-label incidents, prompted with the frozen rubric, emitting a rationale and confidence per assignment. Incident text is attacker-controlled, so the Stage 2 prompt enforces instruction and data separation with delimiter fencing of the incident text and ignores instructions found inside it. Pinned model recorded by weight or provenance hash, pinned prompt, fixed seed for reproducibility. The classifier's accuracy is not asserted. It is estimated per-entry and per-stratum by the gold set and de-biased inside the posterior.

Out-of-scope sink. Most CVE-class and generic incidents map to no entry. An out-of-scope incident is not evidence of low prevalence for anything. A bare `["LLM03"]` default-seed CVE the rubric cannot place goes to the sink, not to LLM03. This is where the contamination quarantine executes.

Rollup sub-test. Each rolled-up candidate gets its own rubric entry and is classified independently of the parent it was folded into. Test whether the candidate carries a large distinct incident cluster the parent does not absorb. Direction and magnitude reported per rollup. Pre-registered finding type.

### 5.3 Gold set and measurement model (approved, v2.0 edits)

Honest limit stated first: the gold set does not create signal where the corpus has none. For a classifier-blind entry it yields a low measured recall with quantified uncertainty, which proves the blind spot rather than asserting it. For a frame-blind entry it cannot even do that, because frame-blind incidents are not in the sampling base. Such entries are reported unmeasurable and routed to the staged frame-coverage audit, never assigned a low prevalence. That distinction is the scientific payoff.

The gold set is the load-bearing artifact and the project critical path. Before any engine build on real data, a staffing plan, a time budget, and a power calculation are produced and committed, and two human domain coders plus a third adjudicator are named. This is a hard pre-build gate, not a soft open question. The gold-set artifact is bound to the snapshot content hash.

Two-frame sampling, because recall cannot be estimated from the classifier's own positives. Precision frame: stratified sample of classifier-positive assignments per entry, yields per-entry false-positive rate. Recall/coverage frame: a sample drawn independently of the classifier label, stratified by `corpus_stratum`, `source_class` and confidence, oversampling rare and classifier-blind entries, low-confidence and multi-label assignments, the out-of-scope sink, and a named contamination stratum (bare `["LLM03"]` CVE-default rows). This frame estimates classifier-stage recall only; it does not estimate ingestion-frame coverage (Section 3, F-frame).

The contamination stratum is measured, not blanket-dropped. Of bare-default CVE rows, what fraction actually evidence some entry and which. This recovers true signal and gives the model a measured leak rate instead of a guess.

Coding protocol. Dual independent coding against the byte-identical frozen rubric, blind to classifier label and blind to vote. Krippendorff's alpha reported. Disagreements adjudicated on the record by a third coder (Rock or a CODEOWNER). Adjudication log is a committed artifact. Exact sample size comes from a power calculation. Order of magnitude is several hundred to about 1,000 labels, driven by the confidence-interval width needed on the rare entries, where the honest result may still be indeterminate.

Measurement model output per entry e and stratum s. Primary: per-entry, per-stratum precision and recall as Beta posteriors from the labeled counts. Upgrade and robustness: a block confusion matrix among the declared overlap clusters only, because errors flow along the boundary pairs the rubric named, not at random. Primary is the simple version so the method is not hostage to a matrix that cannot be populated.

### 5.4 Bayesian inference and vote posterior (approved, v2.0 edits)

Goal: a posterior over each measurable entry's latent prevalence, then over the ranking, accounting for classifier error, contamination, classifier-blind spots, and the mixture structure of the single channel. Frame-blind entries are censored as unmeasurable and excluded from the ranked posterior, listed explicitly in the measurability map.

Model sketch, to be confirmed by prior-predictive checks in the new repo:

- Latent incident intensity lambda_e >= 0 per measurable entry e, the quantity ranked on. lambda_e is the per-unit-stratum incident rate (dimensionless ratio); the weakly-informative HalfNormal prior on lambda_e has scale matched to the expected fraction of stratum incidents per entry (Plan 1 default 0.5, derived in `docs/METHODOLOGY-CHANGELOG.md`), not to absolute stratum size.
- Observed classified count y_{e,s} per stratum s is a noisy function of the truth through the measurement model: true positives discounted by recall r_{e,s}, false positives added through (1 - precision) leakage along the *declared overlap weights* (v2.2). The overlap weights matrix W carries W[target][source] = fraction of source's false positives landing in target; the column constraint sum_target W[target][source] ≤ 1 prevents the double-counting that a binary-indicator formulation produces when one source has multiple downstream targets. The remainder (1 - column sum) goes to out-of-scope. Contamination stratum contributes a measured leak.
- Likelihood: Negative-Binomial on counts for over-dispersion. Robustness spec: Dirichlet-Multinomial on prevalence shares. Choice documented via prior predictive checks and the data.
- Priors: weakly-informative on lambda_e scaled to stratum size. Beta priors on r_{e,s} and p_{e,s} centered on the gold-set estimates with the gold-set sample size as pseudo-counts. The gold set parameterizes these priors. That is the entire point of calibration.
- Partial identification and the never-falsely-low requirement: where the gold-set recall interval is uninformative, lambda_e must stay wide, never low. This is a release gate, not an aspiration. A pre-registered prior-predictive test on the synthetic cycle injects a low-count, recall-unknown entry and a frame-blind entry and asserts the de-biased posterior is wide (or the entry is censored unmeasurable), not falsely precise or falsely low, under the actual Beta plus pseudo-count parameterization. The build fails loudly and emits no report if this test does not pass.
- Single channel. There is no second modeled channel and no shared-lambda triangulation. Corpus B is not in the likelihood. Strata within corpus A are modeled with separate measurement parameters and a shared latent prevalence per entry, with the mixture made explicit.
- Vote-rank posterior. The vote is resampled by bootstrap over the `Raw Results (Anonymized)` respondents to produce a posterior over the vote ranking and vote tiers. The vote posterior is computed independently of the incident pipeline and joined only at `decide`.
- Temporal: recency weighting is a pre-registered primary dimension. Surface reports all windows.
- Inference: NUTS via a pinned probabilistic programming library (numpyro or PyMC, pinned version). Diagnostics gated: R-hat <= 1.01, sufficient effective sample size, zero post-warmup divergences. A run that fails diagnostics fails loudly and emits no report.
- Output: posterior draws of lambda_e for measurable entries, ranked per draw, giving a per-entry rank distribution, a full-ordering posterior, P(entry in tier), and a separate vote-rank posterior.

### 5.5 Decision layer and outputs (approved, v2.0 edits)

- Measurability map first. The report leads with the per-entry verdict: measurable, classifier-blind-but-bounded, or frame-blind-unmeasurable. This is the headline, before any concordance number.
- Vote side: CASE 2 ranking carried as a posterior (Section 5.4) mapped to pre-registered tiers (Top 5, 6 to 10, 11 to 20 for the LLM project, adjusted per the target Top 10 structure for other projects).
- Concordance headline, always published with transparency (v2.1). Posterior distribution of tier agreement (weighted Cohen's kappa and a tier confusion matrix) between vote tiers and incident tiers, integrated over both the vote posterior and the incident posterior, reported as median plus credible interval, restricted to the measurable subset. Every report publishes the headline together with: the explicit measurable-subset denominator (e.g., "kappa = 0.42 [0.18, 0.69], computed over 8 of 16 entries (50% coverage)"); the coverage ratio; the pre-registered measurability minimum (Section 9 item 8) shown as a target, with the gap (if any) shown as a tag on the headline ("below pre-registered target of 10/16"); and a standing scope-and-bias caveat ("internal triangulation against a contaminated index, not validation against reality. This concordance is computed over the measurable subset only; entries the corpus frame cannot observe or the classifier cannot recover are listed separately in the measurability map."). Suppression as a defense against misreading is rejected: it trades methodology honesty for unquotability that cannot be enforced anyway, and confuses readers about what was measured. Supporting: posterior of Spearman rho and Kendall tau, plus a permutation null answering whether observed concordance beats chance. If the measurable subset is too small for the statistic to be *meaningful* (fewer than four entries, v2.2 raised from two), the headline reports "N/A: insufficient measurable subset" with the coverage ratio shown — still transparent, still no suppression. The threshold is statistical, not mathematical: kappa is defined for n ≥ 2 but only interpretable at n ≥ 4 where the 2×2 tier confusion matrix can populate non-degenerately. Concordance uses *quadratic-weighted* Cohen's kappa across the tier structure (e.g., Top-5, 6–10, 11–20 for the LLM project), with weights w_ij = (i − j)^2 so a Top-5-vs-11-20 disagreement is penalized more than a Top-5-vs-6-10 disagreement. The tier-membership (binary) kappa degenerate case is computed separately and reported alongside, but the weighted version is the headline statistic.
- Per-entry probabilistic flag: flag entry e if P(incident tier != vote tier) > tau_flag (tau_flag pre-registered) AND the divergence direction is consistent across all robustness specs AND the entry is measurable. A wide posterior cannot exceed tau_flag, so classifier-blind and frame-blind entries are excluded by construction. Verify this in tests. Flag sensitivity decreases as the robustness-spec count grows because of the all-specs-consistent conjunction. The report discloses the robustness-spec count and this monotonic effect. A flag triggers documented working-group adjudication and never an automatic ranking change. Direction reported: vote over-ranks, vote under-ranks, or indeterminate.
- Rollup sub-test output: per rolled-up candidate, posterior of its independent incident intensity and P(it carries a distinct cluster the parent does not absorb), reported as rollup supported, contradicted, or indeterminate.
- Corpus B corroboration: the agreement between corpus A's curated-head labels and corpus B on shared incidents, reported as a declared agree/disagree artifact. Not a posterior input.
- Non-Bayesian robustness twin: the plain tiered point estimate from de-biased counts with no posterior, computed independently. If the twin and the Bayesian headline disagree in direction, that disagreement is a reported finding, not reconciled away.
- Threats-to-validity register: a shipped structured artifact enumerating every known bias (audit F1 through F6, F-frame, corpus mixture, adversarial ingestion, corpus selection bias, crosswalk single-author risk, gold-set coder subjectivity, classifier-blind and frame-blind spots, cross-year and cross-project non-comparability) with the design's mitigation or bound and the residual risk. Part of the publishable output.
- Pre-registration diff: the final report includes a pre-reg vs actual diff. Any deviation is disclosed with rationale.
- Erratum and version policy: a published cycle carries a versioned erratum policy. A post-publication corpus or code change that flips a measurability verdict or a flag is issued as a numbered erratum against that cycle's frozen inputs.
- Reproduction bundle: a single command regenerates the full report from frozen cycle inputs. Snapshot hashes, provenance, engine version, and the dependency lockfile are recorded in the report header.

---

## 6. Integrity controls (the reason the exercise exists)

1. Pre-registration committed and hash-locked before any concordance number exists: rubric, prevalence-signal primary spec, robustness spec list, flag threshold, statistic, measurability gate.
2. Vote-blindness enforced by the phase gate. Vote data is not an input to `classify`, `calibrate`, or `infer`. It enters only at `decide`. The vote-rank posterior is computed independently and joined only at `decide`.
3. Native taxonomy labels never used as a join key or ground truth.
4. One pre-registered primary specification. Everything else is a declared robustness spec. The full robustness surface is reported whole, never mined for the flattering cell. The robustness-spec count and its effect on flag sensitivity are disclosed.
5. Single-author crosswalk risk is closed, not merely disclosed: an independent OWASP working-group member who is not the ranking author signs off on the frozen rubric as a pre-registration gate. Absent that sign-off, internal runs may proceed but the report is non-publishable.
6. Public-grade supply-chain controls from the first commit: pinned dependencies with a lockfile (NIST SP 800-218), gitleaks secret scanning, semgrep SAST gate on executable additions, model weight or provenance hashing for the pinned classifier and pinned probabilistic programming library, a vulnerability disclosure stub, and a NOTICE file with attribution for vendored CC-BY-4.0 (corpus A data) and OWASP (corpus B) snapshots.
7. Frame-coverage honesty: frame-blind entries are reported unmeasurable and never flagged. Frame-coverage is raised to bounded only by the pre-registered, gated, per-entry staged audit.
8. Never-falsely-low release gate: the prior-predictive synthetic test (Section 5.4) is a build gate. Low-count or recall-unknown or frame-blind entries must yield wide or censored, never falsely low or falsely precise.
9. Snapshot integrity: gold set bound to snapshot hash, between-snapshot drift and anomaly detection with manual sign-off before a cycle consumes a drifted snapshot, and adversarial ingestion in the threats register.
10. Transparency-first publication policy (v2.1, replaces the suppression-gate model). Every cycle report publishes the headline kappa + credible interval, the explicit measurable-subset denominator, the coverage ratio, and a standing scope-and-bias caveat. The pre-registered measurability minimum is published as a target and a disclosure tag, never as a suppression gate. Suppression is rejected because it is a paternalistic defense that trades methodology honesty for unquotability the methodology cannot enforce; transparency forces the denominator into every reading instead of hiding it. The fixed scope-and-bias caveat is standing language on every report, not a flag of last resort. See §5.5 concordance-headline bullet for the report-layout contract.
11. Information-firewall discipline (v2.3). The methodology mechanically prevents information from leaking backward through the pipeline. Required controls:
    (a) Hyperparameters (lambda prior scale, NB concentration shape and rate, ESS fraction, meaningful-kappa-N threshold, PRNG seed) are committed in the prereg manifest and hash-locked before `classify`/`calibrate`/`infer`. Plan 1 ships the schema and synthetic defaults; Plan 5 pre-registers the real-cycle values.
    (b) Stage-1 classifier rules are hash-locked before the gold-set is sampled. Gold-set incidents are drawn from a partition disjoint from any incidents used during rule development; the partitioning is committed and verifiable.
    (c) Calibration uses k-fold cross-validation (k = 5). Fold variance is reported as a calibration-stability disclosure alongside the Beta posteriors. Plan 1 ships the schema and stub; Plan 4 implements.
    (d) The rubric drafting attestation records whether the rubric author viewed corpus samples before drafting, and which ones. If "yes," the report carries a "corpus-informed rubric" caveat alongside the standing scope-and-bias caveat. Schema in Plan 1; populated at Plan 3 rubric freeze.
    (e) Reviewer signoff carries `attestation_signed_at` and `viewed_results_before_signoff`. Signoff must precede the first `infer` run for the cycle. A signoff with `viewed_results_before_signoff=True` is not a clean attestation and is disclosed in the report.
    (f) Post-hoc analyses (anything not pre-registered) are logged in `post_hoc_register.json` and tagged EXPLORATORY in the report. Schema in Plan 1; register populated as cycles run.
    (g) Robustness-spec selection is itself a multiple comparison: all specs' headline statistics are reported together with the spread (max − min) as a cherry-picking-adjustment range. Selecting the most flattering spec is mechanically visible.
    (h) Selection-bias quantification: every report includes a non-parametric test of whether vote-rank distributions differ across measurability-verdict groups (Kruskal-Wallis). The verdict labels (frame-blind, classifier-blind, measurable) are nominal, not ordinal — Spearman is inappropriate (v2.4 correction). A statistically significant test (p < 0.05) means the headline kappa is computed over a vote-correlated subset; the report flags this and explains the implication.
    (i) Vote-author / corpus-author identity overlap is reported as a known confounder where identities are knowable (Plan 5+).
    The discipline is mechanism plus disclosure: where mechanism is feasible, the engine refuses to run; where it is not feasible (e.g., (i)), the artifact is disclosed in the report. Driver: Rock's directive "we must build in data leakage, confounding and overfitting firewalls throughout."

---

## 7. The standalone, reusable, dual-purpose repo

### 7.1 Name and posture

Repo name: `incident-rank-validation` (taxonomy-neutral on purpose). Private now, public-ready. Python project.

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
          prereg/             frozen rubric, specs, threshold, measurability gate, reviewer sign-off
          taxonomy/           snapshot of the 20 entry definitions
          corpora/            genai_agentic/ (per-stratum), owasp_asi/ (corroboration only) snapshots + provenance
          goldset/            stratified hand-labels + adjudication log + bound snapshot hash
          vote/               CASE 2 ranking + raw respondents, joined only at decide
          framecoverage/      optional staged audit reference list + per-entry bounds
          results/            measurability map, generated report, posterior, flag list, prereg diff
    owasp-asi/
      project.toml
      cycles/
        20xx/ ...
  tests/                      engine unit tests plus a synthetic end-to-end cycle
  pyproject.toml + lockfile   pinned dependencies
```

### 7.3 Scope boundary (YAGNI, so the build does not bloat)

Adapters are plain Python modules implementing one interface. Projects and cycles are directories. Config is one typed file per project. No plugin marketplace, no web UI, no general meta-analysis platform. Reusability means next year, same team, maybe a new adapter, a new taxonomy snapshot, or a new project. Nothing more. The staged frame-coverage audit is a declared per-entry artifact, not a general framework.

### 7.4 Dual purpose: OWASP LLM Top 10 and OWASP Agentic Top 10

The engine is taxonomy-neutral. A project is the triple (ranked taxonomy, community vote, incident corpora). `owasp-llm` is one project. `owasp-asi` is another. Switching projects changes data, never engine code.

The only Agentic-specific work when that project runs:

- Locate the authoritative OWASP Agentic (ASI) Top 10 entry definitions and the ASI community vote. These are different artifacts from the LLM ones and live in the OWASP Agentic Security Initiative material. Resolve their exact location at that time.
- The OWASP ASI Exploits and Incidents tracker is the natural curated corpus for the ASI project and is purpose-built for the Agentic Top 10.
- Real caveat to record now: for the ASI project both natural corpora are agentic-focused and may share selection bias. The single-channel plus declared-stratum design from v2.0 applies. Do not reintroduce a triangulation claim without a genuinely independent, comparable-N corpus and a measured bias-independence assessment. The same frame-coverage and measurability discipline applies: ASI entries the corpus frame cannot observe are reported unmeasurable.

Everything else (rubric, classifier, gold set, Bayesian model, decision layer, integrity controls) transfers unchanged.

### 7.5 GPU and compute posture (v2.3, amended v2.5)

Default compute is CPU. NUTS inference, prior / posterior predictive sampling, vote bootstrap, and twin computation pin `JAX_PLATFORM_NAME=cpu` and `JAX_ENABLE_X64=true` so cycle outputs are byte-reproducible across machines that share the lockfile. GPU non-determinism in cuBLAS reductions changes posteriors run-to-run and is methodology-breaking for the Bayesian inference path; using GPU there trades methodology integrity for negligible speed gains on a ~300-parameter model.

GPU is permitted *only* for Stage-2 LLM-assisted classification (§5.2). Reproducibility there is maintained via: (1) weight provenance hashing per §6.6, (2) pinned model version, (3) deterministic seed in the Stage-2 manifest, (4) batched inference with documented batch-determinism guarantees. RunPod is the default GPU provider. The per-cycle provisioning plan (`docs/PROVISIONING-PLAN.md`) commits, before any Stage-2 cycle runs: GPU count, model identity, weight hash, cost ceiling per cycle, batch size, wall-time budget, and the determinism configuration. "Provision as many of the fastest available GPUs at the time" is interpreted operationally as: maximize Stage-2 wall-time parallelism within the per-cycle cost ceiling; do *not* provision GPUs for the CPU-bound workloads (NUTS, twin, vote bootstrap) because doing so trades methodology integrity for nothing. Mid-cycle GPU scale-out is permitted; mid-cycle model swap is not.

**Provider selection rule (v2.5).** Use RunPod for any Stage-2 GPU workload that cannot complete on the local development GPU (typically Jetson AGX Orin or equivalent) in under 30 minutes wall time. Above 30 minutes, RunPod's per-iteration speed advantage (H100 / A100) dominates the upload, authentication, and weight-transfer overhead — and outweighs the per-cycle cost. Below 30 minutes, the local GPU is the faster end-to-end path. The 30-minute threshold is wall time, not compute time. Decisions are *per workload*, not per cycle: an ad-hoc adjudication batch may run local while the full Stage-2 classification cycle runs on RunPod within the same Plan 5 pass. Expected outcome for the full Stage-2 cycle on real corpus (~7,000 ambiguous incidents with a 70B-class model): >50 hours on Jetson-class hardware, so RunPod always wins. Expected outcome for ad-hoc batches (<200 incidents) or embedding-based rubric clustering: typically <30 min on Jetson, so local wins. Per-workload estimates are committed to `cycle/provenance/local_run_estimate.json` before the workload starts; a misestimate by ≥ 2× during execution requires abort + re-provision on RunPod and is logged as a post-hoc analysis per §6 control 11(f). CPU-bound workloads (NUTS, predictive sampling, twin, vote bootstrap) remain CPU-pinned regardless of provider — the 30-minute rule applies only to GPU-bound Stage-2 work.

---

## 8. What to do next, concretely

1. Repo and the Section 7.2 layout exist. Work proceeds on a feature branch with PR-only discipline (the LLM Top 10 repo convention).
2. This file is at `docs/HANDOFF.md` at version 2.0.
3. Run the Superpowers `writing-plans` skill with this document as the approved spec. The plan must sequence: engine skeleton and CLI phase gate, the canonical schema and per-stratum adapter base, the corpus A adapter with declared sub-corpus bias profiles, the rubric drafting and independent-review and adjudication workflow, the classifier with input hardening, the gold-set sampler and coding protocol with the staffing and power-calc gate, the Bayesian model with prior-predictive, never-falsely-low and blind-spot simulation gates, the vote-rank posterior, the decision layer with the measurability map and publication gate, the corpus B corroboration cross-check, the staged frame-coverage audit extension, the report and reproduction bundle, and the public-grade controls.
4. Build engine-first with the synthetic end-to-end cycle test before touching real corpus data. This protects pre-registration: the machinery and the never-falsely-low gate are proven on synthetic data before the real rubric is frozen.
5. Only then run the LLM 2026 cycle: draft rubric vote-blind, independent reviewer signs off, Rock adjudicates, freeze, classify per stratum, build gold set, infer, build vote posterior, decide, publish the measurability map and the gated concordance.

---

## 9. Open questions, now blocking gates resolved during planning (not by reopening Section 4)

1. Exact gold-set sample size and stratum allocation, from a power calculation targeting usable confidence-interval width on the rare entries. Blocking before engine build on real data.
2. Who codes the gold set (two human domain coders plus a third adjudicator), the time budget, and who is the independent rubric reviewer who is not the ranking author. Blocking before `classify`. This is the human bottleneck.
3. Bayesian likelihood family final choice (Negative-Binomial vs Dirichlet-Multinomial) decided by prior-predictive checks, with the alternative kept as a robustness spec.
4. The numeric value of tau_flag, pre-registered before `decide`.
5. The pinned classifier model with its weight or provenance hash, and the pinned probabilistic programming library and versions.
6. Whether any corpus native label is used at all as a weak classifier feature. Rock leaned toward no. Default to no unless a measured benefit is shown on the gold set.
7. The primary recency window and weighting for the temporal primary dimension.
8. The pre-registered measurability minimum (how many top-tier entries must be measurable for the aggregate concordance headline to publish).
9. The frame-coverage audit acceptance criterion: the external reference-list construction and the per-entry bound-with-uncertainty definition that upgrades an entry from unmeasurable to bounded.
10. For the ASI project later: authoritative ASI Top 10 definitions and vote location, and the corpus posture per Section 7.4.

---

## 10. Memory pointers on the authoring machine

Auto-memory for the GenAI-LLM-Top10 project records the v1.0 work at:
`~/.claude/projects/-Users-klambros-github-projects-GenAI-LLM-Top10/memory/`

- `incident-dataset-measurement-weaknesses.md`: the corpus A weaknesses in Section 3.
- `ranking-validation-methodology.md`: the converged v1.0 design summary and a pointer to this file.

These are project-scoped to the LLM Top 10 repo and predate v2.0. The new repo relies on this file at version 2.0 as the single source of truth, not on that machine-local memory. Section 11 is the authoritative record of the v1.0 to v2.0 changes.

---

## 11. Premortem remediation log (v1.0 to v2.0)

Source: adversarial premortem, 2026-05-19. Each row is an approved change. Decision owner: Rock Lambros, 2026-05-19.

| Change | Premortem finding closed | v1.0 text superseded |
|---|---|---|
| Item 0, staged. Purpose rescoped to measurable entries only, frame-blind entries reported unmeasurable, pre-registered gated per-entry frame-coverage audit added as a declared extension. | F1.1 (ingestion-frame not identifiable from within the corpus), F5.1 (metric does not map to outcome), F5.3 (opportunity cost), F1.4 (taxonomy-frame circularity). | §4 Purpose and Frame-coverage-audit rows, §3 consequence, §5.3 honest-limit paragraph. |
| Corpus B demoted from triangulation channel to qualitative corroboration of the curated head. A-head vs B-head agreement reported as a declared artifact. | Triangulation degeneracy at N≈dozens, residual triangulation-honesty risk. | §4 Calibration and Corpus B role rows, §5.4 single-channel bullet, §5.5 corpus B corroboration bullet. |
| Vote-side posterior. Vote resampled by bootstrap over respondents, concordance integrates over both posteriors. | F1.2 (vote modeled as error-free, asymmetric rigor). | §5.5 concordance, §4 decision-rule and Vote-measurement rows, §2. |
| Independent rubric reviewer (non-author OWASP WG member) sign-off as a pre-registration gate. Absent it, runs are non-publishable. | F5.2 (single-author crosswalk, author = ranking author). | §4 Crosswalk-authorship row, §6 control 5, §5.2. |
| Per-sub-corpus bias profiles and per-stratum measurement parameters. Corpus A treated as a mixture. | F1.3 (undeclared genre mixture as one instrument). | §5.1 corpus-adapter paragraph, §4 Corpus-A-is-a-mixture row, §3 mixture note. |
| Report leads with the measurability map. Aggregate concordance headline gated by a pre-registered measurability minimum, else the fixed caveat. | F5.1, F5.4 (misleading authoritative headline under indeterminacy). | §5.5 concordance-headline bullet, §6 control 10. |
| Pre-registered never-falsely-low prior-predictive release gate on the synthetic cycle. | F2.1 (guarantee in tension with the Beta plus pseudo-count mechanism). | §5.4 partial-identification bullet, §6 control 8. |
| Temporal promoted to a pre-registered primary dimension. Future-dated rows dropped or repaired in the adapter. | F1.5 (temporal heterogeneity treated as a robustness afterthought). | §4 Temporal row, §5.4 temporal bullet. |
| Adversarial ingestion added to the threats register. Between-snapshot drift and anomaly detection with manual sign-off. Gold set bound to snapshot hash. | F3.1 (external poisoning of a public auto-refresh), F4.2 (gold-set to snapshot binding). | §5.1 snapshotting paragraph, §6 control 9. |
| Stage-2 classifier instruction and data separation with delimiter fencing. Model weight or provenance hashing added to the supply-chain controls. | F3.2 (prompt injection of the classifier via incident text), F3.3 (model-weight supply chain). | §5.2 classifier paragraph, §6 control 6. |
| Gold-set staffing, time budget, and power calculation made a hard pre-build gate. Section 9 reclassified from soft questions to blocking gates. Erratum and version policy and flag-sensitivity disclosure added. | F4.1 (unstaffed critical path), F4.3 (no erratum path), F2.3 (flag de-sensitization undisclosed). | §5.3 gold-set paragraph, §5.5 erratum and flag bullets, §9 heading. |
| v2.0 → v2.1 (2026-05-19): transparency-first publication policy replaces the suppression-gate model. Concordance headline always published with explicit measurable-subset denominator, coverage ratio, pre-registered-minimum tag, and standing scope-and-bias caveat. Suppression rejected as a paternalistic defense that traded methodology honesty for unquotability the methodology could not actually enforce. Driver: Rock's directive "don't suppress, be transparent" after the Plan 1 v2 adversarial premortem surfaced publication-gate political risk (premortem Round 5 finding 5.1). | F5.4 (misleading authoritative headline under indeterminacy) closed differently — via mandatory denominator disclosure rather than headline suppression. The reader can no longer miss the scope. | §5.5 concordance-headline bullet, §6 control 10. |
| v2.1 → v2.2 (2026-05-19): Premortem 2 closure across R1–R33. Most material: (1) leakage matrix becomes weighted (column-stochastic), eliminating multi-target double-counting; (2) NB concentration prior changed from Exponential(1) to Gamma(5, 0.1), restoring weakly-informative-toward-Poisson behavior; (3) PRIOR_SCALE = 0.5 derived from rate-per-unit-stratum interpretation, not from an arbitrary divisor; (4) kappa N/A threshold raised from 2 to 4 measurable entries; (5) weighted Cohen's kappa is quadratic, not binary; (6) reviewer signoff is attested artifact, not boolean; (7) drift detector's requires_signoff flag is enforced at the CLI; (8) lock + git-attestation enforced at infer and decide; (9) lockfile hash captured at infer time, not at report time; (10) cross-cycle comparisons refused by lineage check; (11) erratum module shipped; (12) Stage-2 injection fixture in tests/security; (13) CODEOWNERS for security configs; (14) external reviewers (rubric + statistical) identified at Plan 1 acceptance. | F-author residual reduced: signoff is attested, not declared. F-circ standing caveat is in machine-readable artifacts too. Cross-cycle non-comparability now mechanically enforced, not just documented. | §5.4 leakage bullet + lambda bullet, §5.5 kappa bullet, §6 control 5. |
| v2.4 → v2.5 (2026-05-20): GPU provider-selection rule made explicit. RunPod by default for any Stage-2 GPU workload that cannot complete on the local Jetson GPU in under 30 minutes wall time; below 30 minutes the local Jetson wins on end-to-end latency. Per-workload decision, not per-cycle. Workload estimates committed to `cycle/provenance/local_run_estimate.json`; ≥ 2× misestimate during execution forces abort + re-provision + post-hoc analysis log. CPU-bound workloads (NUTS, twin, predictive, vote bootstrap) remain CPU-pinned unchanged. Driver: Rock 2026-05-20 — "Use RUNPOD GPU where possible unless the we can run on local GPU on the jetson in under 30 min. Be explicit." | F-defenseindepth re-engaged for GPU-policy decisions: misestimates are loud (post-hoc-logged), not silent. | §7.5 provider-selection-rule paragraph. |
| v2.3 → v2.4 (2026-05-20): Premortem 3 closure (M1–M23). Most material: (1) selection-bias statistic switched from Spearman ρ (ordinal-on-nominal misuse) to Kruskal-Wallis (correct for nominal verdict labels); (2) two-cycle parity holdout added to Plan 5 publication prereqs (30-day reviewer audit before any external sharing); (3) `OverlapWeights` self-loop rejection; (4) post-hoc register Merkle-chain integrity; (5) `signed_at` derived from `git log` of the attestation file (mechanical for timing, discipline retained for content); (6) hyperparameters fully sourced from project.toml + a `synthetic-stress` project added to exercise multi-tier quadratic kappa, the N/A branch, and untuned-hyperparameter behavior in CI; (7) drift-signoff length floor + persisted rationale; (8) cross-platform output-diff CI job; (9) quadratic kappa zero-denominator fix; (10) F-defenseindepth threat surfaced. Driver: Rock-authorized Premortem 3 (2026-05-20). | F-leak strengthened with mechanism for (a)(e)(g)(h); F-confound disclosure tightened via Kruskal-Wallis; F-defenseindepth newly added. | §6 control 11(h), §11. |
| v2.2 → v2.3 (2026-05-19): information-firewall discipline (§6 control 11) and GPU / compute posture (§7.5) added. Closes leakage / confounding / overfitting gaps across the pipeline: (a) hyperparameter pre-registration (lambda prior, NB concentration, ESS fraction, meaningful-kappa-N, PRNG seed); (b) classifier-rule hash-lock before gold-set draw; (c) k-fold calibration CV with fold-variance disclosure; (d) rubric drafting attestation; (e) reviewer signoff `signed_at` + `viewed_results_before_signoff`; (f) post-hoc analysis register with EXPLORATORY tags; (g) robustness-spec cherry-picking spread disclosure; (h) selection-bias quantification (measurability ↔ vote-rank Spearman); (i) vote-author / corpus-author overlap as a disclosed confounder. GPU posture: CPU pinned for inference/predictive/twin/bootstrap (reproducibility); GPU permitted only for Stage-2 LLM classification with `docs/PROVISIONING-PLAN.md` committed before Plan 5 cycle execution. Driver: Rock's directive 2026-05-19 — "we must build in data leakage, confounding and overfitting firewalls throughout" + "anywhere where dedicated GPU will work, we must provision as many of the fastest runpod gpus available at the time to run." | F-leak (information leakage backward through pipeline) closed via mechanisms (a)–(h); F-confound (vote/corpus author overlap) reduced via disclosure (i). | §6 control 11, §7.5. |

Residual risk accepted, recorded for the threats register: taxonomy-frame circularity (F1.4) cannot be removed, only reframed. The published artifact must state it is internal triangulation against a contaminated index, not validation against reality (now as standing language per v2.1, not a flag of last resort). Corpus B contributes no triangulation power by design. It is corroboration only. If the staged frame-coverage audit is never built, the high-salience entries stay unmeasurable, which is the honest outcome, not a failure.
