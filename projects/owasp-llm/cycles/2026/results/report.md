# Cycle Report: 2026
Engine version: 1.2.0
**STATUS: NON-PUBLISHABLE** (single-author rubric, uncontrolled)

## Measurability Map
Coverage ratio: 85.00%
Measurable: LLM01, LLM02, LLM03, LLM05, LLM06, LLM07, LLM09, NEW-ITSCD, NEW-MA, NEW-MSDA, NEW-MTIE, NEW-PMP, NEW-WLA, ROLL-CFAS, ROLL-CMSB, ROLL-LAPTF, ROLL-SICG
Classifier-blind: none
Frame-blind: LLM04, LLM08, LLM10

## Concordance
Weighted kappa: 0.20 [-0.16, 0.57]
Computed over 17 of 20 entries (85% coverage)

> Internal triangulation against a contaminated index, not validation against reality. This concordance is computed over the measurable subset only; entries the corpus frame cannot observe or the classifier cannot recover are listed separately in the measurability map.

## Selection Bias
Statistic: kruskal_wallis_h
H = 0.5507, p = 0.4580
Severity: low

## Flags
- LLM01: P(tier mismatch) = 0.87, direction = vote_over_ranks
- LLM09: P(tier mismatch) = 0.99, direction = vote_under_ranks
- NEW-MTIE: P(tier mismatch) = 0.83, direction = vote_over_ranks
- NEW-PMP: P(tier mismatch) = 0.92, direction = vote_over_ranks
- NEW-WLA: P(tier mismatch) = 0.84, direction = vote_under_ranks

No deviations from pre-registration.

## Corpus B Corroboration
Declared qualitative artifact — NOT a posterior input (HANDOFF §4, §5.4).

Corpus B incidents: 46. Shared with corpus A: 46.
Label agreement on shared incidents: 12 agree, 34 disagree (rate = 26%).

Context: cycle headline kappa = 0.275. Agreement reporting at N = 46 is qualitative, not statistical.

Note: 3 entries are frame-blind (LLM04, LLM08, LLM10). Agreement on incidents classified to these entries is reported but has no bearing on posterior estimates.

## Threats to Validity
- **F1-ingestion-frame**: corpus sampling frame is blind to incidents that never become CVE/GHSA/OSV entries or harm-database rows
- **F2-default-seed-contamination**: CVE ingest seeds every entry with LLM03; bare-default labels are contamination
- **F3-classifier-blind-spots**: classifier is a heuristic with measured error, not a source of truth
- **F4-structural-blind-spots**: owasp_llm distribution reflects classifier ruleset, not threat landscape
- **F5-circular-atlas**: MITRE ATLAS labels are backfill from OWASP, not independent
- **F-frame**: corpus A built by CVE/GHSA/OSV keyword crawl; deployed-app failures outside these sources never enter the sampling base
- **F-circ**: taxonomy-frame circularity: measuring a taxonomy against incidents classified by that taxonomy
- **F-adversarial-ingestion**: public CVE/GHSA/OSV are open submission surfaces; descriptions are attacker-controlled; infer_attack_vector is pure regex
- **F-defenseindepth**: the engine has many integrity controls; this can create false confidence that all bugs are upstream of the engine when debugging unexpected results
- **F-aiharm-precision**: ai-harm stratum has no direct precision measurements; ai-harm precision keys are absent from posteriors.json entirely, so the model falls back to Beta(1,1) = Uniform(0,1) via the default initialization in inference.py (apply_empirical_precision_prior cannot reach keys that do not exist)

---
Before publishing externally, verify against `docs/REVIEWERS.md` PRE-PUBLISH CHECKLIST. This report is internal-only unless the checklist passes.
